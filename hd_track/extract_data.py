import numpy as np, numpy.linalg as nlg
import rospy
import os, os.path as osp
import cPickle
import cv2

import rosbag
import roslib
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
from ar_track_alvar.msg import AlvarMarkers

from hd_utils.defaults import tfm_link_rof, asus_xtion_pro_f
from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions, extraction_utils as eu

from hd_calib import gripper_calibration, gripper, gripper_lite

getMarkers = None
req = MarkerPositionsRequest()

def get_ar_marker_poses (pc):
    global getMarkers, req
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('keypoints')
    
    if getMarkers is None:
        getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
    
    req.pc = pc
    
    marker_tfm = {}
    res = getMarkers(req)
    for marker in res.markers.markers:
        marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose).tolist()
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    return marker_tfm

    
def save_observations (bag, calib_file, save_file=None):
    """
    Assuming that the bag files have specific topics.
    Also assuming the calib has only one gripper.
    Can add more later.
    """
    
    file_name = osp.join('/home/sibi/sandbox/human_demos/hd_data/calib',calib_file)
    with open(file_name,'r') as fh: calib_data = cPickle.load(fh)
    
    c1_frame = 'camera1_link'
    c2_frame = 'camera2_link'
    hydra_frame = 'hydra_base'
    tfm_c1_c2 = None
    tfm_c1_h = None
    
#     for (topic, msg, _) in bag.read_messages(topics=['/tf']):
#         for tfm in msg.transforms:
#             if tfm.header.frame_id == '/' + c1_frame and tfm.child_frame_id == '/'+c2_frame:
#                 if tfm_c1_c2 is not None:
#                     t,r = tfm.transform.translation, tfm.transform.rotation
#                     trans = (t.x,t.y,t.z)
#                     rot = (r.x, r.y, r.z, r.w)
#                     t_found = conversions.trans_rot_to_hmat(trans, rot)
#                     tfm_c1_c2  = nlg.inv(tfm_link_rof).dot(t_found).dot(tfm_link_rof)
#             
#             elif tfm.header.frame_id == '/' + c1_frame and tfm.child_frame_id == '/'+hydra_frame:
#                 if tfm_c1_h is not None:
#                     t,r = tfm.transform.translation, tfm.transform.rotation
#                     trans = (t.x,t.y,t.z)
#                     rot = (r.x, r.y, r.z, r.w)
#                     t_found = tfm_c1_h.dot(conversions.trans_rot_to_hmat(trans, rot))
#                     tfm_c1_h  = nlg.inv(tfm_link_rof).dot(t_found)
#         
#         if tfm_c1_c2 is not None and tfm_c1_h is not None:
#             break
#         
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c1_frame or tfm['parent'] == '/' + c1_frame:
            if tfm['child'] == c2_frame or tfm['child'] == '/' + c2_frame:
                tfm_c1_c2 = nlg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
    
    if tfm_c1_c2 is None or tfm_c1_h is None:
        redprint("Calibration does not have required transforms")
        return
    
    if not calib_data.get('grippers'):
        redprint("Gripper not found.")
        return

    lr,graph = calib_data['grippers'].items()[0]
    gr = gripper.Gripper(lr, graph)
    assert 'tool_tip' in gr.mmarkers
    gr.tt_calculated = True
    
    ar1_tfms = []
    ar1_count = 0
    cam1_count = 0
    ar2_tfms = []
    ar2_count = 0
    cam2_count = 0
    hyd_tfms = []
    hyd_count = 0
    pot_angles = []
    pot_count = 0
    
    yellowprint('Camera1')
    for (topic, msg, _) in bag.read_messages(topics=['/camera1/depth_registered/points','camera1/depth_registered/points']):
        print topic
        marker_poses = get_ar_marker_poses (msg)
        cam1_count += 1
        if marker_poses:
            for m in marker_poses:
                marker_poses[m] = np.array(marker_poses[m])
            tt_tfm = gr.get_tool_tip_transform(marker_poses, None)
            
            if tt_tfm is not None:
                stamp = msg.header.stamp.to_sec()
                blueprint("Got markers " + str(marker_poses.keys()) + " at time %f"%stamp)
                ar1_tfms.append((tt_tfm, stamp))
                ar1_count += 1
                
        print cam1_count, ar1_count
        
    yellowprint('Camera2')
    for (topic, msg, _) in bag.read_messages(topics=['/camera2/depth_registered/points','camera2/depth_registered/points']):
        marker_poses = get_ar_marker_poses (msg)
        cam2_count += 1
        if marker_poses:
            for m in marker_poses:
                marker_poses[m] = tfm_c1_c2.dot(np.array(marker_poses[m]))
            
            tt_tfm = gr.get_tool_tip_transform(marker_poses, None)
            if tt_tfm is not None:
                stamp = msg.header.stamp.to_sec()
                blueprint("Got markers " + str(marker_poses.keys()) + " at time %f"%stamp)
                ar2_tfms.append((tt_tfm, stamp))
                ar2_count += 1

    yellowprint('Hydra')
    for (topic, msg, _) in bag.read_messages(topics=['/tf']):
        
        hyd_tfm = {}
        for tfm in msg.transforms:
            if tfm.header.frame_id == '/' + hydra_frame and tfm.child_frame_id == '/hydra_left':
                t,r = tfm.transform.translation, tfm.transform.rotation
                trans = (t.x,t.y,t.z)
                rot = (r.x, r.y, r.z, r.w)
                hyd_tfm['left'] = tfm_c1_h.dot(conversions.trans_rot_to_hmat(trans, rot))
                stamp = tfm.header.stamp.to_sec()
                break
        
        if hyd_tfm:
            tt_tfm = gr.get_tool_tip_transform(hyd_tfm, None)
            
            if tt_tfm is not None:
                blueprint("Got hydra_left at time %f"%stamp)  
                hyd_tfms.append((tt_tfm, stamp))
                hyd_count += 1


    for (_, msg, ts) in bag.read_messages(topics=['/l_pot_angle']):
        angle = msg.data
        stamp = ts.to_sec()
        pot_angles.append((angle, stamp))
        blueprint("Got a potentiometer angle of %f at time %f"%(angle, stamp))
        pot_count += 1
        
    if pot_count == 0:
        for (_, msg, ts) in bag.read_messages(topics=['/pot_angle']):
            angle = msg.data
            stamp = ts.to_sec()
            pot_angles.append((angle, stamp))
            blueprint("Got a potentiometer angle of %f at time %f"%(angle, stamp))
            pot_count += 1
                

    yellowprint("Found %i transforms out of %i point clouds from camera1"%(ar1_count, cam1_count))
    yellowprint("Found %i transforms out of %i point clouds from camera2"%(ar2_count, cam2_count))
    yellowprint("Found %i transforms from hydra"%hyd_count)
    yellowprint("Found %i potentiometer readings"%pot_count)
    
    if save_file is None:
        bag_name = osp.basename(bag.filename)
        save_file = ''
        for name in bag_name.split('.')[0:-1]:
            save_file += name + '.'
        save_file += 'data'
    save_filename = osp.join('/home/sibi/sandbox/human_demos/hd_data/demos/obs_data', save_file)
    
    data = {'camera1':ar1_tfms, 'camera2':ar2_tfms, 
	    'hydra':hyd_tfms, 'pot_angles':pot_angles}
    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)


def save_observations_one_camera (bag, calib_file, save_file=None):
    """
    Assuming that the bag files have specific topics.
    Also assuming the calib has only one gripper.
    Can add more later.
    """
    
    file_name = osp.join('/home/sibi/sandbox/human_demos/hd_data/calib',calib_file)
    with open(file_name,'r') as fh: calib_data = cPickle.load(fh)
    
    c1_frame = 'camera1_link'
    hydra_frame = 'hydra_base'
    tfm_c1_h = None
    
#     for (topic, msg, _) in bag.read_messages(topics=['/tf']):
#         for tfm in msg.transforms:
#             if tfm.header.frame_id == '/' + c1_frame and tfm.child_frame_id == '/'+c2_frame:
#                 if tfm_c1_c2 is not None:
#                     t,r = tfm.transform.translation, tfm.transform.rotation
#                     trans = (t.x,t.y,t.z)
#                     rot = (r.x, r.y, r.z, r.w)
#                     t_found = conversions.trans_rot_to_hmat(trans, rot)
#                     tfm_c1_c2  = nlg.inv(tfm_link_rof).dot(t_found).dot(tfm_link_rof)
#             
#             elif tfm.header.frame_id == '/' + c1_frame and tfm.child_frame_id == '/'+hydra_frame:
#                 if tfm_c1_h is not None:
#                     t,r = tfm.transform.translation, tfm.transform.rotation
#                     trans = (t.x,t.y,t.z)
#                     rot = (r.x, r.y, r.z, r.w)
#                     t_found = tfm_c1_h.dot(conversions.trans_rot_to_hmat(trans, rot))
#                     tfm_c1_h  = nlg.inv(tfm_link_rof).dot(t_found)
#         
#         if tfm_c1_c2 is not None and tfm_c1_h is not None:
#             break
#         
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c1_frame or tfm['parent'] == '/' + c1_frame:
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
    

    if not calib_data.get('grippers'):
        redprint("Gripper not found.")
        return

    lr,graph = calib_data['grippers'].items()[0]
    gr = gripper.Gripper(lr, graph)
    assert 'tool_tip' in gr.mmarkers
    gr.tt_calculated = True
    
    ar1_tfms = []
    ar1_count = 0
    cam1_count = 0
    hyd_tfms = []
    hyd_count = 0
    pot_angles = []
    pot_count = 0
    
    yellowprint('Camera1')
    for (topic, msg, _) in bag.read_messages(topics=['/camera1/depth_registered/points','camera1/depth_registered/points']):
        marker_poses = get_ar_marker_poses (msg)
        cam1_count += 1
        if marker_poses:
            for m in marker_poses:
                marker_poses[m] = np.array(marker_poses[m])
            tt_tfm = gr.get_tool_tip_transform(marker_poses, None)
            
            if tt_tfm is not None:
                stamp = msg.header.stamp.to_sec()
                blueprint("Got markers " + str(marker_poses.keys()) + " at time %f"%stamp)
                ar1_tfms.append((tt_tfm, stamp))
                ar1_count += 1
        
    yellowprint('Hydra')
    for (topic, msg, _) in bag.read_messages(topics=['/tf']):
        
        hyd_tfm = {}
        for tfm in msg.transforms:
            if tfm.header.frame_id == '/' + hydra_frame and tfm.child_frame_id == '/hydra_left':
                t,r = tfm.transform.translation, tfm.transform.rotation
                trans = (t.x,t.y,t.z)
                rot = (r.x, r.y, r.z, r.w)
                hyd_tfm['left'] = tfm_c1_h.dot(conversions.trans_rot_to_hmat(trans, rot))
                stamp = tfm.header.stamp.to_sec()
                break
        
        if hyd_tfm:
            tt_tfm = gr.get_tool_tip_transform(hyd_tfm, None)
            
            if tt_tfm is not None:
                blueprint("Got hydra_left at time %f"%stamp)  
                hyd_tfms.append((tt_tfm, stamp))
                hyd_count += 1


    for (_, msg, ts) in bag.read_messages(topics=['/l_pot_angle']):
        angle = msg.data
        stamp = ts.to_sec()
        pot_angles.append((angle, stamp))
        blueprint("Got a potentiometer angle of %f at time %f"%(angle, stamp))
        pot_count += 1
        
    if pot_count == 0:
        for (_, msg, ts) in bag.read_messages(topics=['/pot_angle']):
            angle = msg.data
            stamp = ts.to_sec()
            pot_angles.append((angle, stamp))
            blueprint("Got a potentiometer angle of %f at time %f"%(angle, stamp))
            pot_count += 1


    yellowprint("Found %i transforms out of %i point clouds from camera1"%(ar1_count, cam1_count))
    yellowprint("Found %i transforms from hydra"%hyd_count)
    yellowprint("Found %i potentiometer readings"%pot_count)
    
    if save_file is None:
        bag_name = osp.basename(bag.filename)
        save_file = ''
        for name in bag_name.split('.')[0:-1]:
            save_file += name + '.'
        save_file += 'data'
    save_filename = osp.join('/home/sibi/sandbox/human_demos/hd_data/demos/obs_data', save_file)
    
    data = {'camera1':ar1_tfms, 'hydra':hyd_tfms, 'pot_angles':pot_angles}
    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
    

def save_observations_rgbd(demo_name, calib_file, save_file=None):
    
    demo_dir = osp.join('/home/sibi/sandbox/human_demos/hd_data/demos',demo_name)
    calib_file_path = osp.join('/home/sibi/sandbox/human_demos/hd_data/calib',calib_file)
    
    bag_file = osp.join(demo_dir, 'demo.bag')
    rgbd1_dir = osp.join(demo_dir, 'camera_#1')
    rgbd2_dir = osp.join(demo_dir, 'camera_#2')

    c1_frame = 'camera1_link'
    c2_frame = 'camera2_link'
    hydra_frame = 'hydra_base'
    tfm_c1_c2 = None
    tfm_c1_h = None

    with open(calib_file_path,'r') as fh: calib_data = cPickle.load(fh)
    bag = rosbag.Bag(bag_file)
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c1_frame or tfm['parent'] == '/' + c1_frame:
            if tfm['child'] == c2_frame or tfm['child'] == '/' + c2_frame:
                tfm_c1_c2 = nlg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
    
    if tfm_c1_c2 is None or tfm_c1_h is None:
        redprint("Calibration does not have required transforms")
        return

    if not calib_data.get('grippers'):
        redprint("Gripper not found.")
        return

    grippers = {}
    data = {}
    for lr,gdata in calib_data['grippers'].items():
        gr = gripper_lite.GripperLite(lr,gdata['ar'])
        gr.reset_gripper(lr, gdata['tfms'], gdata['ar'], gdata['hydra'])
        grippers[lr] = gr
        data[lr] ={'camera1':[],
                   'camera2':[],
                   'hydra':[],
                   'pot_angles':[]}
        
    winname = 'abcd'
    yellowprint('Camera1')
    rgbs1fnames, depths1fnames, stamps1 = eu.get_rgbd_names_times(rgbd1_dir)
    cam1_count = len(stamps1)
    for ind in rgbs1fnames:
        rgb = cv2.imread(rgbs1fnames[ind])
        cv2.imshow(winname, rgb)
        cv2.waitKey(1)
        assert rgb is not None
        depth = cv2.imread(depths1fnames[ind],2)
        assert depth is not None
        xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
        pc = ru.xyzrgb2pc(xyz, rgb, frame_id='', use_time_now=False)
        ar_tfms = get_ar_marker_poses(pc)
        if ar_tfms:
            blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps1[ind])
        for lr,gr in grippers.items():
            ar = gr.get_ar_marker() 
            if ar in ar_tfms:
                tt_tfm = gr.get_tool_tip_transform(ar, np.asarray(ar_tfms[ar]))
                data[lr]['camera1'].append((tt_tfm,stamps1[ind]))
        
    yellowprint('Camera2')
    rgbs2fnames, depths2fnames, stamps2 = eu.get_rgbd_names_times(rgbd2_dir)
    cam2_count = len(stamps2)
    for ind in rgbs2fnames:
        rgb = cv2.imread(rgbs2fnames[ind])
        cv2.imshow(winname, rgb)
        cv2.waitKey(1)
        assert rgb is not None
        depth = cv2.imread(depths2fnames[ind],2)
        assert depth is not None

        xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
        pc = ru.xyzrgb2pc(xyz, rgb, frame_id='', use_time_now=False)
        ar_tfms = get_ar_marker_poses(pc)
        if ar_tfms:
            blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps2[ind])
        for lr,gr in grippers.items():
            ar = gr.get_ar_marker()
            if ar in ar_tfms:
                tt_tfm = gr.get_tool_tip_transform(ar, np.asarray(ar_tfms[ar]))
                data[lr]['camera2'].append((tfm_c1_c2.dot(tt_tfm),stamps2[ind]))


    yellowprint('Hydra')
    lr_long = {'l':'left','r':'right'}
    for (topic, msg, _) in bag.read_messages(topics=['/tf']):
        hyd_tfm = {}
        found = ''
        for tfm in msg.transforms:
            if found in ['lr','rl']:
                break
            for lr in grippers:
                if lr in found:
                    continue
                elif tfm.header.frame_id == '/' + hydra_frame and tfm.child_frame_id == '/hydra_'+lr_long[lr]:
                    t,r = tfm.transform.translation, tfm.transform.rotation
                    trans = (t.x,t.y,t.z)
                    rot = (r.x, r.y, r.z, r.w)
                    hyd_tfm = tfm_c1_h.dot(conversions.trans_rot_to_hmat(trans, rot))
                    stamp = tfm.header.stamp.to_sec()
                    tt_tfm = grippers[lr].get_tool_tip_transform(lr_long[lr], hyd_tfm)
                    data[lr]['hydra'].append((tt_tfm, stamp))
                    found += lr
        if found:
            blueprint("Got hydra readings %s at time %f"%(found,stamp))

    yellowprint('Potentiometer readings')
    for lr in grippers:
        for (_, msg, ts) in bag.read_messages(topics=['/%s_pot_angle'%lr]):
            angle = msg.data
            stamp = ts.to_sec()
            data[lr]['pot_angles'].append((angle, stamp))
            blueprint("Got a %s potentiometer angle of %f at time %f"%(lr,angle, stamp))

    for lr in data:
        yellowprint("Gripper %s:"%lr)
        yellowprint("Found %i transforms out of %i point clouds from camera1"%(len(data[lr]['camera1']), cam1_count))
        yellowprint("Found %i transforms out of %i point clouds from camera2"%(len(data[lr]['camera2']), cam2_count))
        yellowprint("Found %i transforms from hydra"%len(data[lr]['hydra']))
        yellowprint("Found %i potentiometer readings"%len(data[lr]['pot_angles']))

    if save_file is None:
        save_file = "demo.data"
    save_filename = osp.join(demo_dir, save_file)

    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
