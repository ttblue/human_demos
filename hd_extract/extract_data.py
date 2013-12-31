import numpy as np, numpy.linalg as nlg
import rospy
import os, os.path as osp
import cPickle
import cv2
import yaml

import rosbag
import roslib
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse,\
                MarkerImagePositions, MarkerImagePositionsRequet, MarkerImagePositionsResponse

from ar_track_alvar.msg import AlvarMarkers

from hd_utils.defaults import tfm_link_rof, asus_xtion_pro_f, demo_files_dir
from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions, extraction_utils as eu

from hd_calib import gripper_calibration, gripper, gripper_lite
from hd_utils.defaults import calib_files_dir, demo_files_dir
from hd_calib.calibration_pipeline import gripper_marker_id, gripper_trans_marker_tooltip

getMarkersPC = None
getImageMarkers = None
reqPC = MarkerPositionsRequest()
reqImage = MarkerImagePositionsRequest()

def get_ar_marker_poses (msg, ar_markers = None, use_pc_service=True):
    '''
    get poses according to ar_markers
    if ar_markers == None, then for all ar markers appeared in the point cloud
    '''
    global getMarkers, reqPC, reqImage
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('ar_marker_poses')
    
    if use_pc_service:
        if getMarkers is None:
            getMarkers = rospy.ServiceProxy("getMarkers", MarkerPositions)
        reqPC.pc = msg
        res = getMarkers(reqPC)
    else:
        if getImageMarkers is None:
            getImageMarkers = rospy.ServiceProxy("getImageMarkers", MarkerImagePositions)
        reqImage.img = msg
        res = getMarkers(reqImage)

    marker_tfm = {}
    for marker in res.markers.markers:
        if ar_markers == None or marker.id in ar_markers:
            marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose).tolist()
    
    #print "Marker ids found: ", marker_tfm.keys()
    
    return marker_tfm



def save_observations_rgbd(demo_type, demo_name, calib_file, num_cameras, for_gpr=False, save_file=None):
    
    demo_dir        = osp.join(demo_files_dir, demo_type, demo_name)
    calib_file_path = osp.join(calib_files_dir, calib_file)
    bag_file        = osp.join(demo_dir, 'demo.bag')
    
    with open(osp.join(demo_dir, "camera_types.yaml")) as fh:
        camera_types = yaml.load(fh)


    video_dirs = []
    for i in range(1, num_cameras + 1):
        video_dirs.append(osp.join(demo_dir, 'camera_#%i'%(i)))

    c_frames = []
    for i in range(1, num_cameras + 1):
        c_frames.append('camera%i_link'%(i))

    hydra_frame = 'hydra_base'
    
    tfm_c1 = {i:None for i in range (1,num_cameras+1)}
    tfm_c1[1] = np.eye(4)
    tfm_c1_h = None

    with open(calib_file_path,'r') as fh: calib_data = cPickle.load(fh)
    bag = rosbag.Bag(bag_file)
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c_frames[0] or tfm['parent'] == '/' + c_frames[0]:
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
            else:
                for i in range(1, num_cameras):
                    if tfm['child'] == c_frames[i] or tfm['child'] == '/' + c_frames[i]:
                        tfm_c1[i] = nlg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)

    if tfm_c1_h is None or not all([tfm_c1[s] != None for s in tfm_c1]):
        redprint("Calibration does not have required transforms")
        return

    if not calib_data.get('grippers'):
        redprint("Gripper not found.")
        return

    grippers = {}
    data = {}
    data['T_cam2hbase'] = tfm_c1_h
    for lr,gdata in calib_data['grippers'].items():
        gr = gripper_lite.GripperLite(lr, gdata['ar'], trans_marker_tooltip=gripper_trans_marker_tooltip[lr])
        gr.reset_gripper(lr, gdata['tfms'], gdata['ar'], gdata['hydra'])
        grippers[lr] = gr
        data[lr] ={'hydra':[],
                   'pot_angles':[],
                   'T_tt2hy': gr.get_rel_transform('tool_tip', gr.hydra_marker)}
        ## place-holder for AR marker transforms:
        for i in xrange(num_cameras):
            data[lr]['camera%d'%(i+1)] = []

    winname = 'cam_image'    
    cam_counts = []
    for i in range(num_cameras):
        yellowprint('Camera%i'%(i+1))
        if camera_types[i+1] == "kinect":
            rgb_fnames, depth_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i])
        else:
            rgb_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i], depth = False)
        cam_count = len(stamps)
        cam_counts.append(cam_count)

        if camera_types[i+1] == "kinect":
            for ind in rgb_fnames:
                rgb = cv2.imread(rgb_fnames[ind])
                cv2.imshow(winname, rgb)
                cv2.waitKey(1)
                assert rgb is not None
                depth = cv2.imread(depth_fnames[ind],2)
                assert depth is not None
                xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
                pc = ru.xyzrgb2pc(xyz, rgb, frame_id='', use_time_now=False)
                ar_tfms = get_ar_marker_poses(pc)
                if ar_tfms:
                    blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps[ind])
    
                for lr,gr in grippers.items():
                    ar = gr.get_ar_marker() 
                    if ar in ar_tfms:
                        tt_tfm = gr.get_tooltip_transform(ar, np.asarray(ar_tfms[ar]))
                        data[lr]['camera%i'%(i+1)].append((tfm_c1[i].dot(tt_tfm),stamps[ind]))
        else:
            for ind in rgb_fnames:
                rgb = cv2.imread(rgb_fnames[ind])
                cv2.imshow(winname, rgb)
                cv2.waitKey(1)
                assert rgb is not None
                ar_tfms = get_ar_marker_poses(rgb,use_pc_service=False)
                if ar_tfms:
                    blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps[ind])
    
                for lr,gr in grippers.items():
                    ar = gr.get_ar_marker() 
                    if ar in ar_tfms:
                        tt_tfm = gr.get_tooltip_transform(ar, np.asarray(ar_tfms[ar]))
                        data[lr]['camera%i'%(i+1)].append((tfm_c1[i].dot(tt_tfm),stamps[ind]))

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
                    tt_tfm = grippers[lr].get_tooltip_transform(lr_long[lr], hyd_tfm)
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

    for lr in 'lr':
        yellowprint("Gripper %s:"%lr)
        for i in range(num_cameras):
            yellowprint("Found %i transforms out of %i point clouds from camera%i"%(len(data[lr]['camera%i'%(i+1)]), cam_counts[i], i+1))
        
        yellowprint("Found %i transforms from hydra"%len(data[lr]['hydra']))
        yellowprint("Found %i potentiometer readings"%len(data[lr]['pot_angles']))

    if save_file is None:
        save_file = "demo.data"
    save_filename = osp.join(demo_dir, save_file)

    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)