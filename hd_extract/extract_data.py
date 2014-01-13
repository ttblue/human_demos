import numpy as np, numpy.linalg as nlg
import rospy
import os, os.path as osp
import cPickle
import cv2, cv
import yaml


import rosbag
import roslib
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse,\
                MarkerImagePositions, MarkerImagePositionsRequest, MarkerImagePositionsResponse,\
                SetCalibInfo, SetCalibInfoRequest, SetCalibInfoResponse
from ar_track_alvar.msg import AlvarMarkers
roslib.load_manifest('cv_bridge')
from cv_bridge import CvBridge, CvBridgeError


from hd_utils.defaults import tfm_link_rof, asus_xtion_pro_f, demo_files_dir, demo_names
from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions, extraction_utils as eu, utils

from hd_calib import gripper_lite
from hd_calib.calibration_pipeline import gripper_trans_marker_tooltip

getMarkersPC = None
getImageMarkers = None
setCalib = None
reqPC = MarkerPositionsRequest()
reqImage = MarkerImagePositionsRequest()
reqCalib = SetCalibInfoRequest()
bridge = None

displayImages = False
verbose = True

def get_ar_marker_poses (msg, ar_markers = None, use_pc_service=True, track=False):
    '''
    get poses according to ar_markers
    if ar_markers == None, then for all ar markers appeared in the point cloud
    '''
    global getMarkersPC, getImageMarkers, reqPC, reqImage, bridge
    
    if rospy.get_name() == '/unnamed':
        rospy.init_node('ar_marker_poses', anonymous=True)
    
    if use_pc_service:
        if getMarkersPC is None:
            getMarkersPC = rospy.ServiceProxy("getMarkers", MarkerPositions)
        reqPC.pc = msg
        reqPC.track = track
        res = getMarkersPC(reqPC)
    else:
        if getImageMarkers is None:
            getImageMarkers = rospy.ServiceProxy("getImageMarkers", MarkerImagePositions)
        if bridge is None:
            bridge = CvBridge()
    
        img = bridge.cv_to_imgmsg(msg)
        reqImage.img = img
        reqImage.track = track
        try:
            res = getImageMarkers(reqImage)
        except Exception as e:
            redprint("Something went wrong with image" )
            print e
            return {}
            

    marker_tfm = {}
    for marker in res.markers.markers:
        if ar_markers == None or marker.id in ar_markers:
            marker_tfm[marker.id] = conversions.pose_to_hmat(marker.pose.pose).tolist()
    
    return marker_tfm



def save_observations_rgbd(demo_type, demo_name, save_file=None):
    """
    Extract data from all sensors and store into demo.data file.
    """
    global setCalib
    # Temp file to show that data is already being extracted
    demo_dir        = osp.join(demo_files_dir, demo_type, demo_name)
    with open(osp.join(demo_dir, demo_names.extract_data_temp),'w') as fh: fh.write('Extracting...')
    
    calib_file_path = osp.join(demo_dir, demo_names.calib_name)
    bag_file        = osp.join(demo_dir, demo_names.bag_name)
    
    with open(osp.join(demo_dir, demo_names.camera_types_name)) as fh:
        camera_types = yaml.load(fh)
    with open(osp.join(demo_dir, demo_names.camera_models_name)) as fh:
        camera_models = yaml.load(fh)
    
    num_cameras = len(camera_types)

    video_dirs = {}
    for i in range(1, num_cameras + 1):
        video_dirs[i] = osp.join(demo_dir, demo_names.video_dir%i)

    c_frames = {}
    for i in range(1, num_cameras + 1):
        c_frames[i]= 'camera%i_link'%(i)
    hydra_frame = 'hydra_base'
    
    tfm_c1 = {i:None for i in range (1,num_cameras+1)}
    tfm_c1[1] = np.eye(4)
    tfm_c1_h = None

    with open(calib_file_path,'r') as fh: calib_data = cPickle.load(fh)
    bag = rosbag.Bag(bag_file)
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c_frames[1] or tfm['parent'] == '/' + c_frames[1]:
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])
            else:
                for i in range(2, num_cameras+1):
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
    for i in range(1,num_cameras+1):
        if verbose:
            yellowprint('Camera%i'%i)
        if camera_types[i] == "rgbd":
            rgb_fnames, depth_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i])
        else:
            rgb_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i], depth = False)
        cam_count = len(stamps)
        cam_counts.append(cam_count)

        if camera_types[i] == "rgbd":
            for ind in rgb_fnames:
                rgb = cv2.imread(rgb_fnames[ind])
                
                if displayImages:
                    cv2.imshow(winname, rgb)
                    cv2.waitKey(1)

                assert rgb is not None
                depth = cv2.imread(depth_fnames[ind],2)
                assert depth is not None
                xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
                pc = ru.xyzrgb2pc(xyz, rgb, frame_id='', use_time_now=False)
                ar_tfms = get_ar_marker_poses(pc,track=True)
                if ar_tfms:
                    if verbose:
                        blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps[ind])

                    for lr,gr in grippers.items():
                        ar = gr.get_ar_marker() 
                        if ar in ar_tfms:
                            tt_tfm = gr.get_tooltip_transform(ar, np.asarray(ar_tfms[ar]))
                            data[lr]['camera%i'%i].append((tfm_c1[i].dot(tt_tfm),stamps[ind]))
        else:
            if setCalib is None: 
                setCalib = rospy.ServiceProxy("setCalibInfo", SetCalibInfo)
            reqCalib.camera_model = camera_models[i]
            setCalib(reqCalib)
            
            if verbose:
                yellowprint("Changed camera calibration parameters to model %s"%camera_models[i])

            for ind in rgb_fnames:                
                rgb = cv.LoadImage(rgb_fnames[ind])
                
                if displayImages:
                    cv.ShowImage(winname, rgb)
                    cv.WaitKey(1)

                assert rgb is not None
                ar_tfms = get_ar_marker_poses(rgb,use_pc_service=False,track=True)

                if ar_tfms:
                    if verbose:
                        blueprint("Got markers " + str(ar_tfms.keys()) + " at time %f"%stamps[ind])
    
                    for lr,gr in grippers.items():
                        ar = gr.get_ar_marker() 
                        if ar in ar_tfms:
                            tt_tfm = gr.get_tooltip_transform(ar, np.asarray(ar_tfms[ar]))
                            data[lr]['camera%i'%i].append((tfm_c1[i].dot(tt_tfm),stamps[ind]))


    found_hydra_data = False
    found_pot_data = False
    # If hydra_only.data file already exists, don't do extra work.
    hydra_data_file        = osp.join(demo_dir, demo_names.hydra_data_name)
    if osp.isfile(hydra_data_file):
        yellowprint ("%s already exists for %s. Getting hydra data from this info."%(demo_names.hydra_data_name,demo_name))
        with open(hydra_data_file,'r') as fh: hdata = cPickle.load(fh)
         
        for lr in hdata:
            if lr in 'lr':
                if hdata[lr].get('hydra'):
                    data[lr]['hydra'] = hdata[lr]['hydra']
                    found_hydra_data = True
                if hdata[lr].get('pot_angles'):
                    data[lr]['pot_angles'] = hdata[lr]['pot_angles']
                    found_pot_data = True


    if not found_hydra_data:
        if verbose:
            yellowprint('Hydra')
        lr_long = {'l':'left','r':'right'}
        for (_, msg, _) in bag.read_messages(topics=['/tf']):
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
            if found and verbose:
                blueprint("Got hydra readings %s at time %f"%(found,stamp))

    if not found_pot_data:
        if verbose:
            yellowprint('Potentiometer readings')
        for lr in grippers:
            for (_, msg, ts) in bag.read_messages(topics=['/%s_pot_angle'%lr]):
                angle = msg.data
                stamp = ts.to_sec()
                data[lr]['pot_angles'].append((angle, stamp))
                if verbose: 
                    blueprint("Got a %s potentiometer angle of %f at time %f"%(lr,angle, stamp))

    if verbose:
        for lr in 'lr':
            yellowprint("Gripper %s:"%lr)
            for i in range(num_cameras):
                yellowprint("Found %i transforms out of %i rgb/rgbd images from camera%i"%(len(data[lr]['camera%i'%(i+1)]), cam_counts[i], i+1))
            
            yellowprint("Found %i transforms from hydra"%len(data[lr]['hydra']))
            yellowprint("Found %i potentiometer readings"%len(data[lr]['pot_angles']))

    if save_file is None:
        save_file = demo_names.data_name
    save_filename = osp.join(demo_dir, save_file)

    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
    yellowprint("Saved %s."%demo_names.data_name)
    
    os.remove(osp.join(demo_dir, demo_names.extract_data_temp))


def save_hydra_only (demo_type, demo_name, save_file=None):
    """
    Save hydra only data.
    """
    # Temp file to show that data is already being extracted
    demo_dir        = osp.join(demo_files_dir, demo_type, demo_name)
    with open(osp.join(demo_dir, demo_names.extract_hydra_data_temp),'w') as fh: fh.write('Extracting...')
    
    bag_file        = osp.join(demo_dir, demo_names.bag_name)
    calib_file_path = osp.join(demo_dir, demo_names.calib_name)
    
    data = {}

    c1_frame = 'camera1_link'
    hydra_frame = 'hydra_base'
    
    tfm_c1_h = None

    with open(calib_file_path,'r') as fh: calib_data = cPickle.load(fh)
    bag = rosbag.Bag(bag_file)
    
    for tfm in calib_data['transforms']:
        if tfm['parent'] == c1_frame or tfm['parent'] == '/' + c1_frame:
            if tfm['child'] == hydra_frame or tfm['child'] == '/' + hydra_frame:
                tfm_c1_h = nlg.inv(tfm_link_rof).dot(tfm['tfm'])

    if tfm_c1_h is None:
        redprint("Calibration does not have hydra transform.")
        return
    
    data['T_cam2hbase'] = tfm_c1_h

    # If demo.data file already exists, don't do extra work.
    found_hydra_data = False
    found_pot_data = False
    all_data_file        = osp.join(demo_dir, demo_names.data_name)
    if osp.isfile(all_data_file):
        yellowprint ("%s already exists for %s. Creating hydra file from this info."%(demo_names.data_name,demo_name))
        with open(all_data_file,'r') as fh: all_data = cPickle.load(fh)
        
        for lr in all_data:
            if lr in ['lr']:
                data[lr] = {}
                if all_data[lr].get('hydra'):
                    data[lr]['hydra'] = all_data[lr]['hydra']
                    found_hydra_data = True
                if all_data[lr].get('pot_angles'):
                    data[lr]['pot_angles'] = all_data[lr]['pot_angles']
                    found_pot_data = True

        
        if found_hydra_data and found_pot_data:
            if save_file is None:
                save_file = demo_names.hydra_data_name
            save_filename = osp.join(demo_dir, save_file)
            with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
            return
    
    if not calib_data.get('grippers'):
        redprint("Gripper not found.")
        return    
    grippers = {}
    for lr,gdata in calib_data['grippers'].items():
        gr = gripper_lite.GripperLite(lr, gdata['ar'], trans_marker_tooltip=gripper_trans_marker_tooltip[lr])
        gr.reset_gripper(lr, gdata['tfms'], gdata['ar'], gdata['hydra'])
        grippers[lr] = gr
        data[lr] ={'hydra':[],
                   'pot_angles':[],
                   'T_tt2hy': gr.get_rel_transform('tool_tip', gr.hydra_marker)}

    if not found_hydra_data:
        if verbose:
            yellowprint('Hydra')
        lr_long = {'l':'left','r':'right'}
        for (_, msg, _) in bag.read_messages(topics=['/tf']):
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
            if found and verbose:
                blueprint("Got hydra readings %s at time %f"%(found,stamp))

    if not found_pot_data:
        if verbose:
            yellowprint('Potentiometer readings')
        for lr in grippers:
            for (_, msg, ts) in bag.read_messages(topics=['/%s_pot_angle'%lr]):
                angle = msg.data
                stamp = ts.to_sec()
                data[lr]['pot_angles'].append((angle, stamp))
                if verbose: 
                    blueprint("Got a %s potentiometer angle of %f at time %f"%(lr,angle, stamp))

    if verbose:
        for lr in 'lr':
            yellowprint("Gripper %s:"%lr)            
            yellowprint("Found %i transforms from hydra"%len(data[lr]['hydra']))
            yellowprint("Found %i potentiometer readings"%len(data[lr]['pot_angles']))

    if save_file is None:
        save_file = demo_names.hydra_data_name
    save_filename = osp.join(demo_dir, save_file)

    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
    yellowprint("Saved %s."%demo_names.hydra_data_name)
    
    os.remove(osp.join(demo_dir, demo_names.extract_hydra_data_temp))
    

def save_init_ar(demo_type, demo_name, ar_marker, save_file=None):
    """
    Extracts the initializing ar marker transform and stores it.
    """    
    # Temp file to show that data is already being extracted
    global setCalib
    demo_dir        = osp.join(demo_files_dir, demo_type, demo_name)
    calib_file_path = osp.join(demo_dir, demo_names.calib_name)
    
    with open(osp.join(demo_dir, demo_names.camera_types_name)) as fh:
        camera_types = yaml.load(fh)
    with open(osp.join(demo_dir, demo_names.camera_models_name)) as fh:
        camera_models = yaml.load(fh)
    
    cameras = range(1,len(camera_types)+1)
    c_frames = {}
    for i in cameras:
        c_frames[i]= 'camera%i_link'%(i)

    video_dirs = {}
    for i in cameras:
        video_dirs[i] = osp.join(demo_dir, demo_names.video_dir%i)
    
    tfm_c1 = {i:None for i in cameras}
    tfm_c1[1] = np.eye(4)

    with open(calib_file_path,'r') as fh: calib_data = cPickle.load(fh)

    for tfm in calib_data['transforms']:
        if tfm['parent'] == c_frames[1] or tfm['parent'] == '/' + c_frames[1]:
            for i in cameras:
                if tfm['child'] == c_frames[i] or tfm['child'] == '/' + c_frames[i]:
                    tfm_c1[i] = nlg.inv(tfm_link_rof).dot(tfm['tfm']).dot(tfm_link_rof)

    if not all([tfm_c1[s] != None for s in tfm_c1]):
        redprint("Calibration does not have required transforms")
        return

    # We want the ar marker transform from all the relevant cameras
    data = {'tfms':{i:None for i in cameras}, 'marker':ar_marker}

    for i in cameras:
        if verbose:
            yellowprint('Camera%i'%i)
        if camera_types[i] == "rgbd":
            rgb_fnames, depth_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i])
        else:
            rgb_fnames, stamps = eu.get_rgbd_names_times(video_dirs[i], depth = False)

        all_tfms = []
        cam_count = 0
        if camera_types[i] == "rgbd":
            for ind in rgb_fnames:
                rgb = cv2.imread(rgb_fnames[ind])
                
                if displayImages:
                    cv2.imshow(winname, rgb)
                    cv2.waitKey(1)

                assert rgb is not None
                depth = cv2.imread(depth_fnames[ind],2)
                assert depth is not None
                xyz = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
                pc = ru.xyzrgb2pc(xyz, rgb, frame_id='', use_time_now=False)
                ar_tfms = get_ar_marker_poses(pc,track=True)
                if ar_tfms and ar_marker in ar_tfms:
                    all_tfms.append(tfm_c1[i].dot(ar_tfms[ar_marker]))
                    cam_count += 1
            
        else:
            if setCalib is None: 
                setCalib = rospy.ServiceProxy("setCalibInfo", SetCalibInfo)
            reqCalib.camera_model = camera_models[i]
            setCalib(reqCalib)

            if verbose:
                yellowprint("Changed camera calibration parameters to model %s"%camera_models[i])

            for ind in rgb_fnames:                
                rgb = cv.LoadImage(rgb_fnames[ind])
                
                if displayImages:
                    cv.ShowImage(winname, rgb)
                    cv.WaitKey(1)

                assert rgb is not None
                ar_tfms = get_ar_marker_poses(rgb,use_pc_service=False,track=True)
                if ar_tfms and ar_marker in ar_tfms:
                    all_tfms.append(tfm_c1[i].dot(ar_tfms[ar_marker]))
                    cam_count += 1
        
        if verbose:
            blueprint ("Got %i values for AR Marker %i from camera %i."%(cam_count, ar_marker, i))
        if cam_count > 0:
            data['tfms'][i] = utils.avg_transform(all_tfms)

    if save_file is None:
        save_file = demo_names.init_ar_marker_name
    save_filename = osp.join(demo_dir, save_file)

    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
    yellowprint("Saved %s."%save_file)
