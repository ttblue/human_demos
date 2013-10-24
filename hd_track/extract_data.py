import numpy as np, numpy.linalg as nlg
import rospy
import os, os.path as osp
import cPickle

import roslib
roslib.load_manifest('ar_track_service')
from ar_track_service.srv import MarkerPositions, MarkerPositionsRequest, MarkerPositionsResponse
from ar_track_alvar.msg import AlvarMarkers

from hd_utils.defaults import tfm_link_rof
from hd_utils.colorize import *
from hd_utils import ros_utils as ru, clouds, conversions

from hd_calib import gripper_calibration, gripper

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
    ar2_tfms = []
    ar2_count = 0
    hyd_tfms = []
    hyd_count = 0
    pot_angles = []
    pot_count = 0
    
    yellowprint('Camera1')
    for (topic, msg, _) in bag.read_messages(topics=['/camera1/depth_registered/points']):
        marker_poses = get_ar_marker_poses (msg)
        
        if marker_poses:
            for m in marker_poses:
                marker_poses[m] = np.array(marker_poses[m])
            tt_tfm = gr.get_tool_tip_transform(marker_poses, None)
            
            if tt_tfm is not None:
                stamp = msg.header.stamp.to_sec()
                blueprint("Got markers " + str(marker_poses.keys()) + " at time %f"%stamp)
                ar1_tfms.append((tt_tfm, stamp))
                ar1_count += 1
        
    yellowprint('Camera2')
    for (topic, msg, _) in bag.read_messages(topics=['/camera2/depth_registered/points']):
        marker_poses = get_ar_marker_poses (msg)
        
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


    for (_, msg, ts) in bag.read_messages(topics=['/pot_angle']):

	angle = msg.data
	stamp = ts.to_sec()
	pot_angles.append((angle, stamp))
	blueprint("Got a potentiometer angle of %f at time %f"%(angle, stamp))
	pot_count += 1


    yellowprint("Found %i transforms from camera1"%ar1_count)
    yellowprint("Found %i transforms from camera2"%ar2_count)
    yellowprint("Found %i transforms from hydra"%hyd_count)
    yellowprint("Found %i potentiometer readings"%pot_count)
    
    if save_file is None:
        save_file = ''
        for name in bag.filename.split('.')[0:-1]:
            save_file += name + '.'
        save_file += 'data'
    save_filename = osp.join('/home/sibi/sandbox/human_demos/hd_data/demos/obs_data', save_file)
    
    data = {'camera1':ar1_tfms, 'camera2':ar2_tfms, 
	    'hydra':hyd_tfms, 'pot_angles':pot_angles}
    with open(save_filename, 'w') as sfh: cPickle.dump(data, sfh)
