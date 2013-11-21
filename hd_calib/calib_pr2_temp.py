#!/usr/bin/env python
import rospy
import numpy as np, numpy.linalg as nlg
import openravepy
import roslib
#roslib.load_manifest('calib_pr2')
roslib.load_manifest('tf')
import tf;
import time
from hd_utils import conversions
from hd_utils.colorize import colorize
import argparse


T_h_k = np.array([[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
 [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
 [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
 [ 0.,          0.,          0.,          1.        ]])

f = 544.260779961

def get_kinect_transform(robot):
    T_w_h = robot.GetLink("head_plate_frame").GetTransform()
    T_w_k = T_w_h.dot(T_h_k)
    return T_w_k

def avg_quaternions(qs):
    """
    Returns the "average" quaternion of the quaternions in the list qs.
    ref: http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf
    """
    M = np.zeros((4,4))
    for q in qs:
        q = q.reshape((4,1))
        M = M + q.dot(q.T)

    l, V = np.linalg.eig(M)
    q_avg =  V[:, np.argmax(l)]
    return q_avg/np.linalg.norm(q_avg)

def get_tool_model(n_tfm, n_avg):

    listener = tf.TransformListener()

    gripper_frame = 'l_gripper_tool_frame'
    base_frame = 'base_link'
    model_frame = ''
    depth_frame = 'camera_depth_optical_frame'
    head_frame = 'head_plate_frame'
    model_tfms = []
    tool_tfms = []
    

    for i in xrange(n_tfm):
        raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms"%(i + 1, n_tfm), "red", True))
        tool_trans_list = np.empty((0, 3))
        tool_quat_list = np.empty((0, 4))
        marker_trans_list = np.empty((0, 3))
        marker_quat_list = np.empty((0, 4))
        head_trans_list = np.empty((0, 3))
        head_quat_list = np.empty((0, 4))
	time.sleep(3)
	for j in xrange(n_avg):

            tool_trans, tool_quat, model_trans, model_quat, head_trans, head_quat = None, None, None, None, None, None

            while tool_trans == None:
                #model_trans, model_quat = listener.lookupTransform(base_frame, model_frame, rospy.Time(0))
                tool_trans, tool_quat = listener.lookupTransform(base_frame, gripper_frame, rospy.Time(0))
                head_trans, head_quat = listener.lookupTransform(base_frame, head_frame, rospy.Time(0))

            tool_trans_list = np.r_[tool_trans_list, np.array(tool_trans, ndmin=2)]
            tool_quat_list = np.r_[tool_quat_list, np.array(tool_quat, ndmin=2)]
            #model_trans_list = np.r_[model_trans_list, np.array(model_trans, ndmin=2)]
            #model_quat_list = np.r_[model_quat_list, np.array(model_quat, ndmin=2)]
            head_trans_list = np.r_[head_trans_list, np.array(head_trans, ndmin=2)]
            head_quat_list = np.r_[head_quat_list, np.array(head_quat, ndmin=2)]
        
        tool_trans_avg = np.sum(tool_trans_list, axis=0) / n_avg
        tool_quat_avg = avg_quaternions(tool_quat_list)
        #model_trans_avg = np.sum(model_trans_list, axis=0) / n_avg
        #model_quat_avg = avg_quaternions(model_quart_list)
        head_trans_avg = np.sum(head_trans_list, axis=0) / n_avg
        head_quat_avg = avg_quaternions(head_quat_list)
    
        tool_tfm = conversions.trans_rot_to_hmat(tool_trans_avg, tool_quat_avg)

        head_tfm = conversions.trans_rot_to_hmat(head_trans_avg, head_quat_avg)
        #depth_to_model_tfm = conversions.trans_rot_to_hmat(model_trans_avg, model_quat_avg)
        depth_tfm = head_tfm.dot(T_h_k)
        #model_tfm = depth_tfm.dot(depth_to_model_tfm)
        
        tool_tfms.append(tool_tfm)
        #model_tfms.append(model_tfm)
    #return tool_tfms, model_tfms
    return tool_tfms

        

def publish_tf(T, from_frame, to_frame):
    print colorize("Transform : ", "yellow", True)
    print T
    tf_pub = tf.TransformBroadcaster()
    trans = T[0:3,3]
    rot = tf.transformations.quaternion_from_matrix(T)
    sleeper = rospy.Rate(100)

    while True:
        tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
        sleeper.sleep()


if __name__ == '__main__':

    rospy.init_node('calib_pr2') 

    parser = argparse.ArgumentParser(description="PR2 Tool Frame and Model Gripper Calibration")
    parser.add_argument('--n_tfm', help="number of transforms to use for calibration.", type=int, default=5)
    parser.add_argument('--n_avg', help="number of estimates of  transform to use for averaging.", type=int, default=100)
    parser.add_argument('--publish_tf', help="whether to publish the transform between hydra_base and camera_link", default=True)
    vals = parser.parse_args()


    #tool_tfms, model_tfms = get_tool_model(vals.n_tfm, vals.n_avg)
    tool_tfms = get_tool_model(vals.n_tfm, vals.n_avg)
    
    tool_to_model_tfms = []
    for i in xrange(vals.n_tfm):
        tool = tool_tfms[i]
        model = model_tfms[i]
        tool_inv = nlg.inv(tool)
        tool_to_model_tfm = tool_inv.dot(model)
        tool_to_model_tfms.append(tool_to_model_tfm)
    
    trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in tool_to_model_tfms]
    trans = np.asarray([trans for (trans, rot) in trans_rots])
    avg_trans = np.sum(trans,axis=0)/trans.shape[0]
    rots = [rot for (trans, rot) in trans_rots]
    avg_rot = avg_quaternions(np.array(rots))

    tool_to_model_tfm_avg = conversions.trans_rot_to_hmat(avg_trans, avg_rot)
    print tool_to_model_tfm_avg


    if vals.publish_tf:
       publish_tf(tool_to_model_tfm_avg, 'l_gripper_tool_frame', 'gripper_model_frame')





