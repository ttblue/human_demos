import numpy as np, numpy.linalg as nlg
import argparse

import roslib, rospy
roslib.load_manifest('tf')
import tf

from hd_utils import conversions, utils
from hd_utils.defaults import tfm_link_rof
from hd_calib.cameras import RosCameras
import hd_calib.camera_calibration as cc

def find_transform (tfms1, tfms2):
    """
    Finds transform between cameras from latest stored observation.
    """
    ar1, ar2 = cc.common_ar_markers(tfms1, tfms2)
    if not ar1 or not ar2:
        return None
    
    if len(ar1.keys()) == 1:
        transform = ar1.values()[0].dot(nlg.inv(ar2.values()[0]))
    else:
        transform = cc.find_rigid_tfm(cc.convert_hmats_to_points(ar1.values()),
                                      cc.convert_hmats_to_points(ar2.values()))

    return transform



def quick_calibrate(NUM_CAM, N_AVG):

    
    rospy.init_node('quick_calibrate')
    
    cameras = RosCameras(NUM_CAM)
    tfm_pub = tf.TransformBroadcaster()
    
    frames = {i:'/camera%i_link'%(i+1) for i in xrange(NUM_CAM)}
    
    calib_tfms = {(frames[0],frames[i]):None for i in xrange(1,NUM_CAM)}
    
    sleeper = rospy.Rate(30)
    
    try:
        while True:
            tfms_found = {i:{} for i in xrange(NUM_CAM)}
            for _ in xrange(N_AVG):
                for i in xrange(NUM_CAM):
                    mtfms = cameras.get_ar_markers(camera=i)
                    for m in mtfms:
                        if m not in tfms_found[i]:
                            tfms_found[i][m] = []
                        tfms_found[i][m].append(mtfms[m])
            
            for i in tfms_found:
                for m in tfms_found[i]:
                    tfms_found[i][m] = utils.avg_transform(tfms_found[i][m])        

            for i in xrange(1,NUM_CAM):
                rof_tfm = find_transform(tfms_found[0], tfms_found[i])
                if rof_tfm is not None:
                    calib_tfms[(frames[0], frames[i])] = tfm_link_rof.dot(rof_tfm).dot(nlg.inv(tfm_link_rof))

            for parent, child in calib_tfms:
                if calib_tfms[parent, child] is not None:
                    trans, rot = conversions.hmat_to_trans_rot(calib_tfms[parent, child])
                    tfm_pub.sendTransform(trans, rot,
                                          rospy.Time.now(),
                                          child, parent)
                    
                    print child, parent, trans, rot
                sleeper.sleep()

    except KeyboardInterrupt:
        print "got ctrl-c"
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cameras", help="number of cameras", default=2, type=int)
    parser.add_argument("--num_average", help="number of avg operations", default=5, type=int)
    args = parser.parse_args()

    quick_calibrate(args.num_cameras, args.num_average)