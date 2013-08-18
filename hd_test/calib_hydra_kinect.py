import roslib
roslib.load_manifest('calib_hydra_kinect')
import rospy
import tf; roslib.load_manifest('tf')

import argparse
import numpy as np
from colorize import colorize



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


def get_transforms(marker, hydra, n_tfm , n_avg):
    """
    Returns a tuple of two list of N_TFM transforms, each
    the average of N_AVG transforms
    corresponding to the AR marker with ID = MARKER and 
    the hydra paddle of the HYDRA ('right' or 'left') side.
    """

    camera_frame = 'camera_link'
    hydra_frame  = 'hydra_base'
    marker_frame = 'ar_marker_%d'%marker    
    assert hydra=='right' or hydra=='left'
    paddle_frame = 'hydra_%s_pivot'%hydra

    tf_sub = tf.TransformListener()

    
    ar_tfms    = []
    hydra_tfms = []
    
    
    for i in xrange(n_tfm+1):
        raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms"%(i, n_tfm), "red", True))
        
        ## transforms which need to be averaged.
        m_ts = np.empty((0, 3))
        m_qs = np.empty((0,4))
        h_ts = np.empty((0, 3))
        h_qs = np.empty((0,4))
        
        for j in xrange(n_avg):            
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)
            
            mtrans, mrot, htrans, hrot = None, None, None, None
            while htrans == None:
                now = rospy.Time.now()
                #tf_sub.waitForTransform(camera_frame, marker_frame, now)
                mtrans, mrot = tf_sub.lookupTransform(camera_frame, marker_frame, rospy.Time(0))
                try:
                    tf_sub.waitForTransform(hydra_frame, paddle_frame, now, rospy.Duration(0.05))
                    (htrans,hrot) = tf_sub.lookupTransform(hydra_frame, paddle_frame, rospy.Time(0))
                except (tf.LookupException):
                    continue
    
            m_ts = np.r_[m_ts, np.array(mtrans, ndmin=2)]
            h_ts = np.r_[h_ts, np.array(htrans, ndmin=2)]
            m_qs = np.r_[m_qs, np.array(mrot, ndmin=2)]
            h_qs = np.r_[h_qs, np.array(hrot, ndmin=2)]
        
        mtrans_avg = np.sum(m_ts, axis=0) / n_avg
        mrot_avg   = avg_quaternions(m_qs)
        htrans_avg = np.sum(h_ts, axis=0) / n_avg
        hrot_avg   = avg_quaternions(h_qs)
        
        ar_tfms.append(tf_sub.fromTranslationRotation(mtrans_avg, mrot_avg))
        hydra_tfms.append(tf_sub.fromTranslationRotation(htrans_avg, hrot_avg))
        
    return (ar_tfms, hydra_tfms)



if __name__ == '__main__':
    rospy.init_node('calib_hydra_kinect')
    
    parser = argparse.ArgumentParser(description="Hydra Kinect Calibration")
    parser.add_argument('--marker', help="AR marker id to track.", required=True, type=int)  
    parser.add_argument('--hydra', help="in {'right', 'left'} : the hydra handle to track.", required=True)
    parser.add_argument('--n_tfm', help="number of transforms to use for calibration.", type=int, default=5)
    parser.add_argument('--n_avg', help="number of estimates of  transform to use for averaging.", type=int, default=5)
    parser.add_argument('--publish_tf', help="whether to publish the transform between hydra_base and camera_link", default=True)
    vals = parser.parse_args()

    ar_tfms, hydra_tfms = get_transforms(vals.marker, vals.hydra, vals.n_tfm, vals.n_avg)
    print "AR Transforms: ", ar_tfms
    print "Hydra Transforms: ", hydra_tfms
