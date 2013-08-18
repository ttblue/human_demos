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
                #now = rospy.Time.now()
                #tf_sub.waitForTransform(camera_frame, marker_frame, now)
                mtrans, mrot = tf_sub.lookupTransform(camera_frame, marker_frame, rospy.Time(0))
                (htrans,hrot) = tf_sub.lookupTransform(hydra_frame, paddle_frame, rospy.Time(0))
                #try:
                #    tf_sub.waitForTransform(hydra_frame, paddle_frame, now, rospy.Duration(0.05))
                #except (tf.LookupException):
                #    continue
    
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



def solve_sylvester (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    s1_t0_inv = np.linalg.inv(tfms1[0])
    s2_t0_inv = np.linalg.inv(tfms2[0])
    
    for i in range(1,len(tfms1)):
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]),I)
        M_final = np.r_[M_final, M]
    
    # add the constraints on the last row of the transformation matrix to be == [0,0,0,1]
    for i in [3,7,11,15]:
        t = np.zeros((1,16))
        t[0,i] = 1
        M_final = np.r_[M_final,t]
    L_final = np.zeros(M_final.shape[0])
    L_final[-1] = 1

    X = np.linalg.lstsq(M_final,L_final)[0]
    print M_final.dot(X)
    tt = np.reshape(X,(4,4),order='F')
    return tt


def publish_tf(T, from_frame, to_frame):
    print colorize("Transform : ", "yellow", True)
    print T
    tf_pub = tf.TransformBroadcaster()
    trans = T[0:3,3] 
    rot   = tf.transformations.quaternion_from_matrix(T)
    sleeper = rospy.Rate(100)

    while True:
        tf_pub.sendTransform(trans, rot, rospy.Time.now(), to_frame, from_frame)
        sleeper.sleep()


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
    #print "AR Transforms: ", ar_tfms
    #print "Hydra Transforms: ", hydra_tfms
    
    if vals.publish_tf:
        T_ms = solve_sylvester(ar_tfms, hydra_tfms)
        T_ch = ar_tfms[0].dot(T_ms).dot(np.linalg.inv(hydra_tfms[0]))
        publish_tf(T_ch, 'camera_link', 'hydra_base')                                
        #publish_tf(, 'ar_marker_%d'%vals.marker, 'hydra_%s_pivot'%vals.hydra)
