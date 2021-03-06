"""
Calibrates the pr2's hand with the hydra : similar to hydra--ar marker calibration.
"""

from __future__ import division

import roslib
#roslib.load_manifest('calib_hydra_pr2')
import rospy
roslib.load_manifest('tf')
import tf; 

import argparse
import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

from hd_utils.colorize import colorize
from hd_utils import conversions

np.set_printoptions(precision=5, suppress=True)

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


def get_transforms(arm, hydra, n_tfm , n_avg):
    """
    Returns a tuple of two list of N_TFM transforms, each
    the average of N_AVG transforms
    corresponding to the left/ right arm of the pr2 and 
    the hydra paddle of the HYDRA ('right' or 'left') side.
    """

    pr2_frame = 'base_footprint'
    assert arm=='right' or 'left'
    arm_frame  = '%s_gripper_tool_frame' % {'right':'r', 'left':'l'}[arm]

    hydra_frame  = 'hydra_base'
    assert hydra=='right' or hydra=='left'
    paddle_frame = 'hydra_%s'%hydra

    tf_sub = tf.TransformListener()

    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    arm_tfms   = []
    hydra_tfms = []

    for i in xrange(n_tfm+1):
        raw_input(colorize("Transform %d of %d : Press return when ready to capture transforms"%(i, n_tfm), "red", True))
        
        ## transforms which need to be averaged.
        p_ts = np.empty((0, 3))
        p_qs = np.empty((0,4))
        h_ts = np.empty((0, 3))
        h_qs = np.empty((0,4))

        sleeper = rospy.Rate(30)        
        for j in xrange(n_avg):            
            print colorize('\tGetting averaging transform : %d of %d ...'%(j,n_avg-1), "blue", True)

            ptrans, prot = tf_sub.lookupTransform(pr2_frame, arm_frame, rospy.Time(0))
            htrans, hrot = tf_sub.lookupTransform(hydra_frame, paddle_frame, rospy.Time(0))

            p_ts = np.r_[p_ts, np.array(ptrans, ndmin=2)]
            h_ts = np.r_[h_ts, np.array(htrans, ndmin=2)]
            p_qs = np.r_[p_qs, np.array(prot, ndmin=2)]
            h_qs = np.r_[h_qs, np.array(hrot, ndmin=2)]
            sleeper.sleep()
          
        ptrans_avg = np.sum(p_ts, axis=0) / n_avg
        prot_avg   = avg_quaternions(p_qs)
        htrans_avg = np.sum(h_ts, axis=0) / n_avg
        hrot_avg   = avg_quaternions(h_qs)
          
        a_tfm = conversions.trans_rot_to_hmat(ptrans_avg,prot_avg)
        h_tfm = conversions.trans_rot_to_hmat(htrans_avg,hrot_avg)
               
        arm_tfms.append(a_tfm)
        hydra_tfms.append(h_tfm)

    return (arm_tfms, hydra_tfms)


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
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]).T,I)
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

def solve_sylvester2 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
    I_0 = np.copy(I)
    I_0[3,3] = 0
        
    M_final = np.empty((0,16))

    s1_t0_inv = np.linalg.inv(tfms1[0])
    s2_t0_inv = np.linalg.inv(tfms2[0])
    
    print "\n CONSTRUCTING M: \n"
    
    for i in range(1,len(tfms1)):
        del1 = np.linalg.inv(tfms1[i]).dot(tfms1[0])
        del2 = np.linalg.inv(tfms2[i]).dot(tfms2[0])

        print "\n del1:"
        print del1
        print del1.dot(I_0).dot(del1.T)
        print "\n del2:"
        print del2, '\n'
        print del2.dot(I_0).dot(del2.T)
        
        M = np.kron(I, del1) - np.kron(del2.T,I)
        M_final = np.r_[M_final, M]
    
    L_final = -1*np.copy(M_final[:,15])
    M_final = scp.delete(M_final, (3,7,11,15), 1) 

    X = np.linalg.lstsq(M_final,L_final)[0]
    print M_final.dot(X) - L_final
    
    X2 = (np.reshape(scp.delete(np.eye(4),3,0),12,order="F"))
    print M_final.dot(X2) - L_final
    
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    print tt.T.dot(tt)
    
    return tt
    
def solve_sylvester3 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    Delta transfrom from previous iteration.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))
    
    print "\n CONSTRUCTING M: \n"
    
    for i in range(1,len(tfms1)):
        s1_inv = np.linalg.inv(tfms1[i-1])
        s2_inv = np.linalg.inv(tfms2[i-1])
        M = np.kron(I, s1_inv.dot(tfms1[i])) - np.kron(s2_inv.dot(tfms2[i]).T,I)
        M_final = np.r_[M_final, M]
    
    L_final = -1*np.copy(M_final)[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1) 

    X = np.linalg.lstsq(M_final,L_final)[0]
    print M_final.dot(X) - L_final
    
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    print tt.T.dot(tt)
    
    return tt

def solve_sylvester4 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    In order to solve the equation, it constrains rotation matrix to be the identity.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    s1_t0_inv = np.linalg.inv(tfms1[0])
    s2_t0_inv = np.linalg.inv(tfms2[0])
    
    print "\n CONSTRUCTING M: \n"
    
    for i in range(1,len(tfms1)):
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]).T,I)
        M_final = np.r_[M_final, M]
    
    # add the constraints on the last row of the transformation matrix to be == [0,0,0,1]
    L_final = -1*np.copy(M_final)[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1)
    I3 = np.eye(3)
    x_init = np.linalg.lstsq(M_final,L_final)[0]

    # Objective function:
    def f_opt (x):
        err_vec = M_final.dot(x)-L_final
        return nlg.norm(err_vec)
    
    # Rotation constraint:
    def rot_con (x):
        R = np.reshape(x,(3,4), order='F')[:,0:3]
        err_mat = R.T.dot(R) - I3
        return nlg.norm(err_mat)
    
    
    #x_init = np.linalg.lstsq(M_final,L_final)[0]
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_opt, x0=x_init, eqcons=[rot_con], iter=200, full_output=1)

    print "Function value at optimum: ", fx

    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    print tt.T.dot(tt)
    
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
    rospy.init_node('calib_hydra_pr2')
    
    parser = argparse.ArgumentParser(description="Hydra Kinect Calibration")
    parser.add_argument('--arm', help="in {'right', 'left'} : the pr2 arm to track.", required=True)  
    parser.add_argument('--hydra', help="in {'right', 'left'} : the hydra handle to track.", required=True)
    parser.add_argument('--n_tfm', help="number of transforms to use for calibration.", type=int, default=5)
    parser.add_argument('--n_avg', help="number of estimates of  transform to use for averaging.", type=int, default=5)
    parser.add_argument('--publish_tf', help="whether to publish the transform between hydra_base and camera_link", default=True)
    vals = parser.parse_args()

    arm_tfms, hydra_tfms = get_transforms(vals.arm, vals.hydra, vals.n_tfm, vals.n_avg)
    
    if vals.publish_tf:
        T_ms = solve_sylvester4(arm_tfms, hydra_tfms)
        T_chs = [arm_tfms[i].dot(T_ms).dot(np.linalg.inv(hydra_tfms[i])) for i in xrange(len(arm_tfms))]
        trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in T_chs]
        
        trans = np.asarray([trans for (trans, rot) in trans_rots])
        avg_trans = np.sum(trans,axis=0)/trans.shape[0]
        rots = [rot for (trans, rot) in trans_rots]
        avg_rot = avg_quaternions(np.array(rots))
        
        T_ch = conversions.trans_rot_to_hmat(avg_trans, avg_rot)
        
        print T_ch
        publish_tf(T_ch, 'base_footprint', 'hydra_base')                                
