import subprocess, os, signal

import numpy as np, numpy.linalg as nla
import conversions
from colorize import redprint
import transformations as tfms

def display_all_keys(d, depth=0, limit=-1):
    """
    Displays all the keys hierarchically in a dictionary. 
    """
    if limit >= 0 and depth > limit:
        return
    if not isinstance(d,dict):
        if depth==0:
            print "Not a dictionary."
        return
    
    keys = sorted(d.keys())
    for key in keys:
        print '  '*depth + '- ' + str(key)
        display_all_keys(d[key],depth+1,limit)
        

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

def avg_transform (tfms):
    """
    Averages transforms by converting to translations + quaternions.
    
    Average out translations as normal (sum of vectors / # of vectors).
    Average out quaternions as in avg_quaternions.
    """

    if len(tfms) == 0:
        #redprint("List empty for averaging transforms!")
        return None
    
    trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in tfms]
    trans = np.asarray([trans for (trans, rot) in trans_rots])
    avg_trans = np.sum(trans,axis=0)/trans.shape[0]
    
    rots = [rot for (trans, rot) in trans_rots]
    avg_rot = avg_quaternions(np.array(rots))

    return conversions.trans_rot_to_hmat(avg_trans, avg_rot)

def rad_angle(angle):
    return angle/180.0*np.pi

def deg_angle(angle):
    return angle*180.0/np.pi

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def make_perp_basis(a0):
  a = a0/nla.norm(a0)
  if a[0] == 0 and a[1] == 0:
      return np.eye(3)
  else :
    b = np.array([a0[1], -a0[0], 0])
    b/= nla.norm(b)
    c = np.cross(a,b)
    return np.c_[a,b,c]


def state_to_hmat(Xs):
    """
    Converts a list of 12 dof state vector (used in the kalman filteR) to a list of transforms.
    """
    Ts = []
    for x in Xs:
        trans = x[0:3]
        rot   = x[6:9]
        T = tfms.euler_matrix(rot[0], rot[1], rot[2])
        T[0:3,3] = np.reshape(trans, 3)
        Ts.append(T)
    return Ts


def terminate_process_and_children(p):
    ps_command = subprocess.Popen("ps -o pid --ppid %d --noheaders" % p.pid, shell=True, stdout=subprocess.PIPE)
    ps_output = ps_command.stdout.read()
    retcode = ps_command.wait()
    assert retcode == 0, "ps command returned %d" % retcode
    for pid_str in ps_output.split("\n")[:-1]:
            os.kill(int(pid_str), signal.SIGINT)
    p.terminate()