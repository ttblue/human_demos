import numpy as np
import conversions

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
    Average out quaternions as above.
    """
    trans_rots = [conversions.hmat_to_trans_rot(tfm) for tfm in tfms]
    trans = np.asarray([trans for (trans, rot) in trans_rots])
    avg_trans = np.sum(trans,axis=0)/trans.shape[0]
    
    rots = [rot for (trans, rot) in trans_rots]
    avg_rot = avg_quaternions(np.array(rots))

    return conversions.trans_rot_to_hmat(avg_trans, avg_rot)