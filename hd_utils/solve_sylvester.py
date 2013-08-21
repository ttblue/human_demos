import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

VERBOSE = 1

def solve1 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    s1_t0_inv = nlg.inv(tfms1[0])
    s2_t0_inv = nlg.inv(tfms2[0])
    
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

    X = nlg.lstsq(M_final,L_final)[0]
    
    if VERBOSE:
        print M_final.dot(X)

    tt = np.reshape(X,(4,4),order='F')
    return tt

def solve2 (tfms1, tfms2):
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

    s1_t0_inv = nlg.inv(tfms1[0])
    s2_t0_inv = nlg.inv(tfms2[0])
    
    for i in range(1,len(tfms1)):
        del1 = nlg.inv(tfms1[i]).dot(tfms1[0])
        del2 = nlg.inv(tfms2[i]).dot(tfms2[0])

        if VERBOSE:
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

    X = nlg.lstsq(M_final,L_final)[0]
    if VERBOSE:
        print M_final.dot(X) - L_final
        
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    if VERBOSE:
        print tt.T.dot(tt)
    
    return tt
    
def solve3 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    Delta transfrom from previous iteration.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    for i in range(1,len(tfms1)):
        s1_inv = nlg.inv(tfms1[i-1])
        s2_inv = nlg.inv(tfms2[i-1])
        M = np.kron(I, s1_inv.dot(tfms1[i])) - np.kron(s2_inv.dot(tfms2[i]).T,I)
        M_final = np.r_[M_final, M]
    
    L_final = -1*np.copy(M_final)[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1) 

    X = nlg.lstsq(M_final,L_final)[0]
    if VERBOSE:
        print M_final.dot(X) - L_final
    
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    if VERBOSE:
        print tt.T.dot(tt)
    
    return tt

def solve4 (tfms1, tfms2):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    In order to solve the equation, it constrains rotation matrix to be the identity.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    # Inverses of transforms of two sensors at time t = 0
    s1_t0_inv = nlg.inv(tfms1[0])
    s2_t0_inv = nlg.inv(tfms2[0])

    for i in range(1,len(tfms1)):
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]).T,I)
        M_final = np.r_[M_final, M]
    
    # add the constraints on the last row of the transformation matrix to be == [0,0,0,1]
    L_final = -1*np.copy(M_final)[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1)
    I3 = np.eye(3)
    x_init = nlg.lstsq(M_final,L_final)[0]

    # Objective function:
    def f_opt (x):
        err_vec = M_final.dot(x)-L_final
        return nlg.norm(err_vec)
    
    # Rotation constraint:
    def rot_con (x):
        R = np.reshape(x,(3,4), order='F')[:,0:3]
        err_mat = R.T.dot(R) - I3
        return nlg.norm(err_mat)
    
    
    #x_init = nlg.lstsq(M_final,L_final)[0]
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_opt, x0=x_init, eqcons=[rot_con], iter=200, full_output=1)

    if VERBOSE:
        print "Function value at optimum: ", fx

    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    if VERBOSE:
        print tt.T.dot(tt)
    
    return tt
