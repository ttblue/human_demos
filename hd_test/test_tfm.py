import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco

np.set_printoptions(precision=5, suppress=True)

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    

def make_obs(trans_SCALE=0.1, theta_SCALE=1.5):
    axis = np.random.randn(3)
    theta = np.random.rand()*np.pi*theta_SCALE
    
    t = np.random.randn(3)*trans_SCALE
    
    tfm = np.eye(4)
    tfm[0:3,0:3] = rotation_matrix(axis, theta)
    tfm[0:3,3] = t
    
    return tfm


#Observations
def test_tfm (n):
    
    Tms = make_obs()
    Tsm = np.linalg.inv(Tms)
    I = np.eye(4)
    
    M_final = np.empty((0,16))
    
    for _ in range(n):
        del_Ts = make_obs()
        del_Tm = Tms.dot(del_Ts.dot(Tsm))
        
        M = np.kron(I,del_Tm)-np.kron(del_Ts.T,I)
        
        M_final = np.r_[M_final, M]
    
    for i in [3,7,11,15]:
        t = np.zeros((1,16))
        t[0,i] = 1
        M_final = np.r_[M_final,t]
        
    L_final = np.zeros(M_final.shape[0])
    L_final[-1] = 1
        
    X = np.linalg.lstsq(M_final,L_final)[0]
    Tfm = np.reshape(X,(4,4),order='F')

    np.set_printoptions(precision=5)    
    
    print Tfm
    print Tms
    
    R = Tfm[0:3,0:3]
    print R.T.dot(R)
    
    X2 = np.reshape(Tms,16,order="F")
    print X2
    
    assert(np.allclose(M_final.dot(X2),L_final, atol=0.001))
    assert (np.allclose(Tfm,Tms, atol=0.001))
    
# Change M after assuming last row of transform is [0,0,0,1]
def test_tfm2 (n, addnoise=False):
    
    Tms = make_obs()
    Tsm = np.linalg.inv(Tms)
    I = np.eye(4)
    
    M_final = np.empty((0,16))
    
    for i in range(n):
        del_Ts = make_obs(0.05, 0.05)
        if addnoise:
            noise = make_obs(0.01,0.01)
            del_Tm = Tms.dot(del_Ts.dot(Tsm)).dot(noise)
        else:
            del_Tm = Tms.dot(del_Ts.dot(Tsm))
        print "Observation %i"%(i+1)
        print "Delta Ts:"
        print del_Ts
        print "Delta Tm:"
        print del_Tm, '\n'
        
        M = np.kron(I,del_Tm)-np.kron(del_Ts.T,I)
        
        M_final = np.r_[M_final, M]
    
    
    L_final = -1*M_final[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1) 

            
    X = np.linalg.lstsq(M_final,L_final)[0]
    Tfm = np.reshape(X,(3,4),order='F')
    print Tfm.shape
    Tfm = np.r_[Tfm,np.array([[0,0,0,1]])]

    np.set_printoptions(precision=5)    
    
    print Tfm
    print Tms
    
    R = Tfm[0:3,0:3]
    print R.T.dot(R)
    
    X2 = scp.delete(np.reshape(Tms,16,order="F"),(3,7,11,15),0)
    print X2
    
    if not addnoise:
        assert(np.allclose(M_final.dot(X2),L_final, atol=0.001))
        assert (np.allclose(Tfm,Tms, atol=0.001))
    
def solve_sylvester2 (tfms1, tfms2, addnoise=True):
    """
    Solves the system of Sylvester's equations to find the calibration transform.
    Returns the calibration transform from sensor 1 (corresponding to tfms1) to sensor 2.
    This functions forces the bottom row to be 0,0,0,1 by neglecting columns of M and changing L.
    """

    assert len(tfms1) == len(tfms2) and len(tfms1) >= 2
    I = np.eye(4)
        
    M_final = np.empty((0,16))

    s1_t0_inv = np.linalg.inv(tfms1[0])
    s2_t0_inv = np.linalg.inv(tfms2[0])
    
    print "\n CONSTRUCTING M: \n"
    
    for i in range(1,len(tfms1)):
        if addnoise:
            noise = make_obs(0.01,0.01)
        else:
            noise = np.eye(4)
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]).dot(noise).T,I)
        M_final = np.r_[M_final, M]
    
    # add the constraints on the last row of the transformation matrix to be == [0,0,0,1]
    L_final = -1*np.copy(M_final)[:,15]
    M_final = scp.delete(M_final, (3,7,11,15), 1) 

    X = np.linalg.lstsq(M_final,L_final)[0]
    print M_final.dot(X) - L_final
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    print tt.T.dot(tt)
    
    return tt

def solve_sylvester4 (tfms1, tfms2, addnoise=True):
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
        if addnoise:
            noise = make_obs(0.01,0.01)
        else:
            noise = np.eye(4)
        M = np.kron(I, s1_t0_inv.dot(tfms1[i])) - np.kron(s2_t0_inv.dot(tfms2[i]).dot(noise).T,I)
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
    
    (X, fx, _, _, _) = sco.fmin_slsqp(func=f_opt, x0=x_init, eqcons=[rot_con], iter=200, acc=1e-3, full_output=1)

    print "Function value at optimum: ", fx

    print nlg.norm(M_final.dot(X) - L_final)
    print nlg.norm(M_final.dot(x_init) - L_final)
    tt = np.reshape(X,(3,4),order='F')
    tt = np.r_[tt,np.array([[0,0,0,1]])]
    
    print tt.T.dot(tt)
    
    return tt
    
def test_tfm3 (n):
    """
    Test of the full pipeline.
    """
    Tch = make_obs(0.5)
    Thc = nlg.inv(Tch)
    Tms = make_obs(0.1)
    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    print "Tch:"
    print Tch
    print Tch.dot(I_0).dot(Tch.T)
    print "Tms:"
    print Tms
    print Tms.dot(I_0).dot(Tms.T), '\n'
    
    
    tfms1 = []
    tfms2 = []
    for i in range(n):
        Tcm = make_obs(0.3)
        Ths = Thc.dot(Tcm).dot(Tms)
        
        tfms1.append(Tcm)
        tfms2.append(Ths)
        print "Observation %i"%(i+1)
        print "Tcm:"
        print Tcm
        print Tcm.dot(I_0).dot(Tcm.T)
        print "Ths:"
        print Ths
        print Ths.dot(I_0).dot(Ths.T)
        print Tcm.dot(Tms).dot(nlg.inv(Ths)).dot(Thc), '\n'
        
    Tms_calib = solve_sylvester4(tfms1, tfms2, False)
    
    print "Tms: \n", Tms
    print "Tms_calib: \n", Tms_calib
    
    R = Tms_calib[0:3,0:3]
    print R.T.dot(R)
    
    assert (np.allclose(Tms_calib,Tms, atol=0.001))
    
def test_tfm4 (n):
    """
    Test of the full pipeline.
    """
    Tch = np.eye(4)#make_obs(0.5)
    Thc = nlg.inv(Tch)
    Tms = make_obs(0.1)
    I_0 = np.eye(4)
    I_0[3,3] = 0
    
    print "Tch:"
    print Tch
    print Tch.dot(I_0).dot(Tch.T)
    print "Tms:"
    print Tms
    print Tms.dot(I_0).dot(Tms.T), '\n'
    
    
    tfms1 = []
    tfms2 = []
    for i in range(n):
        Tcm = make_obs(0.3)
        noise = np.eye(4)#make_obs(0.01,0.02)
        Ths = Thc.dot(Tcm).dot(Tms).dot(noise)
        
        tfms1.append(Tcm)
        tfms2.append(Ths)
        print "Observation %i"%(i+1)
        print "Tcm:"
        print Tcm
        print Tcm.dot(I_0).dot(Tcm.T)
        print "Ths:"
        print Ths
        print Ths.dot(I_0).dot(Ths.T)
        print Tcm.dot(Tms).dot(nlg.inv(Ths)).dot(Thc), '\n'
        
    Tms_calib = solve_sylvester4(tfms1, tfms2, False)
    
    print "Tms: \n", Tms
    print "Tms_calib: \n", Tms_calib
    
    print "Tch: \n", Tch
    print "Tch calib: \n", tfms1[0].dot(Tms).dot(np.linalg.inv(tfms2[0]))
    
    R = Tms_calib[0:3,0:3]
    print R.T.dot(R)
    
    assert (np.allclose(Tms_calib,Tms, atol=0.001))

def test_tfm_val1():

    Tch = make_obs(0.5)
    Thc = nlg.inv(Tch)
    Tms = make_obs(0.1)
    I_0 = np.eye(4)
    I_0[3,3] = 0

    tfms1 = []
    tfms2 = []
    
    tfms1.append([[-0.0096,  -0.03728, -0.99926,  0.75043],
                  [ 0.99995,  0.,      -0.0096,  -0.07002],
                  [ 0.00036, -0.9993,   0.03728, -0.02286],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append(Thc.dot(tfms1[0]).dot(Tms))
    
    tfms1.append([[-0.0273,  -0.70346, -0.71021,  0.75183],
                  [ 0.99952, -0.00897, -0.02954, -0.09279],
                  [ 0.0144,  -0.71068,  0.70337, -0.088  ],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append(Thc.dot(tfms1[1]).dot(Tms))

    tfms1.append([[-0.48571, -0.1099,  -0.86718,  0.65788],
                  [ 0.82301,  0.27675, -0.49605,  0.14502],
                  [ 0.29451, -0.95464, -0.04397, -0.02296],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append(Thc.dot(tfms1[2]).dot(Tms))
    
    Tms_calib = solve_sylvester2(tfms1, tfms2)
    
    print "Tms: \n", Tms
    print "Tms_calib: \n", Tms_calib
    
    R = Tms_calib[0:3,0:3]
    print R.T.dot(R)
    
    assert (np.allclose(Tms_calib,Tms, atol=0.001))


def test_tfm_val2():

#     Tch = make_obs(0.5)
#     Thc = nlg.inv(Tch)
#     Tms = make_obs(0.1)
#     I_0 = np.eye(4)
#     I_0[3,3] = 0

    tfms1 = []
    tfms2 = []
    
    tfms1.append([[-0.0096,  -0.03728, -0.99926,  0.75043],
                  [ 0.99995,  0.,      -0.0096,  -0.07002],
                  [ 0.00036, -0.9993,   0.03728, -0.02286],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append([[ 0.2821,  -0.95904, -0.02587, -0.11373],
                  [-0.70497, -0.18893, -0.68361,  0.03023],
                  [ 0.65072,  0.21108, -0.72939,  0.00709],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms1.append([[-0.0273,  -0.70346, -0.71021,  0.75183],
                  [ 0.99952, -0.00897, -0.02954, -0.09279],
                  [ 0.0144,  -0.71068,  0.70337, -0.088  ],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append([[ 0.09672, -0.98422,  0.14819, -0.14102],
                  [-0.96374, -0.12981, -0.23314,  0.07651],
                  [ 0.24869, -0.12027, -0.96109, -0.06604],
                  [ 0.,       0.,       0.,       1.     ]])

    tfms1.append([[-0.48571, -0.1099,  -0.86718,  0.65788],
                  [ 0.82301,  0.27675, -0.49605,  0.14502],
                  [ 0.29451, -0.95464, -0.04397, -0.02296],
                  [ 0.,       0.,       0.,       1.     ]])
    
    tfms2.append([[ 0.03962, -0.72105,  0.69175, -0.08271],
                  [-0.62218, -0.55951, -0.54758, -0.18059],
                  [ 0.78187, -0.4087,  -0.47079,  0.01134],
                  [ 0.,       0.,       0.,       1.     ]])
    
    Tms_calib = solve_sylvester2(tfms1, tfms2)
    
    #print "Tms: \n", Tms
    print "Tms_calib: \n", Tms_calib
    
    R = Tms_calib[0:3,0:3]
    print R.T.dot(R)
    
    #assert (np.allclose(Tms_calib,Tms, atol=0.001))
