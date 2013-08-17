import numpy as np

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    

def make_obs():
    axis = np.random.randn(3)
    theta = np.random.rand()*np.pi*2
    
    SCALE = 2
    t = np.random.randn(3)*SCALE
    
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