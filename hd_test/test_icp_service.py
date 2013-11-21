import numpy as np, numpy.random as npr
import roslib; roslib.load_manifest('icp_service')
import rospy

from icp_service.srv import ICPTransform, ICPTransformRequest, ICPTransformResponse 

from hd_utils import ros_utils as ru, conversions

def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
    

def make_transform(trans_SCALE=0.1, theta_SCALE=1.5):
    axis = np.random.randn(3)
    theta = np.random.rand()*np.pi*theta_SCALE
    
    t = np.random.randn(3)*trans_SCALE
    
    tfm = np.eye(4)
    tfm[0:3,0:3] = rotation_matrix(axis, theta)
    tfm[0:3,3] = t
    
    return tfm


if __name__=="__main__":

    n = 10;
    SCALE = 10;
    
    rospy.init_node('test_icp')
    findTransform = rospy.ServiceProxy("icpTransform", ICPTransform)    
    
    #Random point set, scaled up
    p1 = npr.rand(n,3)*SCALE;
    
    # Transform the points by some random transform
    tfm = make_transform(trans_SCALE=0.5)
    p2 = np.c_[p1,np.ones((n,1))].dot(tfm.T)[:,0:3]
    
    print p1
    print p2
    print tfm
    print tfm[0:3,0:3].dot(tfm[0:3,0:3].T)
    
    noise = make_transform(0.05,0.1)
    
    req = ICPTransformRequest()
    req.pc1 = ru.xyz2pc(p1, '')
    req.pc2 = ru.xyz2pc(p2, '')
    req.guess = conversions.hmat_to_pose(np.eye(4).dot(noise))
    
    res = findTransform(req)
    ntfm = conversions.pose_to_hmat(res.pose)