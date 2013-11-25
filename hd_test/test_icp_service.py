import numpy as np, numpy.random as npr
import roslib; roslib.load_manifest('icp_service')
import rospy

from icp_service.srv import ICPTransform, ICPTransformRequest, ICPTransformResponse 

from hd_utils import ros_utils as ru, conversions

np.set_printoptions(precision=5, suppress=True)

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


def do_icp(surface_points):
    #Random point set, scaled up
    p1 = surface_points
    n = p1.shape[0]
    
    # Transform the points by some random transform
    tfm = make_transform(trans_SCALE=1)
    p2 = np.c_[p1,np.ones((n,1))].dot(tfm.T)[:,0:3]
    
    #print p1
    #print p2
    print tfm
    
    noise = make_transform(0.01,0.1)
    
    req = ICPTransformRequest()
    req.pc1 = ru.xyz2pc(p1, 'a')
    req.pc2 = ru.xyz2pc(p2, 'b')
    req.guess = conversions.hmat_to_pose(tfm.dot(noise))
    
    res = findTransform(req)
    ntfm = conversions.pose_to_hmat(res.pose)
    print ntfm
    
    print ntfm.dot(np.linalg.inv(tfm))
    print np.linalg.norm(ntfm.dot(np.linalg.inv(tfm)) - np.eye(4))
    return np.linalg.norm(np.abs(ntfm[0:3,3]-tfm[0:3,3]))

if __name__=="__main__":
    import openravepy as opr
    import hd_utils.stl_utils as stlu
    
    SCALE = 10;
    
    rospy.init_node('test_icp')
    findTransform = rospy.ServiceProxy("icpTransform", ICPTransform)
    
    env = opr.Environment()
    env.Load('/home/sibi/sandbox/human_demos/hd_data/sensors/sensor_robot_2.00.xml')
    sr = env.GetRobots()[0]
    ss = env.GetSensors()[0]
    ss.Configure(ss.ConfigureCommand.PowerOn)

#    env.SetViewer('qtcoin')
    check_points = stlu.generate_sphere_points(1)

    surface_points = np.empty((0,3))
    for point in check_points:
        pos_points = stlu.get_points_from_position(sr, point, 2.00, check_valid=False)
        surface_points = np.vstack((surface_points, pos_points))
    