'''
Extract trajectory from rgbd
'''
import argparse
from tracking_utility import *
import cPickle

        
if __name__=="__main__":
    rospy.init_node('viz_demos',anonymous=True)    
    parser = argparse.ArgumentParser()
    parser.add_argument('-freq', help="frequency in filter", action='store', dest='freq', default=30., type=float)
    parser.add_argument('-dname', help="name of demonstration file", action='store', dest='demo_fname', default='demo100', type=str)
    parser.add_argument('-clib', help="name of calibration file", action='store', dest='calib_fname', default = 'cc_two_camera_calib', type=str)
    vals = parser.parse_args()
    
    freq = vals.freq
    demo_fname = vals.demo_fname
    calib_fname = vals.calib_fname

    demo_dir = hd_path + '/hd_data/demos/' + demo_fname;
    data_file = osp.join(demo_dir, 'demo.data')
    traj_file = osp.join(demo_dir, 'demo.traj')
    traj_data = traj_kalman(data_file, calib_fname, freq)
    
    with open(traj_file, 'w') as fh:
        cPickle.dump(traj_data, fh)
