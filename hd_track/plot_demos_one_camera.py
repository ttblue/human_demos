import argparse
from tracking_utility import *

if __name__=="__main__":
    rospy.init_node('viz_demos')    
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', help="Plot the data", action="store_true", default=False)
    parser.add_argument('--rviz', help="Publish the data on topics for visualization on RVIZ", action="store_true", default=True)
    parser.add_argument('-freq', help="frequency in filter", action='store', dest='freq', default=30., type=float)
    parser.add_argument('-dname', help="name of demonstration file", action='store', dest='demo_fname', default='demo100', type=str)
    parser.add_argument('-clib', help="name of calibration file", action='store', dest='calib_fname', default = 'cc_two_camera_calib', type=str)
    vals = parser.parse_args()
    
    freq = vals.freq
    demo_fname = vals.demo_fname
    calib_fname = vals.calib_fname

    if vals.plot:
        demo_dir = hd_path + '/hd_data/demos/obs_data'
        data_file = osp.join(demo_dir, demo_fname+'.data')
        plot_kalman(data_file, calib_fname, freq, use_spline=True, single_camera=True, customized_shift=None, plot_commands='s1fh')
    else:
        data_file = osp.join(hd_path + '/hd_data/demos/obs_data', demo_fname+'.data')
        bag_file = osp.join(hd_path + '/hd_data/demos/recorded', demo_fname+'.bag')
        rviz_kalman('', bag_file, data_file, calib_fname, freq, use_rgbd=False, use_smoother=False, use_spline=False, customized_shift=None, single_camera=True)

