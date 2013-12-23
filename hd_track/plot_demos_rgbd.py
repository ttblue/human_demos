import argparse
from tracking_utility_ankush import *

        
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--plot', help="Plot the data", action="store_true", default=False)
    parser.add_argument('--rviz', help="Publish the data on topics for visualization on RVIZ", action="store_true", default=True)
    parser.add_argument('-freq', help="frequency in filter", action='store', dest='freq', default=30., type=float)
    parser.add_argument('-dname', help="name of demonstration file", action='store', dest='demo_fname', default='demo100', type=str)
    parser.add_argument('-clib', help="name of calibration file", action='store', dest='calib_fname', default = 'cc_two_camera_calib', type=str)
    vals = parser.parse_args()

    
    rospy.init_node('viz_demos',anonymous=True)    
    

    freq = vals.freq
    demo_fname = vals.demo_fname
    calib_fname = vals.calib_fname

    if vals.plot:
        demo_dir = hd_path + '/hd_data/demos/' + demo_fname;
        data_file = osp.join(demo_dir, 'demo.data')
        plot_kalman(data_file, calib_fname, freq, use_spline=False, customized_shift=None, plot_commands='s12fh')
    else:
        demo_dir = hd_path + '/hd_data/demos/' + demo_fname;
        rviz_kalman(demo_dir, '', '', calib_fname, freq, use_rgbd=True, use_smoother=True, use_spline=True, customized_shift=None, single_camera=False)

