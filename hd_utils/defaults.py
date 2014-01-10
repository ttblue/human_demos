import numpy as np
import os, os.path as osp

'''
Transform from camera_link (kinect frame) to camera_rgb_optical_frame
rgb optical frame (rof) 
'''
tfm_link_rof = np.array([[ 0.   ,  0.   ,  1.   ,  0.   ],
                         [-1.   , -0.   ,  0.   , -0.045],
                         [ 0.   , -1.   ,  0.   ,  0.   ],
                         [ 0.   ,  0.   ,  0.   ,  1.   ]])

'''
Camera intrinsic matrix (default value)
'''
cam_mat = np.array([[525.0, 0.0, 319.5],[0.0,525.0,239.5],[0.0,0.0,1.0]])

'''
default distortion coefficients for camera 
'''
dist_coeffs = np.array([0.,0.,0.,0.,0.])

'''
focal length of asus kinect
'''
asus_xtion_pro_f = 544.260779961


hd_path = os.getenv('HD_DIR')
data_dir = os.getenv('HD_DATA_DIR')
calib_files_dir = osp.join(data_dir, 'calib')
demo_files_dir =  osp.join(data_dir, 'demos')


'''
Default names for files inside each demo directory.
'''
master_name = 'master.yaml'
latest_demo_name = 'latest_demo.txt'
verify_name = 'verify'
class demo_names:
    base_name = 'demo%05d'
    bag_name = 'demo.bag'
    ann_name = 'ann.yaml'
    calib_name = 'calib'
    data_name = 'demo.data'
    hydra_data_name = 'hydra_only.data' #for quick visualization
    traj_name = 'demo.traj'
    video_dir = 'camera_#%i'
    camera_types_name = 'camera_types.yaml'
    camera_models_name = 'camera_models.yaml'
    stamps_name = 'stamps.txt'
    rgb_name = 'rgb%05d.jpg'
    rgb_regexp = 'rgb*.jpg'
    depth_name = 'depth%05d.png'
    depth_regexp = 'depth*.png'
    tps_model_name = 'tps_models.cp'

'''
Transform from head to kinect
'''
tfm_head_kinect = np.array(
                            [[-0.02102462, -0.03347223,  0.99921848, -0.186996  ],
                             [-0.99974787, -0.00717795, -0.02127621,  0.04361884],
                             [ 0.0078845,  -0.99941387, -0.03331288,  0.22145804],
                             [ 0.,          0.,          0.,          1.        ]]
                            )


## CAD model numbers:
#  77.83mm
#  44.16mm
