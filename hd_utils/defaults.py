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
calib_files_dir = osp.join(data_dir, 'calib')
demo_files_dir =  osp.join(data_dir, 'demos')


'''
Default names for files inside each demo directory.
'''
master_name = 'master.yaml'
class demo_names:
    demo_base_name = 'demo%05d'
    demo_bag_name = 'demo.bag'
    demo_ann_name = 'ann.yaml'
    demo_calib_name = 'calib'
    demo_data_name = 'demo.data'
    demo_traj_name = 'demo.traj'
    demo_camera_types_name = 'camera_types.yaml'
    demo_camera_models_name = 'camera_models.yaml'

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
