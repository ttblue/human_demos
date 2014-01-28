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

"""
Transform from PR2's head_plate_frame to camera's camera_depth_optical_frame.
"""
tfm_head_dof = np.array([[ 0.019823, -0.035085,  0.999188, -0.189604],
                         [-0.999791,  0.004299,  0.019986,  0.025004],
                         [-0.004997, -0.999375, -0.034993,  0.2199  ],
                         [ 0.      ,  0.      ,  0.      ,  1.      ]])

"""
ONLY FOR SIMULATION:
Transform from PR2's base_footprint to PR2's head_plate_frame.
DO NOT USE THIS WHILE RUNNING REAL EXECUTION. THIS DEPENDS ON ANGLE OF HEAD.
"""
tfm_bf_head = np.array([[ 0.241649, -0.017298,  0.97021 ,  0.069105],
                        [ 0.004181,  0.99985 ,  0.016785,  0.002356],
                        [-0.970355, -0.      ,  0.241685,  1.462202],
                        [ 0.      ,  0.      ,  0.      ,  1.      ]])

"""
Transform from lr_gripper_tool_frame to end_effector_transform.
This is so that you can give openrave the data in the frame it is expecting.
Openrave does IK in end_effector_frame which is different from gripper_tool_frame.
"""
tfm_gtf_ee = np.array([[ 0.,  0.,  1.,  0.],
                       [ 0.,  1.,  0.,  0.],
                       [-1.,  0.,  0.,  0.],
                       [ 0.,  0.,  0.,  1.]])



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
data_dir = os.getenv('HD_DEMO_DATA_DIR')
hd_data_dir = os.getenv('HD_DATA_DIR')
calib_files_dir = osp.join(hd_data_dir, 'calib')
demo_files_dir  =  osp.join(data_dir, 'demos')

'''
Default names for files inside each demo directory and others.
'''
ar_init_dir = osp.join(hd_data_dir,'ar_init')
ar_init_demo_name = 'demo.cp'
ar_init_playback_name = 'playback.cp'

feedback_dir = osp.join(hd_data_dir,'demo_feedback')
simple_feedback_name = 'simple_fb.cp'
ik_feedback_name = 'ik_fb.cp'

perturbation_file = 'old_perturb.cp'
new_pert_file = 'new_perturb.cp'

cad_files_dir = osp.join(hd_data_dir, 'cad_models')



similarity_costs_dir = osp.join(hd_data_dir,'sim_costs')
matrix_file = '%s_matrix.cp'

master_name = 'master.yaml'
latest_demo_name = 'latest_demo.txt'
verify_name = 'verify'

cam_pr2_calib_name = 'cam_pr2'

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
    tps_model_name = 'tps_models.cp' #Change this to l/r
    
    record_demo_temp = 'recording_demo.tmp'
    extract_data_temp = 'extracting_data.tmp'
    extract_hydra_data_temp = 'extracting_hydra_data.tmp'
    run_kalman_temp = 'running_kalman.tmp'

    init_ar_marker_name = 'ar_marker_init.cp'
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
