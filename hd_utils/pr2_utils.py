"""
put Berkeley-specific parameters here
"""
from hd_utils.defaults import tfm_head_dof, tfm_bf_head


def get_kinect_transform(robot, sim=False):    
    """
    Returns transform from base_footprint to camera_depth_optical_frame.
    Call update_rave() before this! 
    """
    if sim:
        return tfm_bf_head.dot(tfm_head_dof)

    T_w_h = robot.GetLink("head_plate_frame").GetTransform()    
    T_w_k = T_w_h.dot(tfm_head_dof)
    return T_w_k

       