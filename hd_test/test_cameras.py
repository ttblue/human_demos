import rospy

from hd_calib.camera_calibration import camera_calibrator
from hd_calib.cyni_cameras import cyni_cameras
from hd_calib.calibration_pipeline import transform_publisher
from hd_utils.colorize import *
from hd_utils.yes_or_no import yes_or_no

rospy.init_node('calibration')
def test_cam_calib ():

    yellowprint("Beginning calibration sequence.")

    NUM_CAMERAS = 2
    cameras = cyni_cameras(NUM_CAMERAS)
    cameras.initialize_cameras()
    
    tfm_pub = transform_publisher(cameras)
    tfm_pub.start()
        
    greenprint("Step 1. Calibrating mutliple cameras.")
    CAM_N_OBS = 10
    CAM_N_AVG = 5
    cam_calib = camera_calibrator(cameras)

    done = False
    while not done:
        cam_calib.calibrate(CAM_N_OBS, CAM_N_AVG, tfm_pub)
        if not cam_calib.calibrated:
            redprint("Camera calibration failed.")
            cam_calib.reset_calibration()
        else:
            tfm_pub.add_transforms(cam_calib.get_transforms())
            if yes_or_no("Are you happy with the calibration? Check RVIZ."):
                done = True
            else:
                yellowprint("Calibrating cameras again.")
                cam_calib.reset_calibration()
    
    tfm_pub.set_publish_pc(True)
    greenprint("Mutliple cameras calibrated.")
    print tfm_pub.publish_pc
    
    import time
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
 
