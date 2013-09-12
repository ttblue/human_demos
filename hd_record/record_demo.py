"""
Load or recalibrate first.
When ready, run loop at some frequency to save stuff.
"""
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("demo_prefix")
parser.add_argument("master_file")
parser.add_argument("num_cameras", default=2)
parser.add_argument("calibration_file", default='')
args = parser.parse_args()

import os, os.path as osp
import itertools
import rospy

from  hd_calib import calibration_pipeline as cpipe, cameras 
from hd_utils.colorize import *


data_dir = os.getenv('HD_DATA_DIR')
master_file = osp.join(data_dir,'demos', args.master_file)
if not osp.isfile(master_file):
    with open(master_file, "w") as f:
        f.write("name: %s\n"%args.master_file)
        f.write("h5path: %s\n"%(args.master_file+".h5"))
        f.write("demos:\n")

with open(master_file, "r") as fh: master_info = yaml.load(fh)   
if master_info["demos"] is None: master_info["demos"] = []
for suffix in itertools.chain("", (str(i) for i in itertools.count())):
    demo_name = args.demo_prefix + suffix
    if not any(demo["demo_name"] == demo_name for demo in master_info["demos"]):
        break
    print 'Demo name: ',demo_name


rospy.init_node('record_demo')
cameras = None
tfm_pub = None

if args.calibration_file == '':
    cpipe.NUM_CAMERAS = args.num_cameras
    cpipe.run_calibration_sequence()
    tfm_pub = cpipe.tfm_pub
    cameras = cpipe.cameras
else:
    file = osp.join(data_dir,'calib',args.file)
    tfm_pub = cpipe.CalibratedTransformPublisher()
    cameras = cameras.RosCameras(num_cameras=args.num_cameras)
    tfm_pub.load_calibration(file)

tfm_pub.start()

raw_input("Hit enter when ready to record demo.")
yellowprint("Recording demonstration now...")

freq = 10
demo_dir = osp.join('data_dir', 'demos', demo_name)

video_dir = osp.join(demo_dir, video_dir)

try:
    
    while True:
        """
        Store stuff.
        """
        1
    
except KeyboardInterrupt:
    greenprint("got control-c")
except Exception:
    import traceback
    traceback.print_exc()
    
"""
Finish up.
"""