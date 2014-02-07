import os.path as osp
import numpy as np
import argparse
import h5py

import cv2, hd_rapprentice.cv_plot_utils as cpu
from hd_utils.defaults import demo_files_dir



parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--h5_name", help="Name of h5", type=str, default='')
parser.add_argument("--demo_name", help="Name of demonstration", type=str, default='')
args = parser.parse_args()



demotype_dir = osp.join(demo_files_dir, args.demo_type)
if args.h5_name == '':
    h5file = osp.join(demotype_dir, args.demo_type+".h5")
else:
    h5file = osp.join(demotype_dir, args.h5_name+".h5")

demofile = h5py.File(h5file, 'r')




if args.demo_name == '':
    demo_names = demofile.keys()
    num_demos = len(demo_names)

    names = dict(enumerate(demo_names))
    images = {i:[] for i in xrange(num_demos)}
        
    print "Getting images."
    for i,dname in names.items():
        print "Processing image %i."%(i+1)
        images[i] = [np.asarray(demofile[dname][sname]["rgb"]) for sname in demofile[dname]]
    print "Done images."

    i = 0
    inc = True
    print "Press q to exit, left/right arrow keys to navigate"
    while True:
        print
        print names[i]
        row = cpu.tile_images(images[i], 1, len(images[i]))
        cv2.imshow("segments", row)
        kb = cv2.waitKey()
        if kb == 1113939 or kb == 65363:
            i = min(i+1,num_demos-1)
            inc = True
        elif kb == 1113937 or kb == 65361:
            i = max(i-1,0)
            inc = False
        elif kb == 1048689 or kb == 113:
            break

else:
    images = [np.asarray(demofile[args.demo_name][sname]["rgb"]) for sname in demofile[args.demo_name]]

    row = cpu.tile_images(images, 1, len(images))
    cv2.imshow("segments", images)
    kb = cv2.waitKey()