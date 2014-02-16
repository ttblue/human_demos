import numpy as np
import cv2, argparse, h5py
import os.path as osp

#from hd_utils.defaults import demo_files_dir

usage = """
To view and label all demos of a certain task type:
python label_crossings.py --demo_type=DEMO_TYPE
"""
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--single_demo", help="View and label a single demo", default=False, type=str)
parser.add_argument("--demo_name", help="Name of demo if single demo.", default="", type=str)

"""
Mark the displayed image with a colored circle, and save the x,y coords
and the type (right or left) of the click to the crossings array.
A left-click signifies an over-crossing, while a right-click signifies
an under-crossing.
"""
def mark_crossing(event,x,y,flags,param):
    if event == 1: #left-click, overcrossing
        cv2.circle(param[0],(x,y),5,(0,200,200),-1)
        param[1].append([x,y,1])
    elif event == 2: #right-click, undercrossing
        cv2.circle(param[0],(x,y),5,(200,170,50),-1)
        param[1].append([x,y,0])

"""
Mark the displayed image with a colored circle, and save the x,y coords
and the type (right or left) of the click to the rope_ends array.
A left-click signifies the "starting" end of the rope, while a right-click
signifies the "finishing" end of the rope.
Over- and under-crossings are defined in relation to the starting and
finishing ends of the rope. An under-crossing is a crossing where a
traversal of the rope from the starting end to the finishing end would
first encounter the crossing on the lower segment, followed by the upper
segment. An over-crossing is a crossing where a traversal of the rope 
from the starting end to the finishing end would first encounter the 
crossing on the lower segment, followed by the upper segment.
"""
def mark_ends(event,x,y,flags,param):
    if event == 1: #left-click, beginning end
        cv2.circle(param[0],(x,y),5,(0,200,200),-1)
        param[1].append([x,y,1])
    elif event == 2: #right-click, finshing end
        cv2.circle(param[0],(x,y),5,(200,170,50),-1)
        param[1].append([x,y,0])

"""
Iterate through the demos in the h5 file denoted by demo_type, and call
label_single_demo on each.
If demo_name is not null, iterate only over the segments of demo_name.
The parameter verify indicates whether to check that every segment in
the h5 file was labeled.
"""
def label_crossings(demo_type, demo_name, verify=False):
    h5filename = osp.join('/Users/George/Downloads', demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    if demo_name != None:
        for seg in hdf[demo].keys():
            label_single_demo(hdf, demo_name)
    else:
        for demo in hdf.keys():
            for seg in hdf[demo].keys():
                label_single_demo(hdf, demo, seg)
    if verify:
        assert(verify_crossings(hdf))

"""
Set up a cv2 window displaying the image corresponding to demo and seg
in the hdf file, and label the x,y coordinates and click type on click
by calling mark_crossing.
"""
def label_single_demo(hdf, demo, seg):
    print "seg", seg
    seg_group = hdf[demo][seg]
    if "crossings" in seg_group.keys():
        print "crossings already labeled for", demo, seg
        print seg_group.keys()
        return
    crossings = []
    image = np.asarray(seg_group['rgb'])
    x = 0
    windowName = demo+seg
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mark_crossing, (image, crossings))
    while(1):
        cv2.imshow(windowName, image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            cv2.destroyAllWindows()
            image = np.asarray(seg_group['rgb'])
            x+=1
            windowName = demo+seg+" v"+str(x)
            cv2.namedWindow(windowName)
            cv2.setMouseCallback(windowName, mark_crossing, (image, crossings))
        elif k == ord('m'):
            break
        elif k == 27:
            return 0
    print crossings
    crossings = np.asarray(crossings)
    print crossings
    seg_group.create_dataset("crossings", data=crossings)
    print seg_group["crossings"]
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #args = parser.parse_args()
    #demo_type = args.demo_type
    #demo_name = args.demo_name
    demo_type = "overhand120"
    demo_name = None
    label_crossings(demo_type, demo_name, verify=True)
