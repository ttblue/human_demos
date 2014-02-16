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

def on_mouse(event,x,y,flags,param):
    print event
    if event == 1: #left-click, overcrossing
        cv2.circle(param[0],(x,y),10,(0,0,255),-1)
        param[1].append([x,y,1])
    elif event == 2: #right-click, undercrossing
        cv2.circle(param[0],(x,y),10,(0,170,100),-1)
        param[1].append([x,y,0])


def label_crossings(demo_type, demo_name, verify=False):
    h5filename = osp.join('/Users/George/Downloads', demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    if demo_name != None:
        label_single_demo(hdf, demo_name)
    else:
        for demo in hdf.keys():
            for seg in hdf[demo].keys():
                label_single_demo(hdf, demo, seg)
    if verify:
        assert(verify_crossings(hdf))

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
    cv2.setMouseCallback(windowName, on_mouse, (image, crossings))
    while(1):
        cv2.imshow(windowName, image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            cv2.destroyAllWindows()
            image = np.asarray(seg_group['rgb'])
            x+=1
            windowName = demo+seg+" v"+str(x)
            cv2.namedWindow(windowName)
            cv2.setMouseCallback(windowName, on_mouse, (image, crossings))
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

def verify_crossings(hdf):
    for demo in hdf.keys():
        for seg in hdf[demo].keys():
            if "crossings" not in hdf[demo][seg].keys():
                return False
    return True

if __name__ == "__main__":
    #args = parser.parse_args()
    #demo_type = args.demo_type
    #demo_name = args.demo_name
    demo_type = "overhand120"
    demo_name = None
    label_crossings(demo_type, demo_name, verify=True)
