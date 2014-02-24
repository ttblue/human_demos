import numpy as np
import cv2, argparse, h5py
import os.path as osp

#from hd_utils.defaults import demo_files_dir
demo_files_dir = '/Users/George/Downloads'

usage = """
To view and label all demos of a certain task type:
python label_crossings.py --demo_type=DEMO_TYPE
"""
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--single_demo", help="View and label a single demo", default=False, type=str)
parser.add_argument("--demo_name", help="Name of demo if single demo.", default="", type=str)
parser.add_argument("--clear", help="Remove crossings info for the given demo_type", default=False, type=bool)
parser.add_argument("--label_ends", help="Label the ends of the rope as start and finish for traversal", default=False, type=bool)
parser.add_argument("--verify", help="Check existence & content of crossings datasets for the given demo_type.", default=False, type=bool)

"""
Mark the displayed image with a colored circle, and save the x,y coords
and the type (right or left) of the click to the crossings array.
A left-click signifies an over-crossing, while a right-click signifies
an under-crossing.
"""
def mark_crossing(event,x,y,flags,param):
    print event
    if event == cv2.EVENT_LBUTTONUP: #left-click, overcrossing
        cv2.circle(param[0],(x,y),5,(0,200,200),-1)
        param[1].append([x,y,1])
    elif event == cv2.EVENT_RBUTTONUP: #right-click, undercrossing
        cv2.circle(param[0],(x,y),5,(200,170,50),-1)
        param[1].append([x,y,-1])

"""
Mark the displayed image with a colored circle, and save the x,y coords
and the type (right or left) of the click to the rope_ends array.
A left-click signifies the "finishing" end of the rope, while a right-click
signifies the "starting/beginning" end of the rope.
Over- and under-crossings are defined in relation to the starting and
finishing ends of the rope. An under-crossing is a crossing where a
traversal of the rope from the starting end to the finishing end would
first encounter the crossing on the lower segment, followed by the upper
segment. An over-crossing is a crossing where a traversal of the rope
from the starting end to the finishing end would first encounter the
crossing on the lower segment, followed by the upper segment.
"""
def mark_end(event,x,y,flags,param):
    print event
    if event == cv2.EVENT_LBUTTONUP: #left-click, finishing end
        cv2.circle(param[0],(x,y),5,(0,200,200),-1)
        param[1].append([x,y,1])
    elif event == cv2.EVENT_RBUTTONUP: #right-click, beginning end
        cv2.circle(param[0],(x,y),5,(200,170,50),-1)
        param[1].append([x,y,-1])

"""
Iterate through the demos in the h5 file denoted by demo_type, and call
label_single_demo on each.
If demo_name is not null, iterate only over the segments of demo_name.
The parameter verify indicates whether to check that every segment in
the h5 file was labeled.
"""
def label_crossings(hdf, demo_name):
    if demo_name != "":
        print "labeling single demo"
        for seg in hdf[demo_name].keys():
            seg_group = hdf[demo_name][seg]
            if "crossings" in seg_group.keys():
                del seg_group["crossings"]
            ret = label_single_demo(seg_group, demo_name+seg, mark_crossing, "crossings")
            while ret == "retry":
                ret = label_single_demo(seg_group, demo_name+seg, mark_crossing, "crossings")
            if ret == "quit":
                return
    else:
        for demo in hdf.keys():
            for seg in hdf[demo].keys():
                seg_group = hdf[demo][seg]
                if "crossings" in seg_group.keys():
                    print "crossings already labeled for", demo, seg
                    break
                ret = label_single_demo(seg_group, demo+seg, mark_crossing, "crossings")
                while ret == "retry":
                    ret = label_single_demo(seg_group, demo+seg, mark_crossing, "crossings")
                if ret == "quit":
                    return

"""
Unfinished - unneeded?
"""
def label_ends(hdf, demo_name):
    if demo_name != "":
        print "labeling single demo"
        for seg in hdf[demo_name].keys():
            seg_group = hdf[demo_name][seg]
            if "ends" in seg_group.keys():
                del seg_group["ends"]
            ret = label_single_demo(seg_group, demo_name+seg, mark_end, "ends")
            while ret == "retry":
                label_single_demo(seg_group, demo_name+seg, mark_end, "ends")
            if ret == "quit":
                return
    else:
        for demo in hdf.keys():
            for seg in hdf[demo].keys():
                seg_group = hdf[demo][seg]
                if "ends" in seg_group.keys():
                    print "ends already labeled for", demo, seg
                    continue
                ret = label_single_demo(seg_group, demo+seg, mark_end, "ends")
                while ret == "retry":
                    ret = label_single_demo(seg_group, demo+seg, mark_end, "ends")
                if ret == "quit":
                    return
                elif ret == "restart":
                    return "restart"

"""
Set up a cv2 window displaying the image corresponding to demo and seg
in the hdf file, and label the x,y coordinates and click type on click
by calling mark_crossing or mark_ends.
"""
def label_single_demo(seg_group, name, on_mouse, dataset):
    print name
    points = []
    image = np.asarray(seg_group['rgb'])
    x = 0
    windowName = dataset + " " + name
    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, on_mouse, (image, points))
    while(1):
        cv2.imshow(windowName, image)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('r'):
            return "retry"
        elif k == ord(' '):
            break
        elif k == ord('p'):
            return "restart"
        elif k == 27:
            return "quit"
    print points
    points = np.asarray(points)
    print points
    seg_group.create_dataset(dataset, shape(len(points), 3), data=points)
    print seg_group[dataset]
    cv2.destroyAllWindows()

def verify_crossings(hdf):
    crossings_types = {}
    pattern_locations = {}
    max = 0
    for demo in hdf.keys():
        for seg in hdf[demo].keys():
            print demo, seg
            if "crossings" in hdf[demo][seg].keys():
                pattern = []
                for item in hdf[demo][seg]['crossings']:
                    pattern.append(item[2])
                pattern = tuple(pattern)
                if pattern in crossings_types:
                    crossings_types[pattern] += 1
                else:
                    crossings_types[pattern] = 1
                pattern_locations[pattern] = (demo, seg)
                max = (len(hdf[demo][seg]['crossings']), demo, seg)
    print "max crossings vector length is", max
    print crossings_types
    for pattern in crossings_types:
        if crossings_types[pattern] < 3:
            redo = raw_input("Pattern " + str(pattern) + " is rare. Reexamine? (y/n): ")
            if redo == 'y' or redo == 'Y' or redo == "yes" or redo == "Yes":
                demo, seg = pattern_locations[pattern]
                seg_group = hdf[demo][seg]
                del seg_group['crossings']
                ret = label_single_demo(seg_group, demo+seg, mark_crossing, 'crossings')
                while ret == "retry":
                    label_single_demo(seg_group, demo+seg, mark_crossing, 'crossings')
                if ret == "quit":
                    return


def remove_data(hdf, dataset):
    for demo in hdf.keys():
        for seg in hdf[demo].keys():
            if dataset not in hdf[demo][seg].keys():
                break
            else:
                del hdf[demo][seg][dataset]


def refactor(hdf):
    for demo in hdf.keys():
        for seg in hdf[demo].keys():
            points = []
            for crossing in hdf[demo][seg]['crossings']:
                if crossing[2] == 0:
                    points.append((crossing[0], crossing[1], -1))
                elif crossing[2] == 1:
                    points.append(crossing)
            del hdf[demo][seg]['crossings']
            points = np.asarray(points)
            hdf[demo][seg].create_dataset('crossings', data=points)


if __name__ == "__main__":

    args = parser.parse_args()
    print args
    demo_type = args.demo_type
    clear = args.clear
    should_label_ends = args.label_ends
    demo_name = args.demo_name
    verify = args.verify
    h5filename = osp.join(demo_files_dir, demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    #refactor(hdf)

    if clear:
        if should_label_ends:
            dataset = "ends"
        else:
            dataset = "crossings"
        confirm = raw_input("Really delete all "+dataset+" info for "+demo_type+"? (y/n):")
        if confirm == 'y':
            print "clearing", dataset, "data for", demo_type
            remove_data(hdf, dataset)
        else:
            print "Canceled."
    elif should_label_ends:
        print "labeling ends"
        ret = label_ends(hdf, demo_name)
        while ret == "restart":
            ret = label_ends(hdf, demo_name)
    elif verify:
        print "verifying"
        verify_crossings(hdf)
    else:
        print "labeling crossings"
        label_crossings(hdf, demo_name)
