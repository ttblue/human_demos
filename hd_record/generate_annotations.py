#!/usr/bin/evn python
import yaml, rosbag
import argparse
import os, os.path as osp
from hd_utils.defaults import demo_files_dir

"""
Voice command meanings:

begin recording: 
    start next demonstration (basically start video and bag commands).
robot look: 
    stores time stamps for looking at point cloud for current segment. (can be recalled if needed)
begin segment: 
    start recording current segment for demonstration -- point from which human trajectory is relevant.
stop segment:
    stop recording current segment -- human trajectory from this point no longer relevant.
new segment:
    robot look + begin segment
stop recording:
    done with current demonstration
finish recording:
    save current demonstration
cancel recording:
    delete current demonstration at any point after it is started
done session: 
    finished with recording all demonstrations for current segment

"""


def extract_segment(bag):
    
    stamps = []
    commands = []
    
    message_stream = bag.read_messages(topics=['/segment'])
    
    for (_, msg, _) in message_stream:
        stamps.append(msg.header.stamp.to_sec())
        commands.append(msg.command)
        
    return stamps, commands


def demos_to_annotations(stamps, commands):
    """return many lists of dicts giving info for each segment
    [{"look": 1234, "start": 2345, "stop": 3456},...]
    """
    
    parent_state = {"begin recording": ["all start", "finish recording", "cancel recording"],
                    "robot look": ["begin recording", "stop segment", "robot look"],
                    "begin segment": ["robot look"],
                    "stop segment": ["begin segment", "new segment"],
                    "new segment": ["stop segment","begin recording"],
                    "stop recording": ["stop segment"],
                    "finish recording": ["stop recording", "stop segment"],
                    "cancel recording": ["stop recording", "begin recording", "robot look", "begin segment", "stop segment", "new segment"],
                    #  should never reach cancel recording.
                    "done session": ["finish recording", "cancel recording", "all start"]} #not relevant, should not reach here.

    state = "all start"
    demo = []
    subseq = {}
    for (stamp, command) in zip(stamps, commands):

        if state in parent_state[command]:
            if command == "begin recording":
                state = command
            elif command  == "robot look":
                subseq['look'] = stamp
                state = command
            elif command == "begin segment":
                subseq['start'] = stamp
                state = command
            elif command == "stop segment":
                subseq['stop'] = stamp
                demo.append(subseq)
                subseq = {}
                state = command
            elif command == "new segment":
                subseq['look'] = stamp
                subseq['start'] = stamp
                state = command
            elif command == "stop recording":
                state = command
            elif command == "finish recording":
                # final frame
                subseq["look"] = stamp
                subseq["start"] = stamp
                subseq["stop"] = stamp
                demo.append(dict(subseq))
                state = command
                break
            elif command == "cancel recording":
                print "SHOULD NOT REACH HERE"
                demo = []
                subseq = {}
                state = command
                break
            elif command == "done session":
                print "SHOULD NOT REACH HERE"
                state = command
                break

    return demo           

def generate_annotation(demo_type, demo_name):
    demo_dir = osp.join(demo_files_dir, demo_type, demo_name)
    bag = rosbag.Bag(osp.join(demo_dir,'demo.bag'))
    ann_file = osp.join(demo_dir,'ann.yaml')
    
    stamps, commands = extract_segment(bag)
    demos = demos_to_annotations(stamps, commands)
    
    for (i_seg, seg_info) in enumerate(demos):
        seg_info["name"] = "seg%.2i"%i_seg
        seg_info["description"] = "(no description)"
            
    demos[-1]["description"] = "final frame"
    demos[-1]["name"] = "done"
            
    print "writing to %s"%ann_file
    with open(ann_file, "w") as fh:
        yaml.dump(demos, fh)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type",help="Type of demonstration")
    parser.add_argument("--demo_name",help="Name of demo", default='', type=str)
    args = parser.parse_args()
    
    if args.demo_name != '':
        generate_annotation(args.demo_type, args.demo_name)
    else: 
        # if args.demo_name == '', run generate annotation for all the demos in the directory
        demo_type_dir = osp.join(demo_files_dir, args.demo_type)
        demo_master_file = osp.join(demo_type_dir, "master.yaml")
        
        with open(demo_master_file, 'r') as fh:
            demos_info = yaml.load(fh)

        for demo_info in demos_info["demos"]:
            generate_annotation(args.demo_type, demo_info["demo_name"])
        

    print "done annotation generation"
    