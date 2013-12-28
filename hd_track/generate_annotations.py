#!/usr/bin/evn python
import yaml, rosbag
import argparse
import os, os.path as osp

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
                    "robot look": ["begin recording", "stop segment"],
                    "begin segment": ["robot look"],
                    "stop segment": ["begin segment", "new segment"],
                    "new segment": ["stop segment","begin recording"],
                    "stop recording": ["stop segment"],
                    "finish recording": ["stop recording", "stop segment"],
                    "cancel recording": ["stop recording", "begin recording", "robot look", "begin segment", "stop segment", "new segment"],
                    #  should never reach cancel recording.
                    "done session": ["finish recording", "cancel recording"]} #not relevant, should not reach here.

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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_type",help="Type of demonstration")
    parser.add_argument("demo_name",help="Name of demo")
    args = parser.parse_args()

    data_dir = os.getenv("HD_DATA_DIR")
    demo_dir = osp.join(data_dir, args.demo_type, args.demo_name)
    bag = rosbag.Bag(osp.join(demo_dir,'demo.bag'))
    ann_file = osp.join(demo_dir,'ann.yaml')
    
    stamps, commands = extract_segment(bag)
    demos = demos_to_annotations(stamps, commands)
    
    for (i_seg, seg_info) in enumerate(demo):
        seg_info["name"] = "seg%.2i"%i_seg
        seg_info["description"] = "(no description)"
            
    demo[-1]["description"] = "final frame"
    demo[-1]["name"] = "done"
            
    print "writing to %s"%ann_file
    with open(ann_file, "w") as fh:
        yaml.dump(demos, fh)
    print "done"
    