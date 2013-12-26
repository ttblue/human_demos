#!/usr/bin/evn python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("bagfile")
parser.add_argument("outfile")
args = parser.parse_args()

import yaml, rosbag



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
                    "new segment": ["stop segment"],
                    "stop recording": ["stop segment"],
                    "finish recording": ["stop recording"],
                    "cancel recording": ["stop recording", "begin recording", "robot look", "begin segment", "stop segment", "new segment"]}
    
    out = []
    
    state = "all start"
    
    demo = []
    subseq = {}
    for (stamp, command) in zip(stamps, commands):

        if command == "begin recording":
            if state in parent_state[command]:
                state = command
        if command  == "robot look":
            if state in parent_state[command]:
                subseq['look'] = stamp
                state = command
        if command == "begin segment":
            if state in parent_state[command]:
                subseq['start'] = stamp
                state = command
        if command == "stop segment":
            if state in parent_state[command]:
                subseq['stop'] = stamp
                demo.append(dict(subseq))
                subseq = {}
                state = command
        if command == "new segment":
            if state in parent_state[command]:
                subseq['look'] = stamp
                subseq['start'] = stamp
                state = command
        if command == "stop recording":
            if state in parent_state[command]:
                state = command
        if command == "finish recording":
            if state in parent_state[command]:
                # final frame
                subseq["look"] = stamp
                subseq["start"] = stamp
                subseq["stop"] = stamp
                demo.append(dict(subseq))
                
                out.append(list(demo))
                demo = []
                state = command
        if command == "cancel recording":
            if state in parent_state[command]:
                demo = []
                subseq = {}
                state = command

        
    return out           


bag = rosbag.Bag(args.bagfile)
stamps, commands = extract_segment(bag)
demos = demos_to_annotations(stamps, commands)

for demo in demos:
    for (i_seg, seg_info) in enumerate(demo):
        seg_info["name"] = "seg%.2i"%i_seg
        seg_info["description"] = "(no description)"
        
    demo[-1]["description"] = "done"

print demos
        
print "writing to %s"%args.outfile
with open(args.outfile, "w") as fh:
    yaml.dump(demos, fh)
print "done"
        
    









        
        
        
        
