#!/usr/bin/env python
import numpy as np
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--read', help="Name of log file to read", required=True, type=str)
parser.add_argument('--write', help="Name of log file to write", required=True, type=str)
args = parser.parse_argus()

dic = cPickle.load(open(args.name, "wa"))




tool_tfms = dic['tool_tfms']
tool_times = dic['tool_times']

ar_tfms = dic['ar_tfms']
ar_times = dic['ar_times']
hydra_tfms  = dic['hydra_tfms']
hydra_times = dic['hydra_times']

ar_tfms_search = [None] * len(tool_tfms)

# indices where ar_times would be inserted into tool_times
ar_in_tools = np.searchsorted(tool_times, ar_times)



for i in xrange(len(ar_in_tools)):
    ar_in_tool = ar_in_tools[i]
    # if before first time or last time of tool_times, just ignore
    if ar_in_tool == 0 or ar_in_tool == len(tool_tfms):
        continue
    # assigns the corresponding ar_tfms, ones that don't get assigned stay as None
    # if closer to the left entry, first one has priority
    # if closer to the right entry, last one has priority
    ar_time = ar_times[i]
    left_time = tool_times[ar_in_tool - 1]
    right_time = tool_times[ar_in_tool]
    closer_left = abs(right_time - ar_time) >= abs(left_time - ar_time)
    if closer_left:
        if ar_tfms_search[ar_in_tool - 1] != None:
            ar_tfms_search[ar_in_tool - 1] = ar_tfms[i]
    else:
        ar_tfms_search[ar_in_tool] = ar_tfms[i]

print ar_tfms_search





