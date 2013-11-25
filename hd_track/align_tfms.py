#!/usr/bin/env python
import numpy as np
import cPickle
import argparse
import matplotlib.pylab as plt



parser = argparse.ArgumentParser()
parser.add_argument('--read', help="Name of log file to read", required=True, type=str)
parser.add_argument('--write', help="Name of log file to write", required=True, type=str)
args = parser.parse_args()

print args.read
dic = cPickle.load(open(args.read))

tool_tfms    = dic['tool_tfms']
tool_times_o = dic['tool_times']
tool_times = []

for t in tool_times_o:
    tool_times.append(t.to_sec())

ar_tfms    = dic['ar_tfms']
ar_times_o = dic['ar_times']
ar_times   = []

for t in ar_times_o:
    ar_times.append(t)

hydra_tfms  = dic['hydra_tfms']
hydra_times_o = dic['hydra_times']
hydra_times = []

for t in hydra_times_o:
    hydra_times.append(t.to_sec())

## visualize time-stamps
print "pr2   # time-stamps : ", len(tool_times)
print "ar    # time-stamps : ", len(ar_times) 
print "hydra # time-stamps : ", len(hydra_times) 
plt.hold(True)
plt.plot(tool_times[5:], label="pr2")
plt.plot(ar_times[45:], label="ar")
plt.plot(hydra_times[45:], label="hydra")

plt.legend()
plt.show()

ar_tfms_search = [None] * len(tool_tfms)
hydra_tfms_search = [None] * len(hydra_tfms)

# indices where ar_times would be inserted into tool_times
ar_in_tools = np.searchsorted(tool_times, ar_times)
hydra_in_tools = np.searchsorted(tool_times, ar_times)

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
        if ar_tfms_search[ar_in_tool - 1] == None:
            ar_tfms_search[ar_in_tool - 1] = ar_tfms[i]
    else:
        ar_tfms_search[ar_in_tool] = ar_tfms[i]

for i in xrange(len(hydra_in_tools)):
    hydra_in_tool = hydra_in_tools[i]
    # if before first time or last time of tool_times, just ignore
    if hydra_in_tool == 0 or hydra_in_tool == len(tool_tfms):
        continue
    # assigns the corresponding ar_tfms, ones that don't get assigned stay as None
    # if closer to the left entry, first one has priority
    # if closer to the right entry, last one has priority
    hydra_time = hydra_times[i]
    left_time = tool_times[hydra_in_tool - 1]
    right_time = tool_times[hydra_in_tool]
    closer_left = abs(right_time - hydra_time) >= abs(left_time - hydra_time)
    if closer_left:
        if hydra_tfms_search[hydra_in_tool - 1] == None:
            hydra_tfms_search[hydra_in_tool - 1] = hydra_tfms[i]
    else:
        hydra_tfms_search[hydra_in_tool] = hydra_tfms[i]

count = 0
for h in hydra_tfms_search:
    if h == None:
        count = count + 1
print "hydra tfms contains " + str(count) + " None"
count = 0
for a in ar_tfms_search:
    if a == None:
        count = count + 1
print "ar tfms contains " + str(count) + " None"


T_gh = np.array([[ 0.01000886,  0.07745693, -0.99694546,  0.04915767],
              [-0.2037757 , -0.97591593, -0.07786887, -0.00264513],
              [-0.97896644,  0.20393264,  0.00601604,  0.02789544],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])

T_ga = np.array([[-0.02269945, -0.03166374,  0.99924078,  0.0253741 ],
              [-0.99914892, -0.03371398, -0.02376569, -0.01322115],
              [ 0.0344409 , -0.99892981, -0.0308715 , -0.01401787],
              [ 0.        ,  0.        ,  0.        ,  1.        ]])


dic = {}
dic['T_gh'] = T_gh
dic['T_ga'] = T_ga
dic['Ts_ba'] = ar_tfms_search
dic['Ts_bh'] = hydra_tfms_search
dic['Ts_bg'] = tool_tfms
cPickle.dump(dic, open(args.write, "wa"))

