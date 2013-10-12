#!/usr/bin/env python
import numpy as np
import cPickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--read', help="Name of log file to read", required=True, type=str)
parser.add_argument('--write', help="Name of log file to write", required=True, type=str)
args = parser.parse_args()

print args.read
dic = cPickle.load(open(args.read))




tool_tfms = dic['tool_tfms']
tool_times_o = dic['tool_times']
tool_times = []
for t in tool_times_o:
    #print t.to_sec()
    tool_times.append(t.to_sec() * 1000000)
ar_tfms = dic['ar_tfms']
ar_times_o = dic['ar_times']
ar_times = []
for t in ar_times_o:
    ar_times.append(t * 1000000)
hydra_tfms  = dic['hydra_tfms']
hydra_times_o = dic['hydra_times']
hydra_times = []
for t in hydra_times_o:
    hydra_times.append(t.to_sec() * 1000000)



ar_tfms_nodup = [None] * len(tool_tfms)
hydra_tfms_nodup = [None] * len(hydra_tfms)
time = 0

for i in xrange(len(ar_tfms)):
    if ar_times[i] != time:
        ar_tfms_nodup[i] = ar_tfms[i]
        time = ar_times[i]

time = 0
for i in xrange(len(hydra_tfms)):
    if hydra_times[i] != time:
        hydra_tfms_nodup[i] = hydra_tfms[i]
        time = hydra_times[i]

count = 0
for h in hydra_tfms_nodup:
    if h == None:
        count = count + 1
print "hydra tfms contains " + str(count) + " None"
count = 0
for a in ar_tfms_nodup:
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
dic['Ts_ba'] = ar_tfms_nodup
dic['Ts_bh'] = hydra_tfms_nodup
dic['Ts_bg'] = tool_tfms
cPickle.dump(dic, open(args.write, "wa"))

