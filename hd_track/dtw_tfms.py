#!/usr/bin/env python
import numpy as np
import cPickle
import argparse
from hd_track import optimal_path

parser = argparse.ArgumentParser()
parser.add_argument('--read', help="Name of log file to read", required=True, type=str)
parser.add_argument('--write', help="Name of log file to write", required=True, type=str)
args = parser.parse_args()

dic = cPickle.load(open(args.read))




tool_tfms = dic['Ts_bg']
ar_tfms = dic['Ts_ba']
hydra_tfms  = dic['Ts_bh']


cost = optimal_path.compute_cost(hydra_tfms, ar_tfms)

path = optimal_path.optimal_warp_path(cost)


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
#dic['Ts_ba'] = 
#dic['Ts_bh'] = 
dic['Ts_bg'] = tool_tfms
#cPickle.dump(dic, open(args.write, "wa"))

