#!/usr/bin/env python
"""
Script to visualize the rope and robot state in openrave viewer.
"""
import argparse
usage="Script to generate snapshots of the environment state\n\tpython save_env_snapshots --demo_type_dir=dir"
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument("--demo_type", type=str)
args = parser.parse_args()

import os, numpy as np, h5py, time, os.path as osp
import cPickle as cp
import numpy as np
from numpy.linalg import norm
import glob
from RopePR2Viz import RopePR2Viz
import scipy, math

from hd_utils import yes_or_no
from hd_utils.utils import make_perp_basis
from hd_utils.colorize import *
from hd_utils.defaults import demo_files_dir, hd_data_dir, cad_files_dir
import time
import  hd_rapprentice.cv_plot_utils as cpu

rope_pr2_viz = RopePR2Viz()
#env_state_dir   = osp.join(demo_files_dir, args.demo_type, 'test_env_states')
env_state_dir = '/home/ankush'
env_state_files = glob.glob(osp.join(env_state_dir, "*.cp"))


for env_state_file in env_state_files:
    snapshot_dir = osp.splitext(env_state_file)[0]
    
    if not osp.exists(snapshot_dir):
        os.mkdir(snapshot_dir)
    
    state_dat = cp.load(open(env_state_file,"r"))
    snapshots = []
    
    num_segs  = len(state_dat['seg_info'])
    for i in xrange(num_segs):
        num_mini_segs = len(state_dat['seg_info'][i])
        for j in xrange(num_mini_segs):

            robot_tfm, robot_dofs, rope_nodes = state_dat['seg_info'][i][j]
            rope_pr2_viz.set_robot_pose(robot_dofs, robot_tfm)

            #if i==0 and j==0:
            rope_pr2_viz.update_rope(rope_nodes)
            #else:
            #    rope_pr2_viz.update_rope_links(rope_nodes)

            env_img = rope_pr2_viz.get_env_snapshot()
            env_img_fname = osp.join(snapshot_dir, '%d_%d.jpg'%(i,j))
            scipy.misc.imsave(env_img_fname, env_img)
            snapshots.append(env_img)
            raw_input()

    bigimg = cpu.tile_images(snapshots, int(math.ceil(len(snapshots)/4.0)), 4, max_width=2500)
    bigimg_fname = osp.join(snapshot_dir, 'composite.jpg')
    scipy.misc.imsave(bigimg_fname, bigimg)
