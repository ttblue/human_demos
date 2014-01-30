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
from RopePR2Viz import RopePR2Viz
import scipy, math

from hd_utils import yes_or_no
from hd_utils.utils import make_perp_basis
from hd_utils.colorize import *
from hd_utils.defaults import testing_results_dir
import time
import  hd_rapprentice.cv_plot_utils as cpu

import cloud
from hd_utils.utils import find_recursive



def get_state_snapshots(state_info):
    rope_pr2_viz       = RopePR2Viz()
    seg_info, save_dir = state_info
    num_segs           = len(seg_info)

    snapshots = {}
    composite = None
    for i in xrange(num_segs):
        num_mini_segs = len(seg_info[i])
        for j in xrange(num_mini_segs):

            robot_tfm, robot_dofs, rope_nodes = seg_info[i][j]
            rope_pr2_viz.set_robot_pose(robot_dofs, robot_tfm)

            if i==0 and j==0:
                rope_pr2_viz.update_rope(rope_nodes)
            else:
                rope_pr2_viz.update_rope_links(rope_nodes)

            env_img = rope_pr2_viz.get_env_snapshot()
            snapshots['%d_%d.jpg'%(i,j)] = env_img
            
    composite = cpu.tile_images(snapshots, int(math.ceil(len(snapshots)/4.0)), 4, max_width=2500)
    return (composite, snapshots, save_dir)


def save_snapshots_from_cloud(cloud_results):
    for res in cloud_results:
        composite_mat, snapshots, save_dir = res
        
        if not osp.exists(save_dir):
            os.makedirs(save_dir)
        
        scipy.misc.imsave(osp.join(save_dir, 'composite.jpg'), composite_mat)
        for snapshot_fname, snapshot_mat in snapshots.items():
            scipy.misc.imsave(osp.join(save_dir, snapshot_fname), snapshot_mat)  


def extract_snapshots_on_cloud(demo_type, core_type='c2'):
    """
    runs snapshot extraction on the cloud and saves the result on local machine. 
    """
    demo_testing_dir = osp.join(testing_results_dir, demo_type)
    env_state_files  = find_recursive(demo_testing_dir, '*.cp')
    
    state_infos = []
    for env_state_file in env_state_files:
        seg_info = cp.load(open(env_state_file,"r"))['seg_info']
        save_dir = osp.join(osp.dirname(env_state_file),  'snapshots', osp.splitext(osp.basename(env_state_file))[0])

        if seg_info == None:
            continue

        state_infos.append((seg_info, save_dir))
        
    print colorize("calling on cloud..", "yellow", True)
    jids = cloud.map(get_state_snapshots, state_infos, _env='RSS3', _type=core_type)
    res  = cloud.result(jids)
    print colorize("got snapshots from cloud for : %s. Saving..."%demo_type, "green", True)
    save_snapshots_from_cloud(res)

            
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Snapshots on Cloud")
    parser.add_argument("--demo_type", type=str)
    parser.add_argument("--instance_type", type=str, default='c2')
    args = parser.parse_args()

    extract_snapshots_on_cloud(args.demo_type, args.instance_type)

    