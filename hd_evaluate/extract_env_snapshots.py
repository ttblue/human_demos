#!/usr/bin/env python
"""
Script to visualize the rope and robot state in openrave viewer.
"""
import argparse

import os, numpy as np, h5py, time, os.path as osp
import cPickle as cp
import numpy as np
import matplotlib.pyplot as plt

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
import multiprocessing


import openravepy as rave
env = rave.Environment()   
env.Load('robots/pr2-beta-static.zae')
pr2 = env.GetRobots()[0]

    

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

        fname = osp.join(save_dir, 'composite.jpg')
        print "saving at  :", fname
        scipy.misc.imsave(fname, composite_mat)
        for snapshot_fname, snapshot_mat in snapshots.items():
            scipy.misc.imsave(osp.join(save_dir, snapshot_fname), snapshot_mat)  


def extract_snapshots_on_cloud(demo_type, core_type):
    """
    runs snapshot extraction on the cloud and saves the result on local machine. 
    """
    demo_testing_dir = osp.join(testing_results_dir, demo_type)
    env_state_files  = find_recursive(demo_testing_dir, '*.cp')
    
    state_infos = []
    for env_state_file in env_state_files[0:2]:
        with open(env_state_file,"r") as fh:
            seg_info = cp.load(fh)['seg_info']
    
        if seg_info == None:
            continue

        save_dir = osp.join(osp.dirname(env_state_file),  'snapshots', osp.splitext(osp.basename(env_state_file))[0])        
        state_infos.append((seg_info, save_dir))

    print colorize("calling on cloud..", "yellow", True)
    jids = cloud.map(get_state_snapshots, state_infos, _env='RSS3', _type=core_type)
    res  = cloud.result(jids)
    print colorize("got snapshots from cloud for : %s. Saving..."%demo_type, "green", True)
    save_snapshots_from_cloud(res)



def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    # draw the renderer
    fig.canvas.draw ( ) 
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
    buf = np.roll ( buf, 3, axis = 2 )
    return buf
    """
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def get_image(rope_cloud, left, right):
    plt.clf()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    mid = np.mean(rope_cloud, axis=0)
    rope_cloud -= mid[None,:]
    left -= mid
    right -= mid
    plt.scatter([left[0], right[0]], [left[1], right[1]], color='r', s=100)
    plt.hold(True)
    plt.plot(rope_cloud[:,0], rope_cloud[:,1])
    
    #plt.scatter([left[0], right[0]], [left[1], right[1]], 'r')
    #plt.hold(False)
    plt.axis('equal')
    return plt.gcf()
    
    
def do_plot(info):
    plt.axis('off')
    rope_cloud, left, right = info
    mid = np.mean(rope_cloud, axis=0)
    rope_cloud -= mid[None,:]
    left -= mid
    right -= mid
    plt.scatter([left[0], right[0]], [left[1], right[1]], color='r', s=50)
    plt.hold(True)
    plt.plot(rope_cloud[:,0], rope_cloud[:,1])
    plt.axis('equal')

ii = 0
def render_local(state_info):
    global pr2, ii
    print ii
    ii += 1

    seg_info, save_dir, fname = state_info
    num_segs           = len(seg_info)

    snapshots = {}
    svals = []
    composite = None
    for i in xrange(num_segs):
        num_mini_segs = len(seg_info[i])
        for j in xrange(num_mini_segs):
            robot_tfm, dofs, rope_nodes = seg_info[i][j]
            pr2.SetDOFValues(dofs, range(len(dofs)))
            ltf = pr2.GetManipulator('leftarm').GetEndEffectorTransform()[:3,3]
            rtf = pr2.GetManipulator('rightarm').GetEndEffectorTransform()[:3,3]
    
            #env_img = fig2data(get_image(rope_nodes, ltf, rtf))
            
            #snapshots['%d_%d.jpg'%(i,j)] = env_img
            svals.append((rope_nodes, ltf, rtf))
            
    num_rows = int(math.ceil(len(svals)/(4+0.0)))
    n =1
    plt.clf()
    plt.gcf().set_size_inches(25,12)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for i in xrange(len(svals)):
        plt.subplot(num_rows, 4, n)
        do_plot(svals[i])
        n +=1

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    comp_fname = osp.join(save_dir, fname)
    print "saving at  :", comp_fname
    
    plt.savefig(comp_fname)

    
def extract_snapshots_matplotlib_cloud(demo_type, core_type):
    """
    runs snapshot extraction on the cloud and saves the result on local machine. 
    """
    demo_testing_dir = osp.join(testing_results_dir, demo_type)
    env_state_files  = find_recursive(demo_testing_dir, '*.cp')
    
    state_infos = []
    for env_state_file in env_state_files:
        with open(env_state_file,"r") as fh:
            seg_info = cp.load(fh)['seg_info']
    
        if seg_info == None:
            continue

        save_dir = osp.join(osp.dirname(env_state_file),  'snapshots')        
        state_infos.append((seg_info, save_dir, osp.splitext(osp.basename(env_state_file))[0]+'.jpg'))

    p = multiprocessing.Pool(8)
    p.map(render_local, state_infos)
    
            
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Snapshots on Cloud")
    parser.add_argument("--core_type", type=str)
    parser.add_argument("--demo_type", type=str)
    args = parser.parse_args()

    #extract_snapshots_on_cloud(args.demo_type, args.core_type)
    extract_snapshots_matplotlib_cloud(args.demo_type, args.core_type)
