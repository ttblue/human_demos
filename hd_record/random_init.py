

import h5py

import scipy.spatial.distance as ssd, scipy.interpolate as si
import random
import numpy as np
from collections import deque
import networkx as nx
import itertools
h5file = "/home/henrylu/henry_sandbox/human_demos_data/overhand.h5"
import rope_initialization
from mayavi import mlab

def sample_rope_state(new_xyz, human_check=True, perturb_points=3, min_rad=0, max_rad=1):
    success = False
    while not success:
        # TODO: pick a random rope initialization
        perturb_radius = random.uniform(min_rad, max_rad)
        print perturb_radius
        rope_nodes = rope_initialization.find_path_through_point_cloud(new_xyz, plotting = False, perturb_peak_dist=perturb_radius,num_perturb_points=perturb_points)
        success = True
    return rope_nodes






def load_random_start_segment(demofile):
	pclouds = []
	for demo_name in demofile:
		if demo_name != "ar_demo":
			for seg_name in demofile[demo_name]:
				if seg_name.endswith('00'):
					#print demo_name, seg_name
					pclouds.append((np.asarray(demofile[demo_name][seg_name]["cloud_xyz"])))
	rand = random.random
	index = np.ceil(rand() * len(pclouds))
	index = 0
	return pclouds[int(index)]



def main():
	demofile = h5py.File(h5file, 'r')
	xyz = load_random_start_segment(demofile)
	xyz = np.array(xyz)
	a = xyz[:, 0]
	b = xyz[:, 1]
	c = xyz[:,2]
	mlab.figure(1)
	mlab.points3d(a, b, c,color=(1,0,0))  
	print '-----------'
	config = sample_rope_state(xyz)
	x,y,z = np.array(config).T
	mlab.figure(2)
	mlab.plot3d(x,y,z,color=(1,0,0),tube_radius=.01,opacity=.2)
	raw_input()
if __name__ == "__main__":
    main()