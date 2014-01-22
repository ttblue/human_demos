import networkx as nx, numpy as np, scipy.spatial.distance as ssd, scipy.interpolate as si
from collections import deque
import itertools
from numpy.random import rand
from hd_rapprentice.rope_initialization import find_path_through_point_cloud
import random



########### TOP LEVEL FUNCTION ###############

MIN_SEG_LEN = 3

def sample_random_rope(xyz, plotting=False, perturb_points=5, min_rad=0, max_rad=.15):
    perturb_radius = random.uniform(min_rad, max_rad)
    new_xyz = find_path_through_point_cloud(xyz, plotting=plotting, perturb_peak_dist=perturb_radius, num_perturb_points=perturb_points)
    
    return new_xyz

