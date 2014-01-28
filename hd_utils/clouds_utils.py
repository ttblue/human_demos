import networkx as nx, numpy as np, scipy.spatial.distance as ssd, scipy.interpolate as si
from collections import deque
import itertools
from numpy.random import rand
from hd_rapprentice.rope_initialization import find_path_through_point_cloud
import random



########### TOP LEVEL FUNCTION ###############

MIN_SEG_LEN = 3

def sample_random_rope(xyz, plotting=False, perturb_points=5, min_rad=0, max_rad=.20):
    perturb_radius = random.uniform(min_rad, max_rad)
    new_xyz = find_path_through_point_cloud(xyz, plotting=plotting, perturb_peak_dist=perturb_radius, num_perturb_points=perturb_points)
    
    
    return new_xyz


def lerp (x, xp, fp, first=None):
    """
    Returns linearly interpolated n-d vector at specified times.
    """

    fp = np.asarray(fp)

    fp_interp = np.empty((len(x),0))
    for idx in range(fp.shape[1]):
        if first is None:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx])).T
        else:
            interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx],left=first[idx])).T
        fp_interp = np.c_[fp_interp, interp_vals]
    
    return fp_interp


def scale_rope (xyz, scale_factor, n_links=None,center=True):
    """
    @xyz: (n X 3) numpy array of points, corresponding to (n-1) links.
    @scale_factor: scaling
    @n_links: number of points in output. Default is n*scale_factor.
    """
    
    if n_links is None: n_links = (xyz.shape[0]-1)*scale_factor
    
    diff_vectors = (xyz[1:,:] - xyz[:-1,:])*scale_factor
    tot_diff = np.r_[[[0,0,0]],np.cumsum(diff_vectors, axis=0)]
    new_xyz = xyz[0,:] + tot_diff

    x = np.linspace(0, new_xyz.shape[0]-1, n_links+1)
    xp = range(new_xyz.shape[0])
    
    final_xyz = lerp(x,xp,new_xyz)
    
    if center:
        old_mean = np.mean(xyz, axis=0)
        new_mean = np.mean(new_xyz, axis=0)
        final_xyz = final_xyz + old_mean - new_mean
    return final_xyz