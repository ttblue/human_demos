"""
Resample time serieses to reduce the number of datapoints
"""
from __future__ import division
import numpy as np
import scipy.interpolate as si
import openravepy

import hd_utils.math_utils as mu

from hd_rapprentice import LOG

def lerp(a, b, fracs):
    "linearly interpolate between a and b"
    fracs = fracs[:,None]
    return a*(1-fracs) + b*fracs

def adaptive_resample(x, t = None, max_err = np.inf, max_dx = np.inf, max_dt = np.inf, normfunc = None):
    """
    SLOW VERSION
    """
    #return np.arange(0, len(x), 100)
    LOG.info("resampling a path of length %i", len(x))
    x = np.asarray(x)
    
    if x.ndim == 1: x = x[:,None]
    else: assert x.ndim == 2
    
    if normfunc is None: normfunc = np.linalg.norm
    
    if t is None: t = np.arange(len(x))
    else: t = np.asarray(t)
    assert(t.ndim == 1)
    
    n = len(t)
    assert len(x) == n

    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(xrange(n))
    for i_src in xrange(n):
        for i_targ in xrange(i_src+1, n):
            normdx = normfunc(x[i_targ] - x[i_src])
            dt = t[i_targ] - t[i_src]
            errs = lerp(x[i_src], x[i_targ], (t[i_src:i_targ+1] - t[i_src])/dt) - x[i_src:i_targ+1]
            interp_err = errs.__abs__().max()#np.array([normfunc(err) for err in errs]).max()
            if normdx > max_dx or dt > max_dt or interp_err > max_err: break
            g.add_edge(i_src, i_targ)
    resample_inds = nx.shortest_path(g, 0, n-1)
    return resample_inds


def adaptive_resample2(x, tol, max_change=-1, min_steps=3):
    
    print type(x)
    print x.shape
    """
    resample original signal with a small number of waypoints so that the the sparsely sampled function, 
    when linearly interpolated, deviates from the original function by less than tol at every time

    input:
    x: 2D array in R^(t x k)  where t is the number of timesteps
    tol: tolerance. either a single scalar or a vector of length k
    max_change: max change in the sparsely sampled signal at each timestep
    min_steps: minimum number of timesteps in the new trajectory. (usually irrelevant)

    output:
    new_times, new_x

    assuming that the old signal has times 0,1,2,...,len(x)-1
    this gives the new times, and the new signal
    """

    x = np.asarray(x)
    assert x.ndim == 2

    if np.isscalar(tol): 
        tol = np.ones(x.shape[1])*tol
    else:
        tol = np.asarray(tol)
        assert tol.ndim == 1 and tol.shape[0] == x.shape[1]

    times = np.arange(x.shape[0])

    if max_change == -1: 
        max_change = np.ones(x.shape[1]) * np.inf
    elif np.isscalar(max_change): 
        max_change = np.ones(x.shape[1]) * max_change
    else:
        max_change = np.asarray(max_change)
        assert max_change.ndim == 1 and max_change.shape[0] == x.shape[1]

    dl = mu.norms(x[1:] - x[:-1],1)
    l = np.cumsum(np.r_[0,dl])

    def bad_inds(x1, t1):
        ibad = np.flatnonzero( (np.abs(mu.interp2d(l, l1, x1) - x) > tol).any(axis=1) )
        jbad1 = np.flatnonzero((np.abs(x1[1:] - x1[:-1]) > max_change[None,:]).any(axis=1))
        if len(ibad) == 0 and len(jbad1) == 0: return []
        else:
            lbad = l[ibad]
            jbad = np.unique(np.searchsorted(l1, lbad)) - 1
            jbad = np.union1d(jbad, jbad1)
            return jbad

    l1 = np.linspace(0,l[-1],min_steps)
    for _ in xrange(20):
        x1 = mu.interp2d(l1, l, x)
        bi = bad_inds(x1, l1)
        if len(bi) == 0:
            return np.interp(l1, l, times), x1
        else:
            l1 = np.union1d(l1, (l1[bi] + l1[bi+1]) / 2 )

    raise Exception("couldn't subdivide enough. something funny is going on. check your input data")


def get_velocities(positions, times, tol):
    positions = np.atleast_2d(positions)
    n = len(positions)
    deg = min(3, n - 1)
    
    good_inds = np.r_[True,(abs(times[1:] - times[:-1]) >= 1e-6)]
    good_positions = positions[good_inds]
    good_times = times[good_inds]
    
    if len(good_inds) == 1:
        return np.zeros(positions[0:1].shape)
    
    (tck, _) = si.splprep(good_positions.T,s = tol**2*(n+1), u=good_times, k=deg)
    #smooth_positions = np.r_[si.splev(times,tck,der=0)].T
    velocities = np.r_[si.splev(times,tck,der=1)].T    
    return velocities

def smooth_positions(positions, tol):
    times = np.arange(len(positions))
    positions = np.atleast_2d(positions)
    n = len(positions)
    deg = min(3, n - 1)
    
    good_inds = np.r_[True,(abs(times[1:] - times[:-1]) >= 1e-6)]
    good_positions = positions[good_inds]
    good_times = times[good_inds]
    
    if len(good_inds) == 1:
        return np.zeros(positions[0:1].shape)
    
    (tck, _) = si.splprep(good_positions.T,s = tol**2*(n+1), u=good_times, k=deg)
    spos = np.r_[si.splev(times,tck,der=0)].T
    return spos

def unif_resample(x,n,weights,tol=.001,deg=3):    
    x = np.atleast_2d(x)
    weights = np.atleast_2d(weights)
    x = mu.remove_duplicate_rows(x)
    x_scaled = x * weights
    dl = mu.norms(x_scaled[1:] - x_scaled[:-1],1)
    l = np.cumsum(np.r_[0,dl])
    
    (tck,_) = si.splprep(x_scaled.T,k=deg,s = tol**2*len(x),u=l)
    
    newu = np.linspace(0,l[-1],n)
    out_scaled = np.array(si.splev(newu,tck)).T
    out = out_scaled/weights
    return out

def test_resample():
    import fastrapp

    x = [0,0,0,1,2,3,4,4,4]
    t = range(len(x))
    inds = adaptive_resample(x, max_err = 1e-5)
    assert inds == [0, 2, 6, 8]
    inds = adaptive_resample(x, t=t, max_err = 0)
    assert inds == [0, 2, 6, 8]
    print "success"


    inds1 = fastrapp.resample(np.array(x)[:,None], t, 0, np.inf, np.inf)
    print inds1
    assert inds1.tolist() == [0,2,6,8]

def test_resample_big():
    import fastrapp

    from time import time
    t = np.linspace(0,1,1000)
    x0 = np.sin(t)[:,None]
    x = x0 + np.random.randn(len(x0), 50)*.1
    tstart = time()
    inds0 = adaptive_resample(x, t=t, max_err = .05, max_dt = .1)
    print time() - tstart, "seconds"

    print "now doing cpp version"
    tstart = time()
    inds1 = fastrapp.resample(x, t, .05, np.inf, .1)
    print time() - tstart, "seconds"
    
    assert np.allclose(inds0, inds1)

def interp_quats(newtimes, oldtimes, oldquats):
    "should actually do slerp"
    quats_unnormed = mu.interp2d(newtimes, oldtimes, oldquats)
    return mu.normr(quats_unnormed)
        

def interp_hmats(newtimes, oldtimes, oldhmats):
    oldposes = openravepy.poseFromMatrices(oldhmats)
    newposes = np.empty((len(newtimes), 7))
    newposes[:,4:7] = mu.interp2d(newtimes, oldtimes, oldposes[:,4:7])
    newposes[:,0:4] = interp_quats(newtimes, oldtimes, oldposes[:,0:4])
    newhmats = openravepy.matrixFromPoses(newposes)
    return newhmats

if __name__ == "__main__":
    test_resample()
    test_resample_big()