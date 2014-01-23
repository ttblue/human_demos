"""
Code to prepare the recorded demo-data so that it can be
visualized/ fed to the kalman filter etc.
"""
import numpy as np
import math, sys
import cPickle as cp
import scipy.interpolate as si
import yaml
import os.path as osp

from hd_utils.utils import avg_transform
import hd_utils.transformations as tfms
from hd_utils.colorize import blueprint, redprint
from hd_utils.defaults import demo_names

from hd_track.kalman import closer_angle
from hd_track.streamer import streamize, segment_stream

def get_cam_types(demo_dir):
    cam_type_fname = osp.join(demo_dir, demo_names.camera_types_name)
    with open(cam_type_fname, 'r') as f:
        cam_types_raw = yaml.load(f)

    cam_types = {}
    for k,v in cam_types_raw.iteritems():
        cam_types['camera%d'%k] = v

    return cam_types

def get_cam_tfms(demo_dir):
    calib_fname = osp.join(demo_dir, demo_names.calib_name)
    with open(calib_fname, 'r') as f:
        calib_dat = cp.load(f)

    return calib_dat['transforms']

def load_data(data_file, lr, freq=30.0):

    with open(data_file, 'r') as f:
        dat = cp.load(f)

    demo_dir    = osp.dirname(data_file)
    cam_types   = get_cam_types(demo_dir)
    T_cam2hbase = dat['T_cam2hbase']

    cam_info = {}
    for kname in dat[lr].keys():
        if 'cam' in kname:
            #if kname != 'camera1': continue
            tfs = [tt[0] for tt in dat[lr][kname]]
            ts  = [tt[1] for tt in dat[lr][kname]]
            #ctype_name = int(kname[-1])
            ## don't append any empty-streams:
            if len(ts) > 0:
                cam_strm = streamize(tfs, ts, freq, avg_transform)#, tstart=-1./freq)
                cam_info[kname] = {'type'   : cam_types[kname],
                                   'stream' : cam_strm}

    ## hydra data:
    hydra_tfs = [tt[0] for tt in dat[lr]['hydra']]     
    hydra_ts  = np.array([tt[1] for tt in dat[lr]['hydra']])

    if len(hydra_ts) <= 0:
        redprint("ERROR : No hydra data found in : %s"%(osp.basename(data_file)))
        sys.exit(-1)   
    hydra_strm = streamize(hydra_tfs, hydra_ts, freq, avg_transform)#, tstart=-1./freq)

    ## potentiometer angles:
    pot_vals = np.array([tt[0] for tt in dat[lr]['pot_angles']])
    pot_ts   = np.array([tt[1] for tt in dat[lr]['pot_angles']])
    if len(pot_ts) <= 0:
        redprint("ERROR : No potentiometer data found in : %s"%(osp.basename(data_file)))
        sys.exit(-1)
    pot_strm = streamize(pot_vals, pot_ts, freq, np.mean)#, tstart=-1./freq)


    return (T_cam2hbase, cam_info, hydra_strm, pot_strm)

def segment_streams(ann_dat, strms, time_shifts, demo_dir, base_stream='camera1'):
    """
    Segments a list of streams based on the timings in the annotation file
    and the time-shift data.

    STRMS      : A list of streams
    TIME_SHIFT : A dictionary of time-shifts applied to the streams wrt to the raw streams
    DEMO_DIR   : The demo-directory : used to get the annotation file
    BASE_STRM  : The stream relative to which the annotation times are specified

    RETURNS : a list of list of streams. len(list) == number of segments
              start_times: the shifted segment start times
    """
    if base_stream not in time_shifts.keys():
        redprint("Cannot segment streams. Base time shift data is missing for : %s"%base_stream)

    n_segs = len(ann_dat)
    start_times = np.array([seg_info['look']+time_shifts[base_stream] for seg_info in ann_dat])
    stop_times  = np.array([seg_info['stop']+time_shifts[base_stream] for seg_info in ann_dat])
    nsteps      = map(int, (stop_times - start_times) / strms[0].dt) 
    
    strm_segs = []
    for strm in strms:
        strm_segs.append(segment_stream(strm, start_times, stop_times))

    out_segs = []
    for i in xrange(n_segs):
        if ann_dat[i]['name'] == 'done':
            continue
        si = []
        for n in xrange(len(strms)):
            si.append(strm_segs[n][i])
        out_segs.append(si)
    
    return out_segs, nsteps, start_times


def relative_time_streams(strms, freq):
    """
    Return start/end time, number of time samples, and streams for each sensor. 
    - FREQ : is the frequency at which the data-streams must be streamized.
    
    >>>>>> ASSUMES ALL THE STREAMS HAVE AT-LEAST ONE TRANSFORM in THEM. <<<<<<<
    
    The streams are modified in place.
    """
    n_strms = len(strms)
    dt =1./freq

    ## calculate tmin & tmax to get rid of absolute time scale:
    tmin, tmax = float('inf'), float('-inf')
    for strm in strms:
        _, ts = strm.get_data()
        tmin = min(tmin, np.min(ts))            
        tmax = max(tmax, np.max(ts)) 

    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))

    ## create the data-streams:
    for strm in strms:
        strm.ts -= tmin
        strm.tstart = -dt
        strm.reset() 

    return tmin, tmax, nsteps





def fit_spline_to_tf_stream(strm, new_freq, deg=3):
    """
    Interpolates a stream of transforms using splines, such that 
    there is no "NONE" when sampling the stream at NEW_FREQ frequency.
    
    Returns a stream of transforms.
    """
    tfs, ts = strm.get_data()
    tstart  = strm.get_start_time()
    tmax    = ts[-1]
    
    ndt = 1./new_freq
    new_ts  = np.arange(tstart, tmax+ndt/4., ndt/2.) 

    ## get data in xyzrpy format (6xn) matrix:
    N = len(tfs)
    tf_dat = np.empty((6, N))
    
    tf_dat[0:3,0] = tfs[0][0:3,3]    
    tf_dat[3:6,0] = tfms.euler_from_matrix(tfs[0])
    for i in xrange(1,N):
        now_tf  = tfs[i]
        tf_dat[0:3,i] = now_tf[0:3,3]
        
        prev_rpy  = tf_dat[3:6,i-1]
        now_rpy   = tfms.euler_from_matrix(now_tf)
        now_rpy   = closer_angle(now_rpy, prev_rpy)
        
        tf_dat[3:6,i] = now_rpy

    blueprint("\t fitting spline to data (scipy) ..")
    s = N*.001**2
    (tck, _) = si.splprep(tf_dat, s=s, u=ts, k=deg)

    blueprint("\t evaluating spline at new time-stamps ..")
    interp_xyzrpys = np.r_[si.splev(new_ts, tck)].T

    smooth_tfms = []
    for xyzrpy in interp_xyzrpys:
        tfm = tfms.euler_matrix(*xyzrpy[3:6])
        tfm[0:3,3] = xyzrpy[0:3]
        smooth_tfms.append(tfm)
        
    return streamize(smooth_tfms, new_ts, new_freq, strm.favg, tstart)


def align_tf_streams_slow(hydra_strm, cam_strm, wsize=0):
    """
    Calculates the time-offset b/w the camera and hydra transform streams.
    It uses a hydra-stream because it is full -- it has no missing data-points.
                                                    ===========================

      --> If that is not the case, first fit a spline to the hydra stream and then call this function.

    WSIZE : is the window-size to search in : 
            This function gives the:
                argmin_{s \in [-wsize, ..,0,..., wsize]} dist(hydra_stream, cam_stream(t+s))
                where, dist is the euclidean norm (l2 norm).

    NOTE : (1) It uses only the position variables for distance calculations.
           (2) Further assumes that the two streams are on the same time-scale.       
    """
    Xs_hy    = []
    Xs_cam   = []
    cam_inds = []
    idx = 0
    
    ## pre-empt if the stream is small:
    if len(cam_strm.get_data()[0]) <= wsize:
        return 0

    for tfm in cam_strm:
        if tfm != None:
            Xs_cam.append(tfm[0,3])
            cam_inds.append(idx)
        idx += 1

    for hydra_tfm in hydra_strm:   
        Xs_hy.append(hydra_tfm[0,3])

    ## chop-off wsized data from the start and end of the camera-data:
    start_idx, end_idx = 0, len(Xs_cam)-1
    while cam_inds[start_idx] < wsize and start_idx < len(Xs_cam): start_idx += 1
    while cam_inds[end_idx] >= len(Xs_hy) - wsize and end_idx >= 0: end_idx -= 1

    dists    = []
    cam_inds = cam_inds[start_idx:end_idx+1]
    Xs_cam   = np.array(Xs_cam[start_idx:end_idx+1])
    for shift in xrange(-wsize, wsize+1):
        hydra_xs = [Xs_hy[idx + shift] for idx in cam_inds]
        dists.append(np.linalg.norm(np.array(hydra_xs) - Xs_cam))

    shift = xrange(-wsize, wsize+1)[np.argmin(dists)]
    redprint("\t stream time-alignment shift is : %d (= %0.3f seconds)"%(shift,hydra_strm.dt*shift))
    return shift


def align_tf_streams(hydra_strm, cam_strm, wsize=0):
    """
    Calculates the time-offset b/w the camera and hydra transform streams.
    It uses a hydra-stream because it is full -- it has no missing data-points.
                                                    ===========================

      --> If that is not the case, first fit a spline to the hydra stream and then call this function.

    WSIZE : is the window-size to search in : 
            This function gives the:
                argmin_{s \in [-wsize, ..,0,..., wsize]} dist(hydra_stream, cam_stream(t+s))
                where, dist is the euclidean norm (l2 norm).

    NOTE : (1) It uses only the position variables for distance calculations.
           (2) Further assumes that the two streams are on the same time-scale.       
    """
    Xs_hy    = []
    Xs_cam   = []
    cam_inds = []
    idx = 0

    ## pre-empt if the stream is small:
    if len(cam_strm.get_data()[0]) <= wsize:
        return 0

    for tfm in cam_strm:
        if tfm != None:
            Xs_cam.append(tfm[0,3])
            cam_inds.append(idx)
        idx += 1

    for hydra_tfm in hydra_strm:   
        if hydra_tfm != None:
            Xs_hy.append(hydra_tfm[0,3])
        else:
            redprint("None hydra tfm!")

    ## chop-off wsized data from the start and end of the camera-data:
    start_idx, end_idx = 0, len(Xs_cam)-1
    while cam_inds[start_idx] < wsize and start_idx < len(Xs_cam) : start_idx += 1
    while cam_inds[end_idx] >= len(Xs_hy) - wsize and end_idx >= 0: end_idx -= 1

    dists    = []
    cam_inds = np.array(cam_inds[start_idx:end_idx+1])
    Xs_cam   = np.array(Xs_cam[start_idx:end_idx+1])
    Xs_hy    = np.array(Xs_hy)
    for shift in xrange(-wsize, wsize+1):
        dists.append(np.linalg.norm(Xs_hy[shift + cam_inds,:] - Xs_cam))

    shift = xrange(-wsize, wsize+1)[np.argmin(dists)]
    redprint("\t stream time-alignment shift is : %d (= %0.3f seconds)"%(shift,hydra_strm.dt*shift))
    return shift



def reject_outliers_tf_stream(strm, n_window=10, v_th=[0.35,0.35,0.25]):
    """
    Rejects transforms from a stream of TF based on
    outliers in the x,y or z coordinates.
    
    Rejects based on threshold on the median velocity computed over a window.
    
    STRM : The stream of transforms to filter
    N_WINDOW: The number of neighbors of a data-point to consider
              (note the neighbors can be far away in time, if the stream is sparse)
    V_TH : The threshold velocity. A data-point is considered an outlier, if v > vth
           V_TH is specified for x,y,z
    
    Note: N_WINDOW/2 number of data-points at the beginning and
          the end of the stream are not filtered.
    """
    s_window = int(n_window/2)
    tfs, ts  = strm.get_data()
    N = len(tfs)
    Xs = np.empty((N,3))
    for i in xrange(N):
        Xs[i,:] = tfs[i][0:3,3]
    
    n = N-2*s_window
    
    ## when stream is very short, just return the original stream
    if n <= 0:
        return strm
    
    v = np.empty((n, 3, 2*s_window))
    
    X0 = Xs[s_window:s_window+n,:]
    t0 = ts[s_window:s_window+n]
    shifts  = np.arange(-s_window, s_window+1)
    shifts  = shifts[np.nonzero(shifts)]
    for i,di in enumerate(shifts):
        dx = Xs[s_window+di:s_window+n+di,:] - X0
        dt = ts[s_window+di: s_window+n+di] - t0
        v[:,:,i] = np.abs(dx/dt[:,None])

    v_med   = np.median(v,axis=2)
    inliers = s_window + np.arange(n)[np.all(v_med <= v_th, axis=1)]
    inliers = np.r_[np.arange(s_window), inliers, np.arange(n+s_window,N)]

    tfs_in = []
    for i in inliers:
        tfs_in.append(tfs[i])
    ts_in = ts[inliers]

    return streamize(tfs_in, ts_in, 1./strm.dt, strm.favg, strm.tstart)
