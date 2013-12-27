"""
Code to prepare the recorded demo-data so that it can be
visualized/ fed to the kalman filter etc.
"""
import numpy as np
import math
from hd_track.streamer import streamize, time_shift_stream
from hd_utils.utils import avg_transform
import hd_utils.transformations as tfms
from   hd_track.kalman import closer_angle


def correct_hydra(Ts_hydra, T_tt2hy, T_cam2hbase, f_tps):
    """
    Warps the xyz from hydra according to f_tps.
    Note the rpy are left unchanged.
    ================================

    Ts_hydra  : Tool-tip transform from hydra's estimate in cam1 frame
    T_tt2hy   : Transform from tool-tip to hydra-sensor on the gripper
    T_cam2hbase : Transform from camera1 to hydra-base
    """
    T_hbase2cam = np.linalg.inv(T_cam2hbase)
    T_hy2tt     = np.linalg.inv(T_tt2hy)

    Ts_HB = [T_hbase2cam.dot(tfm).dot(T_tt2hy) for tfm in Ts_hydra]
    N     = len(Ts_HB)
    Xs_HB = np.empty((N,3))
    for i in xrange(N):
        Xs_HB[i,:] = Ts_HB[i][0:3,3]
    Xs_aligned     = f_tps.transform_points(Xs_HB)

    Ts_aligned = []    
    for i in xrange(N):
        t_aligned = Ts_HB[i]
        t_aligned[0:3,3] = Xs_aligned[i,:]
        t_aligned_cam2tt = T_cam2hbase.dot(t_aligned).dot(T_hy2tt)
        Ts_aligned.append(t_aligned_cam2tt)

    return Ts_aligned


def load_data(dat_fname, lr):
    """
    Returns four things for the left/right (based on lr) gripper:
    
    1.  A list of tuples of (transforms, stamps).
        The first tuple is the hydra-data.
        All the rest of the tuples correspond to the n-cameras in the data-file
        (usually 1 or 2).
    2. Potentiometer angles : a tuple of (vals, stamps)
    3. The transform from camera1 to hydra-base.
    4. The transform from tool-tip to the hydra-sensor on the gripper.
    
    Assumes that the data was saved using the save_observations_rgbd function in extract_data.py
    """

    with open(dat_fname, 'r') as f:
        dat = cp.load(f)
    T_cam2hbase = dat['T_cam2hbase']
    T_tt2hy     = dat[lr]['T_tt2hy']

    ## find the number of cameras for which we have data.
    num_cameras = 0
    for kname in dat[lr].keys():
        if 'camera' in kname:
            num_cameras += 1
    
    ## extract transform data:
    tfm_data = []

    ## hydra data:
    hy_tfs = [tt[0] for tt in dat[lr]['hydra']]     
    hy_ts  = np.array([tt[1] for tt in dat[lr]['hydra']])  
    tfm_data.append((hy_tfs, hy_ts))
    
    ## camera data:
    for i in xrange(num_cameras):
        cam_name = 'camera%d'%(i+1)
        cam_tfs = [tt[0] for tt in dat[lr][cam_name]]
        cam_ts  = np.array([tt[1] for tt in dat[lr][cam_name]])
        tfm_data.append((cam_tfs, cam_ts))
        
    ## potentiometer angles:
    pot_vals = np.array([tt[0] for tt in dat[lr]['pot_angles']])
    pot_ts   = np.array([tt[1] for tt in dat[lr]['pot_angles']])

    return (tfm_data, (pot_vals, pot_ts), T_cam2hbase, T_tt2hy)



def relative_time_streams(tfm_data, freq):
    """
    Return start/end time, number of time samples, and streams for each sensor.
    
    - TFM_DATA is a list of tuples of the form : [(tfs_1, ts_1), (tfs_2, ts_2), ...]
        where,
          tfs_i : is a list of transforms
          ts_i  : is a numpy array of the time-stamps corresponding to the transforms.
    
      Note this TFM_DATA is returned as the first parameter from load_data function.
      
    - FREQ : is the frequency at which the data-streams must be streamized.
    """
    
    n_series = len(tfm_data)
    print "Found %d data-streams to streamize."%n_series
    
    dt =1./freq

    ## calculate tmin & tmax to get rid of absolute time scale:
    tmin, tmax = float('inf'), float('-inf')
    for series in tfm_data:
        _, ts = series
        if ts.any():
            tmin = min(tmin, np.min(ts))            
            tmax = max(tmax, np.max(ts)) 

    ## calculate the number of time-steps for the kalman filter.    
    nsteps = int(math.ceil((tmax-tmin)/dt))

    ## create the data-streams:
    streams = []
    for series in tfm_data:
        tfs, ts = series
        if ts.any():
            ts -= tmin
            strm = streamize(tfs, ts, freq, avg_transform)

    return tmin, tmax, nsteps, streams



def fit_spline_to_tf_stream(strm, new_freq, deg=3):
    """
    Interpolates a stream of transforms using splines, such that 
    there is no "NONE" when sampling the stream at NEW_FREQ frequency.
    
    Returns a stream of transforms.
    """
    tfs, ts = strm.get_data()
    tstart  = strm.get_strat_time()
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

    s = N*.001**2
    (tck, _) = si.splprep(tf_dat, s=s, u=ts, k=deg)

    interp_xyzrpys = np.r_[si.splev(new_ts, tck)].T

    smooth_tfms = []
    for xyzrpy in interp_xyzrpys:
        tfm = tfms.euler_matrix(*xyzrpy[3:6])
        tfm[0:3,3] = xyzrpy[0:3]
        smooth_tfms.append(tfm)
        
    return streamize(smooth_tfms, new_ts, new_freq, strm.favg, tstart)


def align_tf_streams(hydra_strm, cam_strm, wsize=20):
    """
    Calculates the time-offset b/w the camera and hydra transform streams.
    It uses a hydra-stream because it is full -- it has no missing data-points.
                                                    ===========================

      --> If that is not the case, first fit a spline to the hydra stream and then call this function.

    WSIZE : is the window-size to search in : 
            This function gives the:
                argmin_{s \in [-wsize, ..,0,..., wsize]} dist(hy_stream, cam_stream(t+s))
                where, dist is the euclidean norm (l2 norm).

    NOTE : (1) It uses only the position variables for distance calculations.
           (2) Further assumes that the two streams are on the same time-scale.       
    """
    Xs_hy    = []
    Xs_cam   = []
    cam_inds = []
    idx = 0
    for tfm in cam_strm:
        if tfm != None:
            Xs_cam.append(tfm[0:3,3])
            cam_inds.append(idx)
        idx += 1

    for hy_tfm in hydra_strm:   
        hy_tfm = hydra_strm.next()
        Xs_hy.append(hy_tfm[0:3,3])

    ## chop-off wsized data from the start and end of the camera-data:
    start_idx, end_idx = 0, len(Xs_cam)-1
    while cam_inds[start_idx] < wsize: start_idx += 1
    while cam_inds[end_idx] >= len(Xs_hy) - wsize: end_idx   -= 1

    dists    = []
    cam_inds = cam_inds[start_idx:end_idx+1]
    Xs_cam   = np.array(Xs_cam[start_idx:end_idx+1])
    for shift in xrange(-wsize, wsize+1):
        hy_xs = [Xs_hy[idx + shift] for idx in cam_inds]
        dists.append(np.linalg.norm(np.array(hy_xs) - Xs_cam))

    return xrange(-wsize, wsize+1)[np.argmin(dists)]



def align_all_streams(hy_strm, cam_streams, wsize=20):
    """
    Time aligns all-streams to camera1 stream.
    
    hy_strm     : The hydra-transforms stream. Assumes that hydra stream is full (no missing data).
    cam_streams : A list of camera-data streams.
                  The first stream is assumed to be camera1 stream.
                  All streams are aligned with camera1 (i.e. camera1 stream is not changed).
    """
    n_streams = len(cam_streams)
    tmin, tmax = float('inf'), float('-inf')
    if n_streams >= 1:
        dt_hydra   = align_tf_streams(hy_strm, cam_streams[0], wsize)
        hy_aligned = time_shift_stream(hy_strm, -dt_hydra)
        
        tmin = min(tmin, np.min(hy_aligned.ts))
        tmax = max(tmax, np.max(hy_aligned.ts))
        
        aligned_streams = []
        for i in xrange(1, n_streams):
            dt = align_tf_streams(hy_aligned, cam_streams[i], wsize)
            shifted_stream = time_shift_stream(cam_streams[i], dt)

            tmin = min(tmin, np.min(shifted_stream.ts))
            tmax = max(tmax, np.max(shifted_stream.ts))

            aligned_streams.append(shifted_stream)

        ## shift the start-times such that all the streams start at the minimum time:
        hy_aligned.set_start_time(hy_aligned.get_start_time() - tmin)
        for strm in aligned_streams:
            strm.set_start_time(strm.get_start_time() - tmin)

        nsteps = (tmax-tmin)/hy_aligned.dt
        return (tmin, tmax, nsteps, hy_aligned, aligned_streams)  
    
        