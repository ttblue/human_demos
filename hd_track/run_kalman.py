from __future__ import division

import numpy as np
import os, os.path as osp
import cPickle as cp
import scipy.linalg as scl
import math, sys
import argparse 
import matplotlib.pylab as plt

from hd_utils.colorize import *
from hd_utils.conversions import *
from hd_utils.utils import *
from hd_utils.defaults import demo_files_dir, demo_names, master_name
from hd_utils.yes_or_no import yes_or_no

from hd_track.kalman import kalman, smoother, closer_angle
from hd_track.kalman_tuning import state_from_tfms_no_velocity
from hd_track.streamer import streamize, get_corresponding_data, stream_soft_next, time_shift_stream
from hd_track.demo_data_prep import *
from hd_track.tps_correct    import *
from scipy.signal import butter, filtfilt


def initialize_motion_covariance(freq):
    motion_xyz_std = [0.05, 0.05, 0.05] # 5cm <-- motion model is a hack => use large covariance.
    motion_rpy_std = np.deg2rad(20)
    
    ## initialize velocity covariances: divide by the filter frequency to put on the correct time-scale
    motion_vx_std  = 0.1/freq # 5cm/s 
    motion_vth_std = np.deg2rad(10/freq) # 10 deg/s
    
    I3 = np.eye(3)

    motion_covar = scl.block_diag(np.diag(np.square(motion_xyz_std)),
                                  np.square(motion_vx_std)*I3,
                                  np.square(motion_rpy_std)*I3,
                                  np.square(motion_vth_std)*I3)
    
    return motion_covar

def get_smoother_motion_covariance(freq):
    kf_motion_covar = initialize_motion_covariance(freq)
    kf_motion_covar *= 1e-1
    return kf_motion_covar

def initialize_covariances(freq, demo_dir):
    """
    Initialize empirical estimates of covariances:
    
     -- Cameras and the hydra observe just the xyzrpy (no velocities).
     -- Motion covariance is for all 12 variables.
    """
    cam_types = get_cam_types(demo_dir)
    cam_tfms = get_cam_tfms(demo_dir)
    
    
    rgbd_cam_xyz_std    = [0.01, 0.01, 0.01] # 1cm
    rgb_cam_xyz_std     = [0.05, 0.05, 0.05] # 1cm
    hydra_xyz_std       = [0.08, 0.08, 0.08] # 3cm <-- use small std after tps-correction.

    rgbd_cam_rpy_std    = np.deg2rad(15)
    rgb_cam_rpy_std     = np.deg2rad(15)
    hydra_rpy_std       = np.deg2rad(5)


    I3 = np.eye(3)
    
    rgbd_covar  = scl.block_diag(np.diag(np.square(rgbd_cam_xyz_std)), np.square(rgbd_cam_rpy_std)*I3)
    rgb_covar   = scl.block_diag(np.diag(np.square(rgb_cam_xyz_std)), np.square(rgb_cam_rpy_std)*I3)
    hydra_covar = scl.block_diag(np.diag(np.square(hydra_xyz_std)), np.square(hydra_rpy_std)*I3)

    
    cam_covars = {}
    
    for cam in cam_types:
        print cam
        if cam == 'camera1':
            if cam_types[cam] == 'rgb':
                cam_covars[cam] = rgb_covar
            else:
                cam_covars[cam] = rgbd_covar
        else:
            for i in xrange(len(cam_tfms)):
                tfm_info = cam_tfms[i]
                if tfm_info['parent'] == 'camera1_link' and tfm_info['child'] == '%s_link'%(cam):
                    R = scl.block_diag(tfm_info['tfm'][:3,:3], I3)
                    if cam_types[cam] == 'rgb':
                        cam_covars[cam] = R.dot(rgb_covar).dot(R.transpose())
                    else:
                        cam_covars[cam] = R.dot(rgbd_covar).dot(R.transpose())
                    break
    
    motion_covar = initialize_motion_covariance(freq)
    
    return (motion_covar, cam_covars, hydra_covar)


def get_first_state(tf_streams, freq, start_time):
    """
    Returns the first state and covariance given :
     - TF_STREAMS : a list of streams of transform data.
     - FREQ       : the frequency of these streams (and also the filter).
    """
    dt = 1./freq
    n_streams = len(tf_streams)

    tfs0 = []
    for i in xrange(n_streams):
        tfs, ts = tf_streams[i].get_data()
        for ti, t in enumerate(ts):
            if t <= dt + start_time:
                tfs0.append(tfs[ti])
                
                
    I3 = np.eye(3)
    S0 = scl.block_diag(1e-3*I3, 1e-2*I3, 1e-3*I3, 1e-3*I3)
                
    if len(tfs0)==0:
        redprint("Cannot find initial state for KF: no data found within dT in all streams")
        x0 = state_from_tfms_no_velocity([])
    else:    
        x0 =  state_from_tfms_no_velocity([avg_transform(tfs0)])

    return (x0, S0)


def plot_tf_streams(tf_strms, strm_labels, styles=None, title=None, block=True):
    """
    Plots the x,y,z,r,p,y from a list TF_STRMS of streams.
    """
    assert len(tf_strms)==len(strm_labels)
    if styles!=None:
        assert len(tf_strms)==len(styles)
    else:
        styles = ['.']*len(tf_strms)

    plt.figure()
    ylabels = ['x', 'y', 'z', 'r', 'p', 'y']
    n_streams = len(tf_strms)
    Xs   = []
    inds = []
    for strm in tf_strms:
        tfs, ind = [], []
        for i,tf in enumerate(strm):
            if tf != None:
                tfs.append(tf)
                ind.append(i)
        X = state_from_tfms_no_velocity(tfs, 6)
        Xs.append(X)
        inds.append(ind)

    for i in xrange(6):
        plt.subplot(2,3,i+1)
        plt.hold(True)
        for j in xrange(n_streams):
            xj = Xs[j]
            ind_j = inds[j]
            plt.plot(ind_j, xj[:,i], styles[j], label=strm_labels[j])
            plt.ylabel(ylabels[i])
        plt.legend()

    if title!=None:
        plt.gcf().suptitle(title)
    plt.show(block=block)



def load_demo_data(demo_dir, freq, rem_outliers, tps_correct, tps_model_fname, plot, block):
    cam_dat = {}
    hydra_dat  = {}
    pot_dat = {}
    dt      = 1./freq
    lr_full   = {'l': 'left', 'r':'right'}

    data_file = osp.join(demo_dir, demo_names.data_name)

    T_cam2hbase, cam_dat['l'], hydra_dat['l'], pot_dat['l'] = load_data(data_file, 'l', freq)
    _, cam_dat['r'], hydra_dat['r'], pot_dat['r'] = load_data(data_file, 'r', freq)
    
    ## collect all camera streams in a list:
    all_cam_names = set()
    for lr in 'lr':
        for cam in cam_dat[lr].keys():
            all_cam_names.add(cam)


    ## time-align all tf-streams (wrt their respective hydra-streams):
    ## NOTE THE STREAMS ARE MODIFIED IN PLACE (the time-stamps are changed)
    ## and also the tstart
    blueprint("Getting rid of absolute time..")
    all_cam_strms = []
    for lr in 'lr':
        for cam in cam_dat[lr].keys():
            all_cam_strms.append(cam_dat[lr][cam]['stream'])
    tmin, _, _ = relative_time_streams(hydra_dat.values() + pot_dat.values() + all_cam_strms, freq)
    tshift1 = -tmin

    ## remove outliers in the camera streams
    # plot command: o
    if rem_outliers:
        blueprint("Rejecting TF outliers in camera-data..")
        for lr in 'lr':
            for cam in cam_dat[lr].keys():
                strm_in = reject_outliers_tf_stream(cam_dat[lr][cam]['stream'])
                if 'o' in plot:
                    blueprint("\t Plotting outlier rejection..")
                    cam_name = cam+'_'+lr
                    plot_tf_streams([cam_dat[lr][cam]['stream'], strm_in], strm_labels=[cam_name, cam_name+'_in'], styles=['.','-'], block=block)
                cam_dat[lr][cam]['stream'] = strm_in


    ## do time-alignment (with the respective l/r hydra-streams):
    blueprint("Calculating TF stream time-shifts..")
    time_shifts = {}   ## dictionary : maps a camera-name to the time-shift wrt hydra
    for cam in all_cam_names:
        ndat = {'r':-1, 'l':-1}  ## compare whether left/right has more data-points
        for lr in 'lr':
            if cam in cam_dat[lr].keys():
                ndat[lr] = len(cam_dat[lr][cam]['stream'].get_data()[0])
       
        lr_align = ndat.keys()[np.argmax(ndat.values())]

        time_shifts[cam] = dt* align_tf_streams(hydra_dat[lr_align], cam_dat[lr_align][cam]['stream'])

    greenprint("Time-shifts found : %s"%str(time_shifts))
    
    ## time-shift the streams:
    # plot command: t
    all_cam_strms = []
    blueprint("Time-aligning TF streams..")
    for lr in 'lr':
        redprint("\t Alignment for : %s"%lr_full[lr])
        aligned_cam_strms = []
        for cam in cam_dat[lr].keys():
            aligned_cam_strms.append( time_shift_stream(cam_dat[lr][cam]['stream'], time_shifts[cam]) )

        if 't' in plot:
            unaligned_cam_streams = []
            for cam in cam_dat[lr].values():
                unaligned_cam_streams.append(cam['stream'])
            
            blueprint("\t plotting unaligned TF streams...")
            plot_tf_streams(unaligned_cam_streams + [hydra_dat[lr]], cam_dat[lr].keys()+['hydra'], title='UNALIGNED CAMERA-streams (%s)'%lr_full[lr], block=block)
            blueprint("\t plotting aligned TF streams...")
            plot_tf_streams(aligned_cam_strms+[hydra_dat[lr]], cam_dat[lr].keys()+['hydra'], title='ALIGNED CAMERA-streams (%s)'%lr_full[lr], block=block)
            
        for i,cam in enumerate(cam_dat[lr].keys()):
            cam_dat[lr][cam]['stream'] = aligned_cam_strms[i]
            all_cam_strms.append(cam_dat[lr][cam]['stream']) ###### SIBI'S CHANGE


    ## put the aligned-streams again on the same time-scale: 
    blueprint("Re-aligning the TF-streams after TF based time-shifts..")
    tmin, tmax, nsteps = relative_time_streams(hydra_dat.values() + pot_dat.values() + all_cam_strms, freq)
    tshift2 = -tmin
    greenprint("TOTAL TIME-SHIFT : (%0.3f + %0.3f) = %0.3f"%(tshift1, tshift2, tshift1+tshift2))

    ## TPS-correct the hydra-data:
    if tps_correct:
        tps_models = {'l':None, 'r':None}
        if tps_model_fname == '':
            blueprint("\t Fitting TPS model to demo data..")
            for lr in 'lr':
                blueprint("\t TPS fitting for %s"%lr_full[lr])
                if 'camera1' not in cam_dat[lr].keys():
                    redprint("\t\t camera1 not in the data for %s gripper -- Cannot do tps-fit. Skipping.."%lr_full[lr])
                    continue

                _, hydra_tfs, cam1_tfs = get_corresponding_data(hydra_dat[lr], cam_dat[lr]['camera1']['stream'])
                f_tps = fit_tps_on_tf_data(hydra_tfs, cam1_tfs, plot=False) ## <<< mayavi plotter has some issues with multiple insantiations..
                T_cam2hbase_train = T_cam2hbase
                tps_models[lr]    = f_tps

                tps_save_fname = osp.join(demo_dir, 'tps_models.cp')
                with open(tps_save_fname, 'w') as f:
                    cp.dump([tps_models, T_cam2hbase], f)
        else:
            blueprint("\t Getting TPS-model from : %s"%tps_model_fname)
            with open(tps_model_fname, 'r') as f:
                tps_models, T_cam2hbase_train = cp.load(f)

        blueprint("TPS-correcting hydra-data..")
        # plot command: p
        for lr in 'lr':
            if tps_models[lr] != None:
                hydra_tfs, hydra_ts  = hydra_dat[lr].get_data()
                hydra_tfs_aligned = correct_hydra(hydra_tfs, T_cam2hbase, tps_models[lr], T_cam2hbase_train)
                hydra_strm_aligned = streamize(hydra_tfs_aligned, hydra_ts, 1./hydra_dat[lr].dt, hydra_dat[lr].favg, hydra_dat[lr].tstart)

                if 'p' in plot:
                    if tps_model_fname!=None:
                        plot_tf_streams([hydra_dat[lr], hydra_strm_aligned], ['hydra-old', 'hydra-corr'], title='hydra-correction %s'%lr_full[lr], block=block)
                    else:
                        plot_tf_streams([hydra_dat[lr], hydra_strm_aligned, cam_dat[lr]['camera1']['stream']], ['hydra-old', 'hydra-corr', 'cam1'], title='hydra-correction', block=block)

                hydra_dat[lr] = hydra_strm_aligned
                
    ## return the data:
    filter_data = {'demo_dir': demo_dir,
                   'nsteps' : nsteps,
                   'tmin'   : tmin,
                   'tmax'   : tmax,
                   'pot_dat': pot_dat,
                   'hydra_dat': hydra_dat,
                   'cam_dat': cam_dat,
                   't_shift': tshift1+tshift2,
                   'cam_shifts': time_shifts}
    return filter_data


def prepare_kf_data(demo_dir, freq, rem_outliers, tps_correct, tps_model_fname, plot, block):
    '''
    freq: default 30
    rem_outlier: default True
    tps_correct: default True
    tps_model_fname: default None
    plot: default ''
    '''
    filter_data = load_demo_data(demo_dir, 
                                 freq,
                                 rem_outliers,
                                 tps_correct,
                                 tps_model_fname,
                                 plot,
                                 block)
    
    ann_file = osp.join(demo_dir, demo_names.ann_name) 
    with open(ann_file) as f:
        ann_dat = yaml.load(f)

    time_shifts = filter_data['cam_shifts']
    for strm_name in time_shifts.keys():
        time_shifts[strm_name] += filter_data['t_shift']

    rec_data = {'l':None, 'r':None}
    for lr in 'lr':
        hydra_strm   = filter_data['hydra_dat'][lr]
        cam_strms = filter_data['cam_dat'][lr]
        pot_strm  = filter_data['pot_dat'][lr]
        
        strms = [hydra_strm, pot_strm]
        
        for cam in cam_strms.keys():
            strms.append(cam_strms[cam]['stream'])

        seg_streams, nsteps, shifted_seg_start_times = segment_streams(ann_dat, strms, time_shifts, filter_data['demo_dir'], base_stream='camera1')
        
        n_segs = len(seg_streams)

        seg_data = []
        for i in xrange(n_segs):

            ## get camera streams for the segment in question:
            seg_cam_strms = {}
            for j,cam in enumerate(cam_strms.keys()):
                seg_cam_strms[cam] = {'type'   : cam_strms[cam]['type'],
                                      'stream' : seg_streams[i][2+j]}

            seg = {'name': ann_dat[i]['name'],
                   'hydra_strm'  : seg_streams[i][0],
                   'pot_strm' : seg_streams[i][1],
                   'cam_dat'  : seg_cam_strms,
                   'nsteps'   : nsteps[i],
                   'shifted_seg_start_times': shifted_seg_start_times[i]}

            seg_data.append(seg)

        rec_data[lr] = seg_data

    return rec_data, time_shifts, n_segs


def initialize_KFs(kf_data, freq):
    """
    Initializes the Kalman filters (one-each for left/right x segment).
    KF_DATA : Data as returned by prepare_kf_data function above. 
    """
    KFs = {'l':None, 'r':None}
    motion_covar = initialize_motion_covariance(freq)

    n_segs = len(kf_data['r'])
    for lr in 'lr':
        KF_lr = []
        for i in xrange(n_segs):
            hydra_strm   = kf_data[lr][i]['hydra_strm']
            cam_dat   = kf_data[lr][i]['cam_dat']
            cam_strms= []
            for cinfo in cam_dat.values():
                cam_strms.append(cinfo['stream'])
                


            x0, S0 = get_first_state(cam_strms + [hydra_strm], freq=1./hydra_strm.dt, start_time=kf_data[lr][i]['shifted_seg_start_times'])

            KF = kalman()
            kf_tstart = hydra_strm.get_start_time()
            KF.init_filter(kf_tstart, x0, S0, motion_covar)
            KF_lr.append(KF)
        KFs[lr] = KF_lr
    return KFs


def low_pass(x, freq=30.):
    """
    removes high-frequency content from an array X.
    can be used to smooth out the kalman filter estimates.
    """
    fs      = freq  # sampling freq (Hz)
    lowcut  = 0.0   # lower-most freq (Hz)
    highcut = 3.0   # higher-most freq (Hz)

    nyq = 0.5 * fs  # nyquist frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(4, high, btype='low')
    
    x_filt = filtfilt(b, a, x, axis=0)
    x_filt = np.squeeze(x_filt)
    return  x_filt


def unbreak(rpy):
    """
    removes discontinuity in rpy (nx3 matrix).
    """
    if len(rpy)==1: return
    
    un_rpy      = np.empty(rpy.shape)
    un_rpy[0,:] = rpy[0,:]
    for i in xrange(1,len(rpy)):
        if i==1:
            a_prev = rpy[i-1,:]
        else:
            a_prev = un_rpy[i-1,:]
        a_now  = rpy[i,:]
        un_rpy[i,:] = closer_angle(a_now, a_prev)
    return un_rpy


def run_KF(KF, nsteps, freq, hydra_strm, cam_dat, hydra_covar, cam_covars, do_smooth, plot, plot_title, block):
    """
    Runs the Kalman filter/smoother for NSTEPS using
    {hydra/rgb/rgbd}_covar as the measurement covariances.
    
    HYDRA_STRM   : TF stream from hydra
    CAM_DAT   : Dictionary of camera streams 
    do_smooth: default true
    plot: default true
    """ 
    
    dt = 1./freq
    
    ## place holders for kalman filter's output:
    xs_kf, covars_kf, ts_kf = [KF.x_filt],[KF.S_filt],[KF.t_filt]

    hydra_snext  = stream_soft_next(hydra_strm)

    cam_strms = [cinfo['stream'] for cinfo in cam_dat.values()]
    cam_names = cam_dat.keys()
    cam_types = [cinfo['type'] for cinfo in cam_dat.values()]
    cam_snext = [stream_soft_next(cstrm) for cstrm in cam_strms]
    
    for i in xrange(nsteps):
        KF.register_tf_observation(hydra_snext(), hydra_covar, do_control_update=True)

        for i in xrange(len(cam_names)):
            cam_covar = cam_covars[cam_names[i]]
            KF.register_tf_observation(cam_snext[i](), cam_covar, do_control_update=False)

        xs_kf.append(KF.x_filt)
        covars_kf.append(KF.S_filt)
        ts_kf.append(KF.t_filt)
        
    hydra_strm.reset()
    for cstrm in cam_strms:
        cstrm.reset()
    if do_smooth:

        '''
        # UNCOMMENT BELOW TO RUN KALMAN SMOOTHER:
        A,_  = KF.get_motion_mats(dt)
        R    = get_smoother_motion_covariance(freq)
        xs_smthr, covars_smthr = smoother(A, R, xs_kf, covars_kf)
        '''
        ### do low-pass filtering for smoothing:
        xs_kf_xyz = np.array(xs_kf)[:,0:3] 
        xs_kf_rpy = np.squeeze(np.array(xs_kf)[:,6:9])

        xs_lp_xyz = low_pass(xs_kf_xyz, freq)
        xs_lp_rpy = low_pass(unbreak(xs_kf_rpy), freq)

        xs_smthr  = np.c_[xs_lp_xyz, np.squeeze(np.array(xs_kf))[:,3:6], xs_lp_rpy, np.squeeze(np.array(xs_kf))[:,9:12]].tolist()
        covars_smthr = None

        if 's' in plot:
            kf_strm   = streamize(state_to_hmat(xs_kf), np.array(ts_kf), 1./hydra_strm.dt, hydra_strm.favg)
            sm_strm   = streamize(state_to_hmat(xs_smthr), np.array(ts_kf), 1./hydra_strm.dt, hydra_strm.favg)
            plot_tf_streams([kf_strm, sm_strm, hydra_strm]+cam_strms, ['kf', 'smoother', 'hydra']+cam_dat.keys(), styles=['-','-','.','.','.','.'], title=plot_title, block=block)
            #plot_tf_streams([hydra_strm]+cam_strms, ['hydra']+cam_dat.keys(), block=block)

        return (ts_kf, xs_kf, covars_kf, xs_smthr, covars_smthr)
    else:
        if 's' in plot:
            kf_strm   = streamize(state_to_hmat(xs_kf), np.array(ts_kf), 1./hydra_strm.dt, hydra_strm.favg)
            plot_tf_streams([kf_strm, hydra_strm]+cam_strms, ['kf', 'hydra']+cam_dat.keys(), styles=['-','-','.','.','.','.'], title=plot_title, block=block)
            #plot_tf_streams([hydra_strm]+cam_strms, ['hydra']+cam_dat.keys(), block=block)
        return (ts_kf, xs_kf, covars_kf, None, None)

def filter_traj(demo_dir, tps_model_fname, save_tps, do_smooth, plot, block):
    """
    Runs the kalman filter for BOTH the grippers and writes the demo.traj file.
    TPS_MODEL_FNAME : The name of the file to load the tps-model from
    """
    # Temp file to show that kalman filter/smoother is already being run on demo
    with open(osp.join(demo_dir, demo_names.run_kalman_temp),'w') as fh: fh.write('Running Kalman filter/smoother..')
    
    freq = 30.0
    
    rec_data, time_shifts, n_segs = prepare_kf_data(demo_dir,
                                                    freq=freq,
                                                    rem_outliers=True,
                                                    tps_correct=True, 
                                                    tps_model_fname=tps_model_fname,
                                                    plot=plot,
                                                    block=block)

    print time_shifts

    _, cam_covars, hydra_covar =  initialize_covariances(freq, demo_dir)

    KFs  = initialize_KFs(rec_data, freq)
    traj = {'l' : None, 'r':None}

    for lr in 'lr':
        lr_trajs = {}
        for iseg in xrange(n_segs):
            KF = KFs[lr][iseg]
            hydra_strm  = rec_data[lr][iseg]['hydra_strm']
            pot_strm = rec_data[lr][iseg]['pot_strm']
            cam_dat  = rec_data[lr][iseg]['cam_dat']
            nsteps   = rec_data[lr][iseg]['nsteps']


            ts, xs_kf, covars_kf, xs_smthr, covars_smthr = run_KF(KF, nsteps, freq,
                                                                  hydra_strm, cam_dat,
                                                                  hydra_covar, cam_covars,
                                                                  do_smooth,
                                                                  plot, plot_title='seg %d : %s'%(iseg, {'l':'left', 'r':'right'}[lr]),
                                                                  block=block)

            for i in range(len(ts)):
                ts[i] -= time_shifts['camera1']
                                
            Ts_kf, Ts_smthr = state_to_hmat(xs_kf), state_to_hmat(xs_smthr)
            
            pot_angles = []
            pot_ss = stream_soft_next(pot_strm)
            for _ in xrange(nsteps+1):
                pot_angles.append(pot_ss())             

            
            '''''
            Dirty hack!!!!!!!!!!!!!!!!!!!!!!!!!
            '''''
            n_pot_angles = len(pot_angles)
            if pot_angles[0] == None or np.isnan(pot_angles[0]):
                # find the first non None
                first_non_none_id = -1
                for i in xrange(n_pot_angles):
                    if pot_angles[i] != None and not np.isnan(pot_angles[i]):
                        first_non_none_id = i
                        break
                    
                if first_non_none_id != -1:
                    for i in xrange(first_non_none_id):
                        pot_angles[i] = pot_angles[first_non_none_id]
                else:
                    for i in xrange(n_pot_angles):
                        pot_angles[i] = 0
                    
            if pot_angles[-1] == None or np.isnan(pot_angles[-1]):
                first_non_none_id = -1
                for i in xrange(n_pot_angles - 1, -1, -1):
                    if pot_angles[i] != None and not np.isnan(pot_angles[i]):
                        last_non_none_id = i
                        break
                    
                if first_non_none_id != -1:
                    for i in xrange(last_non_none_id+1, n_pot_angles):
                        pot_angles[i] = pot_angles[last_non_none_id]
                else:
                    for i in xrange(n_pot_angles):
                        pot_angles[i] = 0
                    
        
            # then linear interpolation between non-None elements
            #print n_pot_angles
            #print len(ts)
            #print nsteps
            i = 0
            while i < n_pot_angles:
                if pot_angles[i] == None or np.isnan(pot_angles[i]):
                    non_none_id_0 = i - 1
                    
                    for j in xrange(i+1, n_pot_angles):
                        if pot_angles[j] != None and not np.isnan(pot_angles[j]):
                            non_none_id_1 = j
                            break
                    
                    delta = (pot_angles[non_none_id_1] - pot_angles[non_none_id_0]) / (non_none_id_1 - non_none_id_0)   
                    for j in xrange(non_none_id_0 +1, non_none_id_1):
                        pot_angles[j] = (j - non_none_id_0) * delta + pot_angles[non_none_id_0]
                        
                    #print pot_angles[non_none_id_0: non_none_id_1+1]
                    
                    i = non_none_id_1 + 1
                else:
                    i += 1
            '''
            Finish dirty hack.
            '''

            seg_traj = {"stamps"  : ts,
                        "tfms"    : Ts_kf,
                        "covars"  : covars_kf,
                        "tfms_s"  : Ts_smthr,
                        "covars_s": covars_smthr,
                        "pot_angles": pot_angles}
            
            
            seg_name = rec_data[lr][iseg]["name"]
            lr_trajs[seg_name] = seg_traj
            
        traj[lr] = lr_trajs

    traj_fname = osp.join(demo_dir, demo_names.traj_name)
    with open(traj_fname, 'wb') as f:
        cp.dump(traj, f)
    yellowprint("Saved %s."%demo_names.traj_name)
    
    os.remove(osp.join(demo_dir, demo_names.run_kalman_temp))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type", help="Type of demonstration")
    parser.add_argument("--demo_name", help="Name of demo", default='', type=str)
    parser.add_argument("--save_tps", help="Save tpf correction file", action='store_false', default=True)
    parser.add_argument("--rem_outlier", help="remove outlier", action='store_false', default=True)
    parser.add_argument("--do_smooth", help="perform smoothing", action='store_false', default=True)
    parser.add_argument("--plot", help="plot commands (plots you want to see)", default='')
    parser.add_argument("--tps_fname", help="tps file name to be used", default='', type=str)
    parser.add_argument("--block", help="block plotting", action='store_true', default=False)
    parser.add_argument("--overwrite", help="overwrite if demo.traj already exists", action='store_true', default=False)

    args = parser.parse_args()

    demo_type_dir = osp.join(demo_files_dir, args.demo_type)
    demo_master_file = osp.join(demo_type_dir, master_name)

    with open(demo_master_file, 'r') as fh:
        demos_info = yaml.load(fh)

    if args.demo_name == '':
        for demo in demos_info["demos"]:
            demo_dir = osp.join(demo_type_dir, demo["demo_name"])
            # Wait until extraction is done for current demo.
            while osp.isfile(osp.join(demo_dir, demo_names.extract_data_temp)):
                time.sleep(1)
            # Check if some other node is running kf/ks currently.
            if osp.isfile(osp.join(demo_dir, demo_names.run_kalman_temp)):
                yellowprint("Another node seems to be running kf/ks already for %s."%demo["demo_name"]) 
                continue
            # Check if traj already exists
            if not osp.isfile(osp.join(demo_type_dir, demo["demo_name"], demo_names.traj_name)) or args.overwrite:
                yellowprint("Saving %s."%demo["demo_name"])
                filter_traj(demo_dir, tps_model_fname=args.tps_fname, save_tps=args.save_tps, do_smooth=args.do_smooth, plot=args.plot, block=args.block)
            else:
                yellowprint("Trajectory file exists for %s. Not overwriting."%demo["demo_name"])

    else:
        demo_dir = osp.join(demo_type_dir, args.demo_name)
        if osp.exists(demo_dir):
            # Wait until extraction is done for current demo.
            while osp.isfile(osp.join(demo_dir, demo_names.extract_data_temp)):
                time.sleep(1)
            # Check if some other node is running kf/ks currently.
            if not osp.isfile(osp.join(demo_dir, demo_names.run_kalman_temp)):
                # Check if .traj file already exists
                if osp.isfile(osp.join(demo_type_dir, args.demo_name, demo_names.traj_name)):
                    if args.overwrite or yes_or_no('Trajectory file already exists for this demo. Overwrite?'):
                        yellowprint("Saving %s."%args.demo_name)
                        filter_traj(demo_dir, tps_model_fname=args.tps_fname, save_tps=args.save_tps, do_smooth=args.do_smooth, plot=args.plot, block=args.block)
                else:
                    yellowprint("Saving %s."%args.demo_name)
                    filter_traj(demo_dir, tps_model_fname=args.tps_fname, save_tps=args.save_tps, do_smooth=args.do_smooth, plot=args.plot, block=args.block)
            else:
                yellowprint("Another node seems to be running kf/ks already for %s."%args.demo_name)

    if args.plot and args.block == False:
        raw_input()