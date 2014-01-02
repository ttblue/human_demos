from __future__ import division

import numpy as np
import os, os.path as osp
import cPickle as cp
import scipy.linalg as scl
import math, sys
import matplotlib.pylab as plt

from hd_utils.colorize import *
from hd_utils.conversions import *
from hd_utils.utils import *

from hd_track.kalman import kalman, smoother
from hd_track.kalman_tuning import state_from_tfms_no_velocity

from hd_track.streamer import streamize, soft_next, get_corresponding_data

from hd_track.demo_data_prep import *
from hd_track.tps_correct    import *


def initialize_covariances(freq=30.0):
    """
    Initialize empirical estimates of covariances:
    
     -- Cameras and the hydra observe just the xyzrpy (no velocities).
     -- Motion covariance is for all 12 variables.
     
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         TODO : CHANGE here for different covariances for rgb/rgbd
     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    cam_xyz_std    = 0.01 # 1cm
    hydra_xyz_std  = 0.02 # 2cm <-- use small std after tps-correction.
    motion_xyz_std = 0.05 # 5cm <-- motion model is a hack => use large covariance.
    
    cam_rpy_std    = np.deg2rad(15)
    hydra_rpy_std  = np.deg2rad(5)
    motion_rpy_std = np.deg2rad(20)
    
    ## initialize velocity covariances: divide by the filter frequency to put on the correct time-scale
    motion_vx_std  = 0.05/freq # 5cm/s 
    motion_vth_std = np.deg2rad(10/freq) # 10 deg/s

    I3 = np.eye(3)
    rgbd_covar   =  scl.block_diag(   np.square(cam_xyz_std)*I3, np.square(cam_rpy_std)*I3  )
    rgb_covar    =  rgbd_covar 
    hydra_covar  =  scl.block_diag(   np.square(hydra_xyz_std)*I3, np.square(hydra_rpy_std)*I3  )
    motion_covar =  scl.block_diag(   np.square(motion_xyz_std)*I3,
                                      np.square(motion_vx_std)*I3,
                                      np.square(motion_rpy_std)*I3,
                                      np.square(motion_vth_std)*I3  )

    return (motion_covar, rgb_covar, rgbd_cavar, hydra_covar)


def get_first_state(tf_streams, freq=30.):
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
            if t <= dt:
                tfs0.append(tfs[ti])

    if len(tfs0)==0:
        redprint("Cannot find initial state for KF: no data found within dT in all streams")    

    x0 =  state_from_tfms_no_velocity([avg_transform(tfs0)])
    I3 = np.eye(3)
    S0 = scl.block_diag(1e-3*I3, 1e-2*I3, 1e-3*I3, 1e-3*I3)
    return (x0, S0)


def plot_tf_streams(tf_strms, strm_labels, styles=None, title=None, block=True):
    """
    Plots the x,y,z,r,p,y from a list TF_STRMS of streams.
    """
    assert len(tf_strms)==len(strm_labels)
    if styles!=None:
        assert len(tf_strms)==len(styles)
    else:
        styles = ['-']*len(tf_strms)

    plt.figure()
    ylabels = ['x', 'y', 'z', 'r', 'p', 'y']
    n_streams = len(tf_strms)
    Xs   = []
    inds = []
    for strm in tf_strms:
        tfs, ind = [], []
        #import IPython
        #IPython.embed()
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



def load_demo_data(demo_fname, freq=30.0, rem_outliers=True, tps_correct=True, tps_model_fname=None, plot=False):
    cam_dat = {}
    hy_dat  = {}
    pot_dat = {}
    dt      = 1./freq
    demo_dir  = osp.dirname(demo_fname)
    lr_full   = {'l': 'left', 'r':'right'}

    T_cam2hbase, cam_dat['l'], hy_dat['l'], pot_dat['l'] = load_data(demo_fname, 'l', freq)
    T_cam2hbase, cam_dat['r'], hy_dat['r'], pot_dat['r'] = load_data(demo_fname, 'r', freq)

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
    tmin, _, _ = relative_time_streams(hy_dat.values() + pot_dat.values() + all_cam_strms, freq)
    tshift1 = -tmin


    ## remove outliers in the camera streams
    if rem_outliers:
        blueprint("Rejecting TF outliers in camera-data..")
        for lr in 'lr':
            for cam in cam_dat[lr].keys():
                strm_in = reject_outliers_tf_stream(cam_dat[lr][cam]['stream'])
                if plot and False:
                    blueprint("\t Plotting outlier rejection..")
                    cam_name = cam+'_'+lr
                    plot_tf_streams([cam_dat[lr][cam]['stream'], strm_in], strm_labels=[cam_name, cam_name+'_in'], styles=['.','-'], block=False)
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
        time_shifts[cam] = dt* align_tf_streams(hy_dat[lr_align], cam_dat[lr_align][cam]['stream'])

    greenprint("Time-shifts found : %s"%str(time_shifts))
    
    ## time-shift the streams:
    blueprint("Time-aligning TF streams..")
    for lr in 'lr':
        redprint("\t Alignment for : %s"%lr_full[lr])
        aligned_cam_strms = []
        for cam in cam_dat[lr].keys():
            aligned_cam_strms.append( time_shift_stream(cam_dat[lr][cam]['stream'], time_shifts[cam]) )

        if plot and False:
            unaligned_cam_streams = []
            for cam in cam_dat[lr].values():
                unaligned_cam_streams.append(cam['stream'])
            
            blueprint("\t plotting unaligned TF streams...")
            plot_tf_streams(unaligned_cam_streams + [hy_dat[lr]], cam_dat[lr].keys()+['hydra'], title='UNALIGNED CAMERA-streams (%s)'%lr_full[lr], block=False)
            blueprint("\t plotting aligned TF streams...")
            plot_tf_streams(aligned_cam_strms+[hy_dat[lr]], cam_dat[lr].keys()+['hydra'], title='ALIGNED CAMERA-streams (%s)'%lr_full[lr], block=False)
            
        for i,cam in enumerate(cam_dat[lr].keys()):
            cam_dat[lr][cam]['stream'] = aligned_cam_strms[i]


    ## put the aligned-streams again on the same time-scale: 
    blueprint("Re-aligning the TF-streams after TF based time-shifts..")
    tmin, tmax, nsteps = relative_time_streams(hy_dat.values() + pot_dat.values() + all_cam_strms, freq)
    tshift2 = -tmin
    greenprint("TOTAL TIME-SHIFT : (%0.3f + %0.3f) = %0.3f"%(tshift1, tshift2, tshift1+tshift2))

    ## TPS-correct the hydra-data:
    if tps_correct:
        tps_models = {'l':None, 'r':None}
        if tps_model_fname == None:
            blueprint("\t Fitting TPS model to demo data..")
            for lr in 'lr':
                blueprint("\t TPS fitting for %s"%lr_full[lr])
                if 'camera1' not in cam_dat[lr].keys():
                    redprint("\t\t camera1 not in the data for %s gripper -- Cannot do tps-fit. Skipping.."%lr_full[lr])
                    continue

                _, hy_tfs, cam1_tfs = get_corresponding_data(hy_dat[lr], cam_dat[lr]['camera1']['stream'])
                f_tps = fit_tps_on_tf_data(hy_tfs, cam1_tfs, plot=False) ## <<< mayavi plotter has some issues with multiple insantiations..
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
        for lr in 'lr':
            if tps_models[lr] != None:
                hy_tfs, hy_ts  = hy_dat[lr].get_data()
                hy_tfs_aligned = correct_hydra(hy_tfs, T_cam2hbase, tps_models[lr], T_cam2hbase_train)
                hy_strm_aligned = streamize(hy_tfs_aligned, hy_ts, 1./hy_dat[lr].dt, hy_dat[lr].favg, hy_dat[lr].tstart)

                if plot:
                    if tps_model_fname!=None:
                        plot_tf_streams([hy_dat[lr], hy_strm_aligned], ['hy-old', 'hy-corr'], title='hydra-correction %s'%lr_full[lr], block=False)
                    else:
                        plot_tf_streams([hy_dat[lr], hy_strm_aligned, cam_dat[lr]['camera1']['stream']], ['hy-old', 'hy-corr', 'cam1'], title='hydra-correction', block=True)

                hy_dat[lr] = hy_strm_aligned

    ## return the data:
    filter_data = {'demo_dir': demo_dir,
                   'nsteps' : nsteps,
                   'tmin'   : tmin,
                   'tmax'   : tmax,
                   'pot_dat': pot_dat,
                   'hy_dat': hy_dat,
                   'cam_dat': cam_dat,
                   't_shift': tshift1+tshift2,
                   'cam_shifts': time_shifts}
    return filter_data


def prepare_kf_data(demo_fname, freq=30.0, rem_outliers=True, tps_correct=True, tps_model_fname=None, plot=False):
    filter_data = load_demo_data(demo_fname, freq,
                                 rem_outliers,
                                 tps_correct,
                                 tps_model_fname,
                                 plot)

    time_shifts = filter_data['cam_shifts']
    for strm_name in time_shifts.keys():
        time_shifts[strm_name] += filter_data['t_shift']

    rec_data = {'l':None, 'r':None}
    for lr in 'lr':
        hy_strm   = filter_data['hy_dat'][lr]
        cam_strms = filter_data['cam_dat'][lr]
        pot_strm  = filter_data['pot_dat'][lr]
        
        strms = [hy_strm, pot_strm]
        for cam in cam_strms.keys():
            strms.append(cam_strms[cam]['stream'])

        seg_streams, nsteps  = segment_streams(strms, time_shifts, filter_data['demo_dir'], base_stream='camera1')
        n_segs = len(seg_streams)
    
        seg_data = []
        for i in xrange(n_segs):
            
            ## get camera streams for the segment in question:
            seg_cam_strms = {}
            for j,cam in enumerate(cam_strms.keys()):
                seg_cam_strms[cam] = {'type'   : cam_strms[cam]['type'],
                                      'stream' : seg_streams[i][2+j]}

            seg = {'hy_strm'  : seg_streams[i][0],
                   'pot_strm' : seg_streams[i][1],
                   'cam_dat'  : seg_cam_strms,
                   'nsteps'   : nsteps[i]}

            seg_data.append(seg)

        rec_data[lr] = seg_data

    return rec_data, time_shifts, demo_fname, n_segs


def initialize_KFs(kf_data):
    """
    Initializes the Kalman filters (one-each for left/right x segment).
    KF_DATA : Data as returned by prepare_kf_data function above. 
    """
    KFs = {'l':None, 'r':None}
    motion_covar, _, _, _ = initialize_covariances(freq)

    n_segs = len(kf_data['r'])
    for lr in 'lr':
        KF_lr = []
        for i in xrange(n_segs):
            hy_strm   = kf_data[lr][i]['hy_strm']
            cam_dat   = kf_data[lr][i]['cam_dat']
            cam_strms= []
            for cinfo in cam_data.values():
                cam_strms.append(cinfo['stream'])

            x0, S0 = get_first_state(cam_strms + [hy_strm], freq=1./hy_strm.dt)

            KF = kalman()
            kf_tstart = hy_strm.get_start_time()
            KF.init_filter(kf_tstart, x0, S0, motion_covar)
            KF_lr.append(KF)
        KFs[lr] = KF_lr
    return KFs


def run_KF(KF, nsteps, hy_strm, cam_dat, hy_covar, rgb_covar, rgbd_covar, do_smooth=True, plot=False):
    """
    Runs the Kalman filter/smoother for NSTEPS using
    {hydra/rgb/rgbd}_covar as the measurement covariances.
    
    HY_STRM   : TF stream from hydra
    CAM_DAT   : Dicionary of camera streams 
    """ 
    ## place holders for kalman filter's output:
    xs_kf, covars_kf, ts_kf = [KF.x_filt],[KF.S_filt],[KF.t_filt]

    hy_snext  = stream_soft_next(hy_strm)

    cam_strms = [cinfo['stream'] for cinfo in cam_dat.values()]
    cam_types = [cinfo['type'] for cinfo in cam_dat.values()]
    cam_snext = [stream_soft_next(cstrm) for cstrm in cam_strms]
    
    for i in xrange(nsteps):
        KF.register_tf_observation(hy_snext(), KF.hy_covar, do_control_update=True)

        for i in xrange(len(cam_types)):
            cam_covar = rgbd_covar
            if cam_types[i]=='rgb':
                cam_covar = rgb_covar
            KF.register_tf_observation(cam_snext[i](), cam_covar, do_control_update=False)

        xs_kf.append(KF.x_filt)
        covars_kf.append(KF.S_filt)
        ts_kf.append(KF.t_filt)

    if do_smooth:
        A,R  = KF.get_motion_mats()
        xs_smthr, covars_smthr = smoother(A, R, xs_ks, S_kf)
        
        if plot:
            kf_strm   = streamize(state_to_hmat(xs_kf), np.array(ts_kf), 1./hy_strm.dt, hy_strm.favg)
            sm_strm   = streamize(state_to_hmat(xs_smthr), np.array(ts_kf), 1./hy_strm.dt, hy_strm.favg)
            plot_tf_strms([kf_strm, sm_strm, hy_strm]+cam_strms, ['kf', 'smoother', 'hydra']+cam_dat.keys())
        
        return (ts_kf, xs_kf, covars_kf, xs_smthr, covars_smthr)
    else:
        if plot:
            kf_strm   = streamize(state_to_hmat(xs_kf), np.array(ts_kf), 1./hy_strm.dt, hy_strm.favg)
            plot_tf_strms([kf_strm, hy_strm]+cam_strms, ['kf', 'hydra']+cam_dat.keys())
        return (ts_kf, xs_kf, covars_kf, None, None)



def filter_traj(demo_fname, tps_model_fname=None, save_tps=False, do_smooth=True, plot=False):
    """
    Runs the kalman filter for BOTH the grippers and writes the demo.traj file.
    TPS_MODEL_FNAME : The name of the file to load the tps-model from
    """
    rec_data, time_shifts, _, n_segs = prepare_kf_data(demo_fname, freq=30.0,
                                               rem_outliers=True,
                                               tps_correct=True, tps_model_fname=tps_model_fname,
                                               plot=plot)

    _, rgb_covar, rgbd_cavar, hydra_covar =  initialize_covariances()

    KFs  = initialize_KFs(rec_data)
    traj = {'l' : None, 'r':None}

    for lr in 'lr':
        lr_trajs = []
        for iseg in xrange(n_segs):
            KF = KFs[lr][iseg]
            hy_strm  = rec_data[lr][iseg]['hy_strm']
            pot_strm = rec_data[lr][iseg]['pot_strm']
            cam_dat  = rec_data[lr][iseg]['cam_dat']
            nsteps   = rec_data[lr][iseg]['nsteps']

            ts, xs_kf, covars_kf, xs_smthr, covars_smthr = run_KF(KF, nsteps,
                                                                  hy_strm, cam_dat,
                                                                  hy_covar, rgb_covar, rgbd_covar,
                                                                  do_smooth, plot)
            Ts_kf, Ts_smthr = state_to_hmat(xs_kf), state_to_hmat(xs_smthr)
            
            pot_angles = [x for x in pot_strm]

            traj = {"stamps"  : ts,
                    "tfms"    : Ts_kf,
                    "covars"  : covars_kf,
                    "tfms_s"  : Ts_smthr,
                    "covars_s": covars_smthr,
                    "pot_angles": pot_angles}
            lr_trajs.append(traj)

        traj[lr] = lr_trajs

    traj_fname = osp.join(osp.dirname(demo_fname), 'demo.traj')
    with open(traj_fname, 'wb') as f:
        cp.dump(traj, f)


if __name__=='__main__':
    demo_fname = '../hd_data/demos/overhand/demo00001/demo.data'
    filter_traj(demo_fname, tps_model_fname=None, save_tps=True, do_smooth=True, plot=True)

