import os, os.path as osp
import yaml
import cPickle as cp
import numpy as np
import time

import rospy
from sensor_msgs.msg import PointCloud2

from hd_utils.defaults import tfm_link_rof, demo_names, demo_files_dir
from hd_utils.utils import avg_transform
from hd_utils.colorize import yellowprint

from hd_track.streamer import streamize
from hd_track.stream_pc import streamize_rgbd_pc    

def load_data(data_file, lr, freq=30.0, speed=1.0):
    """
    Changed slightly from the one in demo_data_prep to include speed.
    """
    with open(data_file, 'r') as f:
        dat = cp.load(f)

    demo_dir    = osp.dirname(data_file)
    cam_types   = get_cam_types(demo_dir)
    T_cam2hbase = dat['T_cam2hbase']

    cam_info = {}
    for kname in dat[lr].keys():
        if 'cam' in kname:
            tfs = [tt[0] for tt in dat[lr][kname]]
            ts  = [tt[1] for tt in dat[lr][kname]]
            ## don't append any empty-streams:
            if len(ts) > 0:
                cam_strm = streamize(tfs, ts, freq, avg_transform, speed=speed)#, tstart=-1./freq)
                cam_info[kname] = {'type'   : cam_types[kname],
                                   'stream' : cam_strm}

    ## hydra data:
    hydra_tfs = [tt[0] for tt in dat[lr]['hydra']]     
    hydra_ts  = np.array([tt[1] for tt in dat[lr]['hydra']])

    if len(hydra_ts) <= 0:
        redprint("ERROR : No hydra data found in : %s"%dat_fname)
        sys.exit(-1)
    hydra_strm = streamize(hydra_tfs, hydra_ts, freq, avg_transform, speed=speed)#, tstart=-1./freq)

    ## potentiometer angles:
    pot_vals = np.array([tt[0] for tt in dat[lr]['pot_angles']])
    pot_ts   = np.array([tt[1] for tt in dat[lr]['pot_angles']])
    if len(pot_ts) <= 0:
        redprint("ERROR : No potentiometer data found in : %s"%dat_fname)
        sys.exit(-1)
    pot_strm = streamize(pot_vals, pot_ts, freq, np.mean, speed=speed)#, tstart=-1./freq)


    return (T_cam2hbase, cam_info, hydra_strm, pot_strm)


def get_cam_transforms (calib_file, num_cams):
    with open(calib_file, 'r') as fh: dat = cp.load(fh)
    cam_tfm = {cam:None for cam in range(2,num_cams+1)}

    for cam in cam_tfm:
        for tfm in dat['transforms']:
            parent = tfm['parent'] if tfm['parent'][0] != '/' else tfm['parent'][1:]
            child = tfm['child'] if tfm['child'][0] != '/' else tfm['child'][1:]
            if parent == "camera1_link" and child == "camera%i_link"%cam:
                cam_tfm[cam] = np.linalg.inv(tfm_link_rof).dot(tfm['tfm'].dot(tfm_link_rof))

    return cam_tfm


def relative_time_streams(strms, freq, speed=1.0):
    """
    Changed slightly from the one in demo_data_prep to include speed.
    """
    for strm in strms:
        assert strm.get_speed() == speed

    n_strms = len(strms)
    dt =speed/freq

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

def view_demo_on_rviz(demo_type, demo_name, freq, speed=1.0, main='h', prompt=False):
    """
    Visualizes recorded demo on rviz (without kalman filter/smoother data).
    @demo_type, @demo_name: demo identification.
    @freq: basically measure of fine-ness of timesteps.
    @speed: how fast to replay demo.
    @main: which sensor to display the marker for
    @prompt: does the user hit enter after each time step?
    """
    demo_dir = osp.join(demo_files_dir, demo_type, demo_name)
    bag_file = osp.join(demo_dir, demo_names.bag_name)
    data_file = osp.join(demo_dir, demo_names.data_name)
    with open(osp.join(demo_dir, demo_names.camera_types_name),'r') as fh: cam_types = yaml.load(fh)
    with open(data_file, 'r') as fh: dat = cp.load(fh)
    
    # get grippers used
    grippers = dat.keys()

    # data 
    rgbd_dirs = {cam:osp.join(demo_dir,demo_names.video_dir%cam) for cam in cam_types if cam_types[cam] == 'rgbd'}
    cam_frames = {cam:'/camera%i_rgb_optical_frame'%cam for cam in rgbd_dirs}
    
    tfm_pubs = {}

    cam_dat = {}
    hydra_dat = {}
    pot_dat = {}
    
    T_cam2hbase, cam_dat['l'], hydra_dat['l'], pot_dat['l'] = load_data(data_file, 'l', freq, speed)
    _,  cam_dat['r'], hydra_dat['r'], pot_dat['r'] = load_data(data_file, 'r', freq, speed)

    for lr in 'lr':
        for cam in cam_dat[lr].keys():
            all_cam_strms.append(cam_dat[lr][cam]['stream'])
    tmin, _, nsteps = relative_time_streams(hydra_dat.values() + pot_dat.values() + all_cam_strms, freq, speed)

    if rospy.get_name() == "/unnamed":
        rospy.init_node("visualize_demo")


    ## publishers for unfiltered-data:
    for lr in grippers:
        tfm_pubs[lr] = {}
        for cam in cam_types:
            tfm_pubs[lr][cam] = rospy.Publisher('/%s_ar%i_estimate'%(lr,cam), PoseStamped)
        tfm_pub[lr]['h'] = rospy.Publisher('/%s_hydra_estimate'%(lr), PoseStamped)

    ## get the point-cloud stream
    pc_strms = {cam:streamize_rgbd_pc(rgbd_dir[cam], cam_frames[cam], freq, tstart=tmin,speed=speed) for cam in rgbd_dirs}
    pc_pubs = {cam:rospy.Publisher('/point_cloud%i'%cam, PointCloud2) for cam in rgbd_dirs}

    cam_tfms  = get_cam_transforms (calib_file, len(cam_types))
    for cam in rgbd_dirs:
        if cam != 1:
            publish_static_tfm(cam_frames[1], cam_frames[cam], cam_tfms[cam])

    sleeper = rospy.Rate(freq)
    T_far = np.eye(4)
    T_far[0:3,3] = [10,10,10]        
    
    handles = []
    
    prev_ang = {'l': 0, 'r': 0}
    for i in xrange(nsteps):
        if prompt:
            raw_input("Hit enter when ready.")
        print "Time stamp: ", tmin+(0.0+i*speed)/freq
        
        ## show the point-cloud:
        found_pc = False
        for cam in pc_strms:
            try:
                pc = pc_strms[cam].next()
                if pc is not None:
                    print "pc%i ts:"%cam, pc.header.stamp.to_sec()
                    pc.header.stamp = rospy.Time.now()
                    pc_pubs[cam].publish(pc)
                    found_pc = True
                else:
                    print "pc%i ts:"%cam,None
            except StopIteration:
                pass

        next_est = {lr:{} for lr in grippers}
        tfms = []
        ang_vals  = []

        for lr in grippers:
            next_est[lr]['h'] = soft_next(hydra_dat[lr])
            for cam in cam_types:
                next_est[lr][cam] = soft_next(cam_dat[lr][cam]['stream'])

            ang_val = soft_next(ang_strm[lr])
            if ang_val != None:
                prev_ang[lr] = ang_val
                ang_val  = ang_val
            else:
                ang_val = prev_ang[lr]
            
            tfm = next_est[lr][main]
            if tfm is None:
                tfms.append(T_far)
            else:
                tfms.append(tfm)
            ang_vals.append(ang_val)

        handles = draw_trajectory(cam_frames[cam], tfms, color=(1,1,0,1), open_fracs=ang_vals)

        for lr in grippers:
            for m,est in next_est[lr].items():
                if est != None:
                    tfm_pubs[m].publish(pose_to_stamped_pose(hmat_to_pose(est), cam_frame[1]))
                else:
                    tfm_pubs[m].publish(pose_to_stamped_pose(hmat_to_pose(T_far), cam_frame[1]))
        
        sleeper.sleep()




def view_tracking_on_rviz(demo_type, demo_name, freq, speed=1.0, use_smoother=True, prompt=False):
    """
    Visualizes demo after kalman tracking/smoothing on rviz.
    @demo_type, @demo_name: demo identification.
    @freq: basically measure of fine-ness of timesteps.
    @speed: how fast to replay demo.
    @main: which sensor to display the marker for
    @prompt: does the user hit enter after each time step?
    """
    demo_dir = osp.join(demo_files_dir, demo_type, demo_name)
    bag_file = osp.join(demo_dir, demo_names.bag_name)
    traj_file = osp.join(demo_dir, demo_names.traj_name)
    with open(osp.join(demo_dir, demo_names.camera_types_name),'r') as fh: cam_types = yaml.load(fh)
    with open(traj_file, 'r') as fh: traj = cp.load(fh)
    
    # get grippers used
    grippers = traj.keys()

    if rospy.get_name() == "/unnamed":
        rospy.init_node("visualize_demo")

    # data 
    rgbd_dirs = {cam:osp.join(demo_dir,demo_names.video_dir%cam) for cam in cam_types if cam_types[cam] == 'rgbd'}
    pc_pubs = {cam:rospy.Publisher('/point_cloud%i'%cam, PointCloud2) for cam in rgbd_dirs}
    cam_frames = {cam:'/camera%i_rgb_optical_frame'%cam for cam in rgbd_dirs}
        
    cam_tfms  = get_cam_transforms (calib_file, len(cam_types))
    for cam in rgbd_dirs:
        if cam != 1:
            publish_static_tfm(cam_frames[1], cam_frames[cam], cam_tfms[cam])

    # Remove segment "done", it is just a single frame
    segs = sorted(traj[grippers[0]].keys())
    segs.remove('done')
    
    sleeper = rospy.Rate(freq)
    T_far = np.eye(4)
    T_far[0:3,3] = [10,10,10]
    
    for seg in segs:
        if prompt:
            raw_input("Press enter for segment %s."%seg)
        else:
            yellow_print("Segment %s beginning."%seg)
            time.sleep(1)
        
        # Initializing data streams:
        traj_strms = {}
        pot_strms = {}
        tfms_key = 'tfms_s' if use_smooth else 'tfms'

        for lr in grippers:
            traj_strms[lr] = streamize(traj[lr][seg][tfms_key], traj[lr][seg]['stamps'], freq, avg_transform, speed=speed)
            pot_strms[lr] = streamize(traj[lr][seg]['pot_angles'], traj[lr][seg]['stamps'], freq, np.mean, speed=speed)
        
        tmin, tmax, nsteps = relative_time_streams(traj_strms.values() + pot_strms.values(), freq, speed)
        pc_strms = {cam:streamize_rgbd_pc(rgbd_dir[cam], cam_frames[cam], freq, tstart=tmin, tend=tmax,speed=speed) for cam in rgbd_dirs}
    
        prev_ang = {'l': 0, 'r': 0}
        for i in xrange(nsteps):
            if prompt:
                raw_input("Hit enter when ready.")
            print "Time stamp: ", tmin+(0.0+i*speed)/freq
            
            ## show the point-cloud:
            found_pc = False
            for cam in pc_strms:
                try:
                    pc = pc_strms[cam].next()
                    if pc is not None:
                        print "pc%i ts:"%cam, pc.header.stamp.to_sec()
                        pc.header.stamp = rospy.Time.now()
                        pc_pubs[cam].publish(pc)
                        found_pc = True
                    else:
                        print "pc%i ts:"%cam,None
                except StopIteration:
                    pass
    
            tfms = []
            ang_vals  = []
    
            for lr in grippers:
                tfm = soft_next(traj_strms[lr])
                
                ang_val = soft_next(ang_strm[lr])
                if ang_val != None:
                    prev_ang[lr] = ang_val
                    ang_val  = ang_val
                else:
                    ang_val = prev_ang[lr]
                
                if tfm is None:
                    tfms.append(T_far)
                else:
                    tfms.append(tfm)
                ang_vals.append(ang_val)
    
            handles = draw_trajectory(cam_frames[cam], tfms, color=(1,1,0,1), open_fracs=ang_vals)
            
            sleeper.sleep()