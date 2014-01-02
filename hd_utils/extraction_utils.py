import os.path as osp
import numpy as np
import cv2

def searchsortednearest(a,v):
    higher_inds = np.fmin(np.searchsorted(a,v), len(a)-1)
    lower_inds = np.fmax(higher_inds-1, 0)
    closer_inds = higher_inds
    lower_is_better = np.abs(a[higher_inds] - v) > np.abs(a[lower_inds] - v)
    closer_inds[lower_is_better] = lower_inds[lower_is_better]
    return closer_inds

def get_video_frames(video_dir, frame_stamps):
    video_stamps = np.loadtxt(osp.join(video_dir,"stamps.txt"))

    frame_inds = searchsortednearest(video_stamps, frame_stamps)
    
    from glob import glob
    rgbnames = glob(osp.join(video_dir, "rgb*.jpg"))
    depthnames = glob(osp.join(video_dir, "depth*.png"))
        
    ind2rgbfname = dict([(int(osp.splitext(osp.basename(fname))[0][3:]), fname) for fname in rgbnames])
    ind2depthfname = dict([(int(osp.splitext(osp.basename(fname))[0][5:]), fname) for fname in depthnames])
    
    #print ind2depthfname
    
    rgbs = []
    depths = []
    for frame_ind in frame_inds:
        rgb = cv2.imread(ind2rgbfname[frame_ind])
        assert rgb is not None
        rgbs.append(rgb)
        depth = cv2.imread(ind2depthfname[frame_ind],2)
        assert depth is not None
        depths.append(depth)
    return rgbs, depths

def get_rgbd_names_times (video_dir, depth=True):
    from glob import glob
    
    video_stamps = np.loadtxt(osp.join(video_dir,"stamps.txt"))
    rgbnames = glob(osp.join(video_dir, "rgb*.jpg"))
    #inds_rgb = [int(osp.splitext(osp.basename(fname))[0][3:]) for fname in rgbnames]
    min_ind_rgb = 0#min(inds_rgb)
    ind2rgbfname = dict([(int(osp.splitext(osp.basename(fname))[0][3:])-min_ind_rgb, fname) for fname in rgbnames])
    
    if depth:
        depthnames = glob(osp.join(video_dir, "depth*.png"))
        #inds_d = [int(osp.splitext(osp.basename(fname))[0][3:]) for fname in depthnames]
        min_ind_d = 0#min(inds_d)
        ind2depthfname = dict([(int(osp.splitext(osp.basename(fname))[0][5:])-min_ind_d, fname) for fname in depthnames])
        return ind2rgbfname, ind2depthfname, video_stamps
    else: return ind2rgbfname, video_stamps