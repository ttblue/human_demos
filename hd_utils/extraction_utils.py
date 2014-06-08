import os.path as osp
import numpy as np
import cv2

from defaults import demo_names
from cloud_proc_funcs import extract_rope_tracking, extract_red, remove_outlier_connected_component, remove_shadow, rope_shape_filter, extract_cloud
import bulletsimpy
import cbulletracpy2
from hd_utils import clouds, color_match
from rapprentice import rope_initialization

import openravepy
from sklearn.neighbors import KDTree

from clouds import XYZ_to_xy, depth_to_xyz
from hd_utils.defaults import asus_xtion_pro_f



def searchsortednearest(a,v):
    higher_inds = np.fmin(np.searchsorted(a,v), len(a)-1)
    lower_inds = np.fmax(higher_inds-1, 0)
    closer_inds = higher_inds
    lower_is_better = np.abs(a[higher_inds] - v) > np.abs(a[lower_inds] - v)
    closer_inds[lower_is_better] = lower_inds[lower_is_better]
    return closer_inds


def get_video_frames(video_dir, frame_stamps):
    
    video_stamps = np.loadtxt(osp.join(video_dir,demo_names.stamps_name))
    frame_inds = np.searchsorted(video_stamps, frame_stamps)
    
    if frame_inds[-1] >= len(video_stamps):
        frame_inds[-1] = len(video_stamps) - 1
    
    rgbs = []
    depths = []
    for frame_ind in frame_inds:
        rgb = cv2.imread(osp.join(video_dir,demo_names.rgb_name%frame_ind))
        assert rgb is not None
        rgbs.append(rgb)
        depth = cv2.imread(osp.join(video_dir,demo_names.depth_name%frame_ind), cv2.CV_16UC1)
        assert depth is not None
        depths.append(depth)
    return rgbs, depths


def make_table_xml(translation, extents):
    xml = """
<Environment>
  <KinBody name="table">
    <Body type="static" name="table_link">
      <Geom type="box">
        <Translation>%f %f %f</Translation>
        <extents>%f %f %f</extents>
        <diffuseColor>.96 .87 .70</diffuseColor>
      </Geom>
    </Body>
  </KinBody>
</Environment>
""" % (translation[0], translation[1], translation[2], extents[0], extents[1], extents[2])
    return xml


from mpl_toolkits.mplot3d import axes3d
import pylab
fig = pylab.figure()
from mayavi import mlab


def track_video_frames_online(video_dir, T_w_k, init_tfm, table_plane, ref_image = None, rope_params = None):    
    video_stamps = np.loadtxt(osp.join(video_dir,demo_names.stamps_name))
    
    env = openravepy.Environment()
    
    stdev = np.array([])
    
    
    
    rgb = cv2.imread(osp.join(video_dir,demo_names.rgb_name%0))
    rgb = color_match.match(ref_image, rgb)
    assert rgb is not None
    depth = cv2.imread(osp.join(video_dir,demo_names.depth_name%0), 2)
    assert depth is not None
    rope_xyz = extract_red(rgb, depth, T_w_k)

    table_dir = np.array([table_plane[0], table_plane[1], table_plane[2]])
    table_dir = table_dir / np.linalg.norm(table_dir)
    
    table_dir = init_tfm[:3,:3].dot(table_dir)    
    if np.dot([0,0,1], table_dir) < 0:
        table_dir = -table_dir
    table_axis = np.cross(table_dir, [0,0,1])
    table_angle = np.arccos(np.dot([0,0,1], table_dir))
    table_tfm = openravepy.matrixFromAxisAngle(table_axis, table_angle)
    table_center_xyz = np.mean(rope_xyz, axis=0)
        
    table_tfm[:3,3] = - table_tfm[:3,:3].dot(table_center_xyz) + table_center_xyz
    init_tfm = table_tfm.dot(init_tfm)
    
    tracked_nodes = None
    
    for (index, stamp) in zip(range(len(video_stamps)), video_stamps):
        print index, stamp
        rgb = cv2.imread(osp.join(video_dir,demo_names.rgb_name%index))
        rgb = color_match.match(ref_image, rgb)
        assert rgb is not None
        depth = cv2.imread(osp.join(video_dir,demo_names.depth_name%index), 2)
        assert depth is not None
        color_cloud = extract_rope_tracking(rgb, depth, T_w_k) # color_cloud is at first in camera frame
        
        
        #print "cloud in camera frame"
        #print "==================================="
        #print color_cloud[:, :3]

        
    
        color_cloud = clouds.downsample_colored(color_cloud, .01)
        color_cloud[:,:3] = color_cloud[:,:3].dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:] # color_cloud now is in global frame
        
        raw_color_cloud = np.array(color_cloud)
        ##########################################
        ### remove the shadow points on the desk
        ##########################################
        color_cloud = remove_shadow(color_cloud)
        
        if tracked_nodes is not None:
            color_cloud = rope_shape_filter(tracked_nodes, color_cloud)
             
        if index == 0:
            rope_xyz = extract_red(rgb, depth, T_w_k)
            rope_xyz = clouds.downsample(rope_xyz, .01)
            rope_xyz = rope_xyz.dot(init_tfm[:3, :3].T) + init_tfm[:3,3][None,:]
            rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz) # rope_nodes and rope_xyz are in global frame
                        
            # print rope_nodes
            
            table_height = rope_xyz[:,2].min() - .02
            table_xml = make_table_xml(translation=[1, 0, table_height], extents=[.85, .55, .01])
            env.LoadData(table_xml)            
            
            bulletsimpy.sim_params.scale = 10
            bulletsimpy.sim_params.maxSubSteps = 200
            if rope_params is None:
                rope_params = bulletsimpy.CapsuleRopeParams()
                rope_params.radius = 0.005
                #angStiffness: a rope with a higher angular stiffness seems to have more resistance to bending.
                #orig self.rope_params.angStiffness = .1
                rope_params.angStiffness = .1
                #A higher angular damping causes the ropes joints to change angle slower.
                #This can cause the rope to be dragged at an angle by the arm in the air, instead of falling straight.
                #orig self.rope_params.angDamping = 1
                rope_params.angDamping = 1
                #orig self.rope_params.linDamping = .75
                #Not sure what linear damping is, but it seems to limit the linear accelertion of centers of masses.
                rope_params.linDamping = .75
                #Angular limit seems to be the minimum angle at which the rope joints can bend.
                #A higher angular limit increases the minimum radius of curvature of the rope.
                rope_params.angLimit = .4
                #TODO--Find out what the linStopErp is
                #This could be the tolerance for error when the joint is at or near the joint limit
                rope_params.linStopErp = .2  
            
            bt_env = bulletsimpy.BulletEnvironment(env, [])
            bt_env.SetGravity([0,0,-0.1])
            rope = bulletsimpy.CapsuleRope(bt_env, 'rope', rope_nodes, rope_params)
            
            continue
        
        
        
 #==============================================================================
 #       rope_nodes = rope.GetNodes()
 #       #print "rope nodes in camera frame"
 #       R = init_tfm[:3,:3].T
 #       t = - R.dot(init_tfm[:3,3])
 #       rope_in_camera_frame = rope_nodes.dot(R.T) + t[None,:]
 #       #print rope_in_camera_frame
 #       uvs = XYZ_to_xy(rope_in_camera_frame[:,0], rope_in_camera_frame[:,1], rope_in_camera_frame[:,2], asus_xtion_pro_f)
 #       uvs = np.vstack(uvs)
 #       #print uvs
 #       #print "uvs"
 #       uvs = uvs.astype(int)
 #       
 #       n_rope_nodes = len(rope_nodes)
 #       
 #       DEPTH_OCCLUSION_DIST = .03
 #       occ_dist = DEPTH_OCCLUSION_DIST
 #       
 #       vis = np.ones(n_rope_nodes)
 #       
 #       rope_depth_in_camera = np.array(rope_in_camera_frame)
 #       depth_xyz = depth_to_xyz(depth, asus_xtion_pro_f)
 # 
 #       for i in range(n_rope_nodes):
 #           u = uvs[0, i]
 #           v = uvs[1, i]
 #           
 #           neighbor_radius = 10;
 #           v_range = [max(0, v-neighbor_radius), v+neighbor_radius+1]
 #           u_range = [max(0, u-neighbor_radius), u+neighbor_radius+1]
 #           
 #           xyzs = depth_xyz[v_range[0]:v_range[1], u_range[0]:u_range[1]]
 #                       
 #           xyzs = np.reshape(xyzs, (xyzs.shape[0]*xyzs.shape[1], xyzs.shape[2]))
 #           dists_to_origin = np.linalg.norm(xyzs, axis=1)
 #           
 #           dists_to_origin = dists_to_origin[np.isfinite(dists_to_origin)]
 #           
 #           #print dists_to_origin
 #           min_dist_to_origin = np.min(dists_to_origin)
 #                       
 #           print v, u, min_dist_to_origin, np.linalg.norm(rope_in_camera_frame[i])
 #                                   
 #           if min_dist_to_origin + occ_dist > np.linalg.norm(rope_in_camera_frame[i]):
 #               vis[i] = 1
 #           else:
 #               vis[i] = 0
 #               
 #       #print "vis result"
 #       #print vis
 #       
 #       rope_depth_in_global = rope_depth_in_camera.dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]
 #==============================================================================
        
        
        depth_cloud = extract_cloud(depth, T_w_k)
        depth_cloud[:,:3] = depth_cloud[:,:3].dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:] # depth_cloud now is in global frame


        [tracked_nodes, new_stdev] = bulletsimpy.py_tracking(rope, bt_env, init_tfm, color_cloud, rgb, depth, 5, stdev)
        stdev = new_stdev
        
        #print tracked_nodes

        #if index % 10 != 0:
        #    continue
        
        print index
        
        
        xx, yy = np.mgrid[-1:3, -1:3]
        zz = np.ones(xx.shape) * table_height
        table_cloud = [xx, yy, zz]
        
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.set_autoscale_on(False)     
        
        print init_tfm[:,3]
        ax.plot(depth_cloud[:,0], depth_cloud[:,1], depth_cloud[:,2], 'go', alpha=0.1)
        ax.plot(color_cloud[:,0], color_cloud[:,1], color_cloud[:,2]-0.1, 'go')
        ax.plot(raw_color_cloud[:,0], raw_color_cloud[:,1], raw_color_cloud[:,2] -0.2, 'ro')
        #ax.plot(rope_depth_in_global[:,0], rope_depth_in_global[:,1], rope_depth_in_global[:,2], 'ro')
        ax.plot_surface(table_cloud[0], table_cloud[1], table_cloud[2], color = (0,1,0,0.5))
        ax.plot(tracked_nodes[:,0], tracked_nodes[:,1], tracked_nodes[:,2], 'bo')
        ax.plot([init_tfm[0,3]], [init_tfm[1,3]], [init_tfm[2,3]], 'ro')   


        fig.show()
        raw_input()

        
        
def track_video_frames_offline(video_dir, T_w_k, init_tfm, ref_image, rope_params = None):
    
    video_stamps = np.loadtxt(osp.join(video_dir,demo_names.stamps_name))    
    
    rgbs = []
    depths = []
    color_clouds = []
    
    for (index, stamp) in zip(range(len(video_stamps)), video_stamps):        
        print index, stamp
        rgb = cv2.imread(osp.join(video_dir,demo_names.rgb_name%index))
        rgb = color_match.match(ref_image, rgb)
        assert rgb is not None
        depth = cv2.imread(osp.join(video_dir,demo_names.depth_name%index), 2)
        assert depth is not None
        color_cloud = extract_rope_tracking(rgb, depth, T_w_k)
        color_cloud = clouds.downsample_colored(color_cloud, .01)
        color_cloud[:,:3] = color_cloud[:,:3].dot(init_tfm[:3,:3].T) + init_tfm[:3,3][None,:]

        if index == 0:
            rope_xyz = extract_red(rgb, depth, T_w_k)
            rope_xyz = clouds.downsample(rope_xyz, .01)
            rope_xyz = rope_xyz.dot(init_tfm[:3, :3].T) + init_tfm[:3,3][None,:]
            rope_nodes = rope_initialization.find_path_through_point_cloud(rope_xyz)
            rope_radius = 0.005
                
        rgbs.append(rgb)
        depths.append(depth)
        color_clouds.append(color_cloud)
        
    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)    
    
    all_tracked_nodes = cbulletracpy2.py_tracking(rope_nodes, rope_radius, T_w_k, color_clouds, rgbs, depths, 5)
    
    for i in range(len(rgbs)):
        if i % 30 != 0:
            continue
        print i
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.set_autoscale_on(False)
        tracked_nodes = all_tracked_nodes[i, :, :]
        ax.plot(tracked_nodes[:, 0], tracked_nodes[:,1], tracked_nodes[:,2], 'o')
        fig.show()
        raw_input()



def get_rgbd_names_times (video_dir, depth=True):
    from glob import glob
    
    video_stamps = np.loadtxt(osp.join(video_dir,demo_names.stamps_name))
    rgbnames = glob(osp.join(video_dir, demo_names.rgb_regexp))
    ind2rgbfname = dict([(int(osp.splitext(osp.basename(fname))[0][3:]), fname) for fname in rgbnames])
    
    if depth:
        depthnames = glob(osp.join(video_dir, demo_names.depth_regexp))
 
        ind2depthfname = dict([(int(osp.splitext(osp.basename(fname))[0][5:]), fname) for fname in depthnames])
        return ind2rgbfname, ind2depthfname, video_stamps
    else: return ind2rgbfname, video_stamps
    
    
def get_videos(video_dir):
        
    from glob import glob
    rgbnames = glob(osp.join(video_dir, demo_names.rgb_regexp))
    depthnames = glob(osp.join(video_dir, demo_names.depth_regexp))
    
    ind2rgbfname = dict([(int(osp.splitext(osp.basename(fname))[0][3:]), fname) for fname in rgbnames])
    ind2depthfname = dict([(int(osp.splitext(osp.basename(fname))[0][5:]), fname) for fname in depthnames])

                
    rgbs = []
    depths = []
    
    nframes = len(rgbnames)
    
    for i in xrange(nframes): 
        rgb = cv2.imread(ind2rgbfname[i])
        assert rgb is not None
        rgbs.append(rgb)
        depth = cv2.imread(ind2depthfname[i],2)
        assert depth is not None
        depths.append(depth)
    return rgbs, depths