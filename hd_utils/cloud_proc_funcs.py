import cv2, numpy as np
import os.path as osp
import skimage.morphology as skim
DEBUG_PLOTS=False

from hd_utils import clouds
from hd_utils.defaults import asus_xtion_pro_f, hd_data_dir
from hd_utils.pr2_utils import get_kinect_transform
from hd_rapprentice.rope_initialization import points_to_graph
import networkx as nx
from openravepy import matrixFromAxisAngle

def remove_outlier_connected_component(xyz, max_dist = .03):
    G = points_to_graph(xyz, max_dist)

    components = nx.connected_components(G)
    
    xyz = []
    max_component_len = len(components[0])
    for component in components:
        if len(component) > 0.5 * max_component_len:
            xyz = xyz + [G.node[i]["xyz"] for i in component]
        else: 
            break
    return xyz
    
def extract_color(rgb, depth, mask, T_w_k, xyz_mask=None, use_outlier_removal=False, outlier_thresh=2, outlier_k=20):

    """
    extract red points and downsample
    """
        
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    
    h_mask = mask[0](h)
    s_mask = mask[1](s)
    v_mask = mask[2](v)
    color_mask = h_mask & s_mask & v_mask
    
    valid_mask = (depth > 0)
    
    xyz_k = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    z = xyz_w[:,:,2]   
    z0 = xyz_k[:,:,2]

    # 'height' or distance from the camera
    height_mask = (xyz_w[:,:,2] > 0.6) & (xyz_w[:,:,2] < 1.1)
    if xyz_mask: height_mask = height_mask & xyz_mask(xyz_w)
        
    good_mask = color_mask & valid_mask & height_mask
    #good_mask = skim.remove_small_objects(good_mask,min_size=64)

    if DEBUG_PLOTS:
        cv2.imshow("z0",z0/z0.max())
        cv2.imshow("z",z/z.max())
        cv2.imshow("hue", h_mask.astype('uint8')*255)
        cv2.imshow("sat", s_mask.astype('uint8')*255)
        cv2.imshow("val", v_mask.astype('uint8')*255)
        cv2.imshow("final",good_mask.astype('uint8')*255)
        cv2.imshow("rgb", rgb)
        cv2.waitKey()

    good_xyz = xyz_w[good_mask]
    
    if use_outlier_removal:
        #good_xyz = clouds.remove_outliers(good_xyz, outlier_thresh, outlier_k)
        good_xyz = remove_outlier_connected_component(good_xyz)
        #good_xyz = clouds.cluster_filter(good_xyz, 0.03, int(0.4 * len(good_xyz)))

    return good_xyz


def extract_red(rgb, depth, T_w_k):
    red_mask = [lambda(x): (x<15)|(x>125), lambda(x): x>80, lambda(x): x>100]
    xyz_mask = (lambda(xyz): xyz[:, :, 2] > 0.95)
    return extract_color(rgb, depth, red_mask, T_w_k, xyz_mask, True)

def extract_white(rgb, depth, T_w_k):
    white_mask = [lambda(x): (x>0), lambda(x): x<30, lambda(x): (x>100)]
    return extract_color(rgb, depth, white_mask, T_w_k)

def extract_yellow(rgb, depth, T_w_k):
    yellow_mask = [lambda(x): (x>23)&(x<40), lambda(x): (x>200), lambda(x): x<100]
    #xyz_mask = (lambda(xyz): xyz[:, :, 2] < 0.9)
    xyz_mask = None
    return extract_color(rgb, depth, yellow_mask, T_w_k, xyz_mask)
    

def extract_hitch(rgb, depth, T_w_k, dir=None, radius=0.016, length =0.215, height_range=[0.70,0.80]):
    """
    template match to find the hitch in the picture, 
    get the xyz at those points using the depth image,
    extend the points down to aid in tps.
    """
    template_fname = osp.join(hd_data_dir, 'hitch_template.jpg')
    template = cv2.imread(template_fname)
    h,w,_    = template.shape
    res = cv2.matchTemplate(rgb, template, cv2.TM_CCORR_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)

    top_left     = max_loc
    bottom_right = (top_left[0]+w, top_left[1]+h)

    if DEBUG_PLOTS:
        cv2.rectangle(rgb,top_left, bottom_right, (255,0,0), thickness=2)
        cv2.imshow("hitch detect", rgb)
        cv2.waitKey()

    # now get all points in a window around the hitch loc:
    xyz_k = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]
    
    hitch_pts = xyz_w[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0],:]
    height_mask = (hitch_pts[:,:,2] > height_range[0]) & (hitch_pts[:,:,2] < height_range[1])
    hitch_pts = hitch_pts[height_mask]

    ## add pts along the rod:
    center_xyz = np.median(hitch_pts, axis=0)
    ang = np.linspace(0, 2*np.pi, 30)
    circ_pts = radius*np.c_[np.cos(ang), np.sin(ang)] + center_xyz[None,:2]
    circ_zs  = np.linspace(center_xyz[2], length+center_xyz[2], 30)

    rod_pts  = np.empty((0,3))
    for z in circ_zs:
        rod_pts = np.r_[rod_pts, np.c_[circ_pts, z*np.ones((len(circ_pts),1))] ]
    
    all_pts = np.r_[hitch_pts, rod_pts]
    if dir is None:
        return all_pts, center_xyz
    else:
        axis = np.cross([0,0,1], dir)
        angle = np.arccos(np.dot([0,0,1], dir))
        tfm = matrixFromAxisAngle(axis, angle)
        tfm[:3,3] = - tfm[:3,:3].dot(center_xyz) + center_xyz
        return np.asarray([tfm[:3,:3].dot(point) + tfm[:3,3] for point in all_pts]), center_xyz
        
        
            
    
    

def grabcut(rgb, depth, T_w_k):
    xyz_k = clouds.depth_to_xyz(depth, asus_xtion_pro_f)
    xyz_w = xyz_k.dot(T_w_k[:3,:3].T) + T_w_k[:3,3][None,None,:]

    valid_mask = depth > 0

    import interactive_roi as ir
    xys = ir.get_polyline(rgb, "rgb")
    xy_corner1 = np.clip(np.array(xys).min(axis=0), [0,0], [639,479])
    xy_corner2 = np.clip(np.array(xys).max(axis=0), [0,0], [639,479])
    polymask = ir.mask_from_poly(xys)
    #cv2.imshow("mask",mask)
        
    xy_tl = np.array([xy_corner1, xy_corner2]).min(axis=0)
    xy_br = np.array([xy_corner1, xy_corner2]).max(axis=0)

    xl, yl = xy_tl
    w, h = xy_br - xy_tl
    mask = np.zeros((h,w),dtype='uint8')    
    mask[polymask[yl:yl+h, xl:xl+w] > 0] = cv2.GC_PR_FGD
    print mask.shape
    #mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD

    tmp1 = np.zeros((1, 13 * 5))
    tmp2 = np.zeros((1, 13 * 5))    
    cv2.grabCut(rgb[yl:yl+h, xl:xl+w, :],mask,(0,0,0,0),tmp1, tmp2,10,mode=cv2.GC_INIT_WITH_MASK)

    mask = mask % 2
    #mask = ndi.binary_erosion(mask, utils_images.disk(args.erode)).astype('uint8')
    contours = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(rgb[yl:yl+h, xl:xl+w, :],contours,-1,(0,255,0),thickness=2)
    
    cv2.imshow('rgb', rgb)
    print "press enter to continue"
    cv2.waitKey()

    zsel = xyz_w[yl:yl+h, xl:xl+w, 2]
    mask = (mask%2==1) & np.isfinite(zsel)# & (zsel - table_height > -1)
    mask &= valid_mask[yl:yl+h, xl:xl+w]
    
    xyz_sel = xyz_w[yl:yl+h, xl:xl+w,:][mask.astype('bool')]
    return clouds.downsample(xyz_sel, .01)
    #rgb_sel = rgb[yl:yl+h, xl:xl+w,:][mask.astype('bool')]
        
    
def generate_hitch_points(pos, radius=0.016, length=0.215):
    ang = np.linspace(0, 2*np.pi, 30)
    circ_pts = radius*np.c_[np.cos(ang), np.sin(ang)] + pos[None,:2]
    circ_zs  = np.linspace(pos[2], length+pos[2], 30)

    rod_pts  = np.empty((0,3))
    for z in circ_zs:
        rod_pts = np.r_[rod_pts, np.c_[circ_pts, z*np.ones((len(circ_pts),1))] ]
        
    return rod_pts


def extract_red_alphashape(cloud, robot):
    import cloudprocpy
    """
    extract red, get alpha shape, downsample
    """
    raise NotImplementedError
    
    # downsample cloud
    cloud_ds = cloudprocpy.downsampleCloud(cloud, .01)
    
    # transform into body frame
    xyz1_kinect = cloud_ds.to2dArray()
    xyz1_kinect[:,3] = 1
    T_w_k = get_kinect_transform(robot)
    xyz1_robot = xyz1_kinect.dot(T_w_k.T)
    
    # compute 2D alpha shape
    xyz1_robot_flat = xyz1_robot.copy()
    xyz1_robot_flat[:,2] = 0 # set z coordinates to zero
    xyz1_robot_flatalphashape = cloudprocpy.computeAlphaShape(xyz1_robot_flat)
    
    # unfortunately pcl alpha shape func throws out the indices, so we have to use nearest neighbor search
    cloud_robot_flatalphashape = cloudprocpy.CloudXYZ()
    cloud_robot_flatalphashape.from2dArray(xyz1_robot_flatalphashape)
    cloud_robot_flat = cloudprocpy.CloudXYZ()
    cloud_robot_flat.from2dArray(xyz1_robot_flat)
    alpha_inds = cloudprocpy.getNearestNeighborIndices(xyz1_robot_flatalphashape, xyz1_robot_flat)

    xyz_robot_alphashape = xyz1_robot_flatalphashape[:,:3]
    
    # put back z coordinate
    xyz_robot_alphashape[:,2] = xyz1_robot[alpha_inds,2] 

    return xyz_robot_alphashape[:,:3]
