import numpy as np

cx = 320.-.5
cy = 240.-.5
DEFAULT_F = 535.

def xyZ_to_XY(x,y,Z,f=DEFAULT_F):
    X = (x - cx)*(Z/f)
    Y = (y - cy)*(Z/f)
    return (X,Y)

def XYZ_to_xy(X,Y,Z,f=DEFAULT_F):
    x = X*(f/Z) + cx
    y = Y*(f/Z) + cy
    return (x,y)

def depth_to_xyz(depth,f=DEFAULT_F):
    x,y = np.meshgrid(np.arange(640), np.arange(480))
    assert depth.shape == (480, 640)
    XYZ = np.empty((480,640,3))
    
    Z = XYZ[:,:,2] = depth / 1000. # convert mm -> meters
    np.putmask(Z, Z==0, np.nan)
    
    XYZ[:,:,0] = (x - cx)*(Z/f)
    XYZ[:,:,1] = (y - cy)*(Z/f)

    return XYZ
    
def downsample(xyz, v):
    import cloudprocpy
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)
    cloud = cloudprocpy.downsampleCloud(cloud, v)
    return cloud.to2dArray()[:,:3]

def median_filter(xyz, window_size, max_allowed_movement):
    import cloudprocpy
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)
    cloud = cloudprocpy.medianFilter(cloud, window_size, max_allowed_movement)
    return cloud.to2dArray()[:,:3]


def remove_outliers(xyz, thresh=2.0, k=15):
    import cloudprocpy
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)
    cloud = cloudprocpy.removeOutliers(cloud, thresh, k)
    return cloud.to2dArray()[:,:3]

def cluster_filter(xyz, tol=0.08, minsize=200):
    import cloudprocpy
    cloud = cloudprocpy.CloudXYZ()
    xyz1 = np.ones((len(xyz),4),'float')
    xyz1[:,:3] = xyz
    cloud.from2dArray(xyz1)

    while True:
        filter_result = cloudprocpy.clusterFilter(cloud, tol, minsize)
        filter_result = filter_result.to2dArray()[:,:3]
        if len(filter_result) > 0:
            cloud = filter_result
            break
        else:
            minsize = int(0.5 * minsize)

    return cloud

def clouds_plane(xyz):
    A = np.cov(xyz.T)
    U, _, _ = np.linalg.svd(A)
    
    if U[2, -1] < 0:
        return -U[:, -1]
    else:
        return U[:, -1]
    

