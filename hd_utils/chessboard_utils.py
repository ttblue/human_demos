import cv, cv2
import numpy as np, numpy.linalg as nlg

from pycb import extract_chessboards, get_3d_chessboard_points
from hd_utils import ros_utils as ru, conversions, transformations
from hd_utils.defaults import cam_mat, dist_coeffs

cb_rows = 8
cb_cols = 6



def get_corners_rgb(rgb, method='pycb', rows=None,cols=None):
    
    if not rows: rows = cb_rows
    if not cols: cols = cb_cols
    
    if method=='cv':
        cv_rgb = cv.fromarray(rgb)
        rtn, corners = cv.FindChessboardCorners(cv_rgb, (rows, cols))
        return rtn, corners
    elif method=='pycb':
        corners, cbs = extract_chessboards(rgb)
        if len(cbs) == 0:
            return 0, None
        elif len(cbs) == 1:
            return 1, corners[cbs[0]]
        else:
            cbcorners = []
            for cb in cbs:
                cbcorners.append(corners[cb])
            return len(cbs), cbcorners

def get_xyz_from_corners (corners, xyz):
    points = []
    for j,i in corners:
        x = i - np.floor(i)
        y = j - np.floor(j)
        p1 = xyz[np.floor(i),np.floor(j)]
        p2 = xyz[np.floor(i),np.ceil(j)]
        p3 = xyz[np.ceil(i),np.ceil(j)]
        p4 = xyz[np.ceil(i),np.floor(j)]        
        p = p1*(1-x)*(1-y) + p2*(1-x)*y + p3*x*y + p4*x*(1-y)
        if np.isnan(p).any(): print p
        points.append(p)

    return np.asarray(points)

def get_corresponding_points(points1, points2, guess_tfm, rows=None, cols=None):
    """
    Returns two lists of points such that the transform explains the relation between
    pointsets the most. Also, returns the norm of the difference between point sets.
    tfm is from cam1 -> cam2
    """
    if not rows: rows = cb_rows
    if not cols: cols = cb_cols

    
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    p12 = np.c_[points1,points2]
    p12 = p12[np.bitwise_not(np.isnan(p12).any(axis=1)),:]
    p1 = p12[:,0:3]
    p2 = p12[:,3:6]
    est = np.c_[p2,np.ones((p2.shape[0],1))].dot(guess_tfm.T)[:,0:3]
    dist = nlg.norm(p1-est,ord=np.inf)
    
    corr = range(rows*cols-1,-1,-1)
    p12r = np.c_[points1,points2[corr,:]]
    p12r = p12r[np.bitwise_not(np.isnan(p12r).any(axis=1)),:]
    p1r = p12r[:,0:3]
    p2r = p12r[:,3:6]
    est = np.c_[p2r,np.ones((p2r.shape[0],1))].dot(guess_tfm.T)[:,0:3]
    dist_new = nlg.norm(p1r-est, ord=np.inf)
    if dist_new < dist:
        points1, points2, dist = p1, p2, dist_new
    else:
        points1, points2 = p1, p2

    return points1, points2, dist


def get_corners_from_pc(pc,method='pycb', rows=None,cols=None):
    xyz, rgb = ru.pc2xyzrgb(pc)
    rgb = np.copy(rgb)
    rtn, corners = get_corners_rgb(rgb, method, rows, cols)
    if rtn == 1:
        points = get_xyz_from_corners(corners, xyz)
        return rtn, points
    if len(corners) == 0 or rtn == 0:
            return 0, None
    if method=='pycb':
        allpoints = []
        for cor in corners:
            cor = cor.reshape(cor.shape[0]*cor.shape[1],2,order='F')
            points.append(get_zyx_from_corners(cor, xyz))
        return rtn, points
    
def get_checkerboard_transform (rgb, chessboard_size=0.024):
    """
    Assuming only one checkerboard is found.
    """

    rtn, corners = get_corners_rgb(rgb, method='pycb')
    if rtn <= 0:
        print "Failed", rtn
        return None
    
    print "Checking",rtn
    if rtn == 1:
        objPoints = get_3d_chessboard_points(corners.shape[0],corners.shape[1],cb_size)
        imgPoints = corners.reshape(corners.shape[0]*corners.shape[1],2,order='F')
        
        
        ret, rvec, tvec = cv2.solvePnP(objPoints, imgPoints, cam_mat, dist_coeffs)
        print ret
        if ret:
            ang = np.linalg.norm(rvec)
            dir = rvec/ang
            
            tfm = transformations.rotation_matrix(ang,dir)
            tfm[0:3,3] = np.squeeze(tvec)
            return tfm
        else:
            return None