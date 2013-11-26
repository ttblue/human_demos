import numpy as np
import os.path as osp
from mayavi import mlab
from reg_clouds.mayavi_plotter import *
import openravepy as rave
import scipy.optimize as opt
from scipy import spatial


dirname = './build/'
c1name = 'c1_icped.pcd'
c2name = 'c2_out.pcd'

c1name = osp.join(dirname, c1name)
c2name = osp.join(dirname, c2name)

pc1 = np.loadtxt(c1name, skiprows=11)[::5,:3]
pc2 = np.loadtxt(c2name, skiprows=11)[::5,:3]

plotter = PlotterInit()

## calculate kd-tree for pc1:
print "> computing KD-Tree.."
pc1tree = spatial.KDTree(pc1.copy())
print "\t KD-Tree done."

def calc_T(xyz, rod):
    Tt = rave.matrixFromAxisAngle(rod)
    Tt[:3,3] = xyz
    return Tt

def plot_point(x):
    """
    display intermediate result : in a callback
    """
    Tg   = calc_T(*vec2args(x))
    pc2t = (np.c_[pc2, np.ones((pc2.shape[0],1))].dot(Tg.T))[:,:3]

    clearreq = gen_mlab_request(mlab.clf)
    plotter.request(clearreq)

    c1req   =  gen_mlab_request(mlab.points3d, pc1[:,0], pc1[:,1], pc1[:,2], color=(1,0,0), scale_factor=0.001)
    c2req   =  gen_mlab_request(mlab.points3d, pc2t[:,0], pc2t[:,1], pc2t[:,2], color=(0,1,0), scale_factor=0.001)
    plotter.request(c1req)
    plotter.request(c2req)


i = 0
def calc_error(xyz, rod):
    global i
    print "\t %d : calc.."%i
    i+=1
    Tg   = calc_T(xyz, rod)
    pc2t = (np.c_[pc2, np.ones((pc2.shape[0],1))].dot(Tg.T))[:,:3]
    d,i_nn   = pc1tree.query(pc2t)
    d = d[d<0.1]
    return np.sum(d)

def calc_error_wrapper(x):
    return calc_error(*vec2args(x))
def vec2args(x):
    return x[0:3], x[3:6]
def args2vec(xyz, rod):
    out = np.empty(6)
    out[0:3] = xyz
    out[3:6] = rod
    return out

T_init   = np.eye(4)
xyz_init = T_init[:3,3]
rod_init = rave.axisAngleFromRotationMatrix(T_init[:3,:3])

print "> optimizing..."
soln = opt.fmin(calc_error_wrapper, args2vec(xyz_init, rod_init), xtol=1e-2, ftol=1e-1, callback=plot_point)
print "\t optimization done."

(best_xyz, best_rod) = vec2args(soln)
print "xyz, rod:", best_xyz, best_rod
T_best = calc_T(best_xyz, best_rod)
print "T_h_k:", T_best

## final display:
pc2 = (np.c_[pc2, np.ones((pc2.shape[0],1))].dot(T_best.T))[:,:3]
clearreq = gen_mlab_request(mlab.clf)
plotter.req(clearreq)
c1req   =  gen_mlab_request(mlab.points3d, pc1[:,0], pc1[:,1], pc1[:,2], color=(1,0,0), scale_factor=0.001)
c2req   =  gen_mlab_request(mlab.points3d, pc2t[:,0], pc2t[:,1], pc2t[:,2], color=(0,1,0), scale_factor=0.001)
plotter.request(c1req)
plotter.request(c2req)
