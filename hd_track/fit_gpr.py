import error_characterization as ec
from hd_track.test_kalman_hydra import load_data
import numpy as np
import matplotlib.pylab as plt
from hd_utils.colorize import colorize
import hd_utils.transformations as tfms
from hd_track.kalman_tuning import state_from_tfms
from hd_visualization.mayavi_plotter import *
import openravepy as rave
import cPickle

r_split=0.5
Ts_bh, Ts_bg, T_gh, Ts_bg_gh, X_bh, X_bg_gh = load_data()
X_bg_gh = np.c_[X_bg_gh[:,0], X_bg_gh]
X_bh    = np.c_[X_bh[:,0], X_bh]

N = len(Ts_bh)
n_split = int(r_split*N)

# indices for the training data
def gen_indices(N, k):
    """
    generate indices for splitting data in blocks
    """
    n = int(N/k)
    inds = np.arange(k)
    
    trn = np.empty(0, dtype=int)
    tst = np.empty(0, dtype=int)

    for i in xrange(n):
        if i%2==0:
            trn = np.r_[trn , (k*i + inds)]
        else:
            tst = np.r_[tst , (k*i + inds)]
    return (trn, tst)

 
def rpy2axang(rpy):
    """
    Converts a matrix of rpy (nx3) into a matrix of 
    axis-angles (nx3). 
    """
    n = rpy.shape[0]
    assert rpy.shape[1]==3, "unknown shape."
    ax_ang = np.empty((n,3))
    for i in xrange(n):
        th = rpy[i,:]
        ax_ang[i,:] = rave.axisAngleFromRotationMatrix(tfms.euler_matrix(th[0], th[1], th[2]))
    return ax_ang

BLOCK_WIDTH = 100
trn, tst = gen_indices(N, BLOCK_WIDTH)
print trn
print tst

pr2_1 = [Ts_bg_gh[i] for i in trn]
pr2_2 = [Ts_bg_gh[i] for i in tst]
hy_1 = [Ts_bh[i] for i in trn]
hy_2 = [Ts_bh[i] for i in tst]
xp_1 = np.take(X_bg_gh[6:9,:].T, [i for i in trn], axis=0)
xp_2 = np.take(X_bg_gh[6:9,:].T, [i for i in tst], axis=0)

## visualize the states -- the roll pitch yaws.
plotter = PlotterInit()
ax1     = rpy2axang(xp_1)
ax2     = rpy2axang(xp_2)
plotter.request(gen_mlab_request(mlab.points3d, ax1[:,0], ax1[:,1], ax1[:,2], color=(0,1,0), scale_factor=0.05))
plotter.request(gen_mlab_request(mlab.points3d, ax2[:,0], ax2[:,1], ax2[:,2], color=(1,0,0), scale_factor=0.05))


xpf_2 = np.take(X_bg_gh.T, [i for i in tst], axis=0)
xh_2  = np.take(X_bh.T, [i for i in tst], axis=0)

#A,B,C,D,E = Ts_bg_gh[:n_split], Ts_bh[:n_split],  X_bg_gh[6:9,:n_split].T, Ts_bh[n_split:], X_bg_gh[6:9,n_split:].T
A,B,C,D,E = pr2_1, hy_1, xp_1, hy_2, xp_2

#hi_params = ec.train_hyperparams(A,B,C)
#cPickle.dump(hi_params, open('hyper-params.cpkl', 'wb')) ## save the hyper-parameters to a file.
ests = ec.gp_correct_poses(A,B,C,D,E, None)
X_est = state_from_tfms(ests, dt=1./30.).T
X_est = np.c_[X_est[:,0], X_est]

axlabels = ['x','y','z','roll','pitch','yaw']
plt.clf()

xpf_t_2 = xpf_2.T
xh_t_2 = xh_2.T

for i in xrange(6):
    j = i+3 if i > 2 else i

    plt.subplot(3,2,i+1)
    plt.plot(xpf_t_2[j,:], label='pr2')
    plt.plot(xh_t_2[j,:], label='hydra')
    plt.plot(X_est[j, :], label='estimate')
    plt.ylabel(axlabels[i])
    plt.legend()
plt.show(block=False)


orig_cal_pose_error = ec.calc_pose_error(A, B)
print colorize("original camera calibration pose error", "blue", True)
print "mean", np.mean(orig_cal_pose_error, axis=0)
print "std", np.std(orig_cal_pose_error, axis=0)
print colorize("--------------------------------------", "blue", True)


# systematic and GP corrected robot poses
gp_pose_error = ec.calc_pose_error(D, ests)
print colorize("systematic and GP corrected pose error", "blue", True)
print "mean", np.mean(gp_pose_error, axis=0)
print "std", np.std(gp_pose_error, axis=0)
print colorize("--------------------------------------", "blue", True)
