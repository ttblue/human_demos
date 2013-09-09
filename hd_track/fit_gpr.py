import error_characterization as ec
from hd_track.test_kalman_hydra import load_data
import numpy as np
import matplotlib.pylab as plt
from hd_utils.colorize import colorize
from hd_track.kalman_tuning import state_from_tfms


r_split=0.5
Ts_bh, Ts_bg, T_gh, Ts_bg_gh, X_bh, X_bg_gh = load_data()
X_bg_gh = np.c_[X_bg_gh[:,0], X_bg_gh]
X_bh    = np.c_[X_bh[:,0], X_bh]


N = len(Ts_bh)
n_split = int(r_split*N)

# indices for the training data
pr2_1 = [Ts_bg_gh[i] for i in xrange(N) if i%4 == 0 ]
pr2_2 = [Ts_bg_gh[i] for i in xrange(N) if i%4 != 0 ]
hy_1 = [Ts_bh[i] for i in xrange(N) if i%4 == 0]
hy_2 = [Ts_bh[i] for i in xrange(N)  if i%4 != 0]
xp_1 = np.take(X_bg_gh[6:9,:].T, [i for i in xrange(N) if i%4 == 0], axis=0)
xp_2 = np.take(X_bg_gh[6:9,:].T, [i for i in xrange(N) if i%4 != 0], axis=0)


xpf_2 = np.take(X_bg_gh.T, [i for i in xrange(N) if i%4 != 0], axis=0)
xh_2 = np.take(X_bh.T, [i for i in xrange(N) if i%4 != 0], axis=0)



#A,B,C,D,E = Ts_bg_gh[:n_split], Ts_bh[:n_split],  X_bg_gh[6:9,:n_split].T, Ts_bh[n_split:], X_bg_gh[6:9,n_split:].T
A,B,C,D,E = pr2_1, hy_1, xp_1, hy_2, xp_2

ests = ec.gp_correct_poses(A,B,C,D,E)
X_est = state_from_tfms(ests, dt=1./30.).T
X_est = np.c_[X_est[:,0], X_est]


axlabels = ['x','y','z','roll','pitch','yaw']
plt.clf()

xpf_t_2 = xpf_2.T
xh_t_2 = xh_2.T


for i in xrange(6):
    j = i+4 if i > 2 else i

    plt.subplot(3,2,i+1)
    plt.plot(xpf_t_2[j,:], label='pr2')
    plt.plot(xh_t_2[j,:], label='hydra')
    plt.plot(X_est[j, :], label='estimate')
    plt.ylabel(axlabels[i])
    plt.legend()
plt.show()


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
