import numpy as np, numpy.linalg as nlg
import scipy as scp, scipy.optimize as sco
from GPR import gpr
from Tools.general import feval
import openravepy as rave

def tf_to_vec(tf):
    return tf[:3,:].reshape(-1,1)

def vec_to_tf(vec):
    tf = vec.reshape(3,4)
    tf = np.r_[tf,np.array([[0,0,0,1]])]
    return tf

def tf_to_pose(tf):
    return np.r_[tf[:3,3], rave.axisAngleFromRotationMatrix(tf[:3,:3])]

def pose_to_tf(pose):
    tf = np.eye(4)
    tf[:3,:3] = rave.rotationMatrixFromAxisAngle(np.array(pose[3:]))
    tf[:3,3] = pose[:3]
    return tf

# fast matrix inversion (special case of homogeneous transform)
def tf_inv(tf):
    R = tf[:3,:3]
    t = tf[:3,3]
    inv = np.eye(4)
    inv[:3,:3] = R.T
    inv[:3,3] = -R.T.dot(t)
    return inv

# pose error takes from tf1 to tf0
# typically, tf0 is ground truth and tf1 is the pose being fixed
def calc_pose_error(tf0, tf1):
    n_data = len(tf0) # should be the same for tf1
    pose_error = np.empty((n_data, 6))
    for i_data in range(n_data):
        tf1_to_tf0 = tf0[i_data].dot(tf_inv(tf1[i_data]))
        pose_error[i_data,:] = tf_to_pose(tf1_to_tf0)
    return pose_error

# Finds tf such that gt_pose = tf.dot(pose)
def sys_correct_tf(gt_poses, poses, tf_init):
	x_init = tf_to_vec(tf_init)
	#x_init = tf_to_vec(np.eye(4))

	# Objective function:
	def f_opt (x):
		  n_poses = len(gt_poses)
		  err_vec = np.zeros((0, 1))
		  for i in range(n_poses):
		      #err_tf = vec_to_tf(x).dot(poses[i]).dot(tf_inv(gt_poses[i])) - np.eye(4)
		      err_tf = vec_to_tf(x).dot(poses[i]) - gt_poses[i]
		      err_vec = np.r_[err_vec, tf_to_vec(err_tf)]
		  ret = nlg.norm(err_vec)
		  return ret

	# Rotation constraint:
	def rot_con (x):
		  R = vec_to_tf(x)[0:3,0:3]
		  err_mat = R.T.dot(R) - np.eye(3)
		  ret = nlg.norm(err_mat)
		  return ret

	(X, fx, _, _, _) = sco.fmin_slsqp(func=f_opt, x0=x_init, eqcons=[rot_con], iter=50, full_output=1)

	#print "Function value at optimum: ", fx

	tf = vec_to_tf(np.array(X))
	return tf

def gp_pred_precompute_alpha(logtheta, covfunc, X, y):
	# compute training set covariance matrix (K)
	K = feval(covfunc, logtheta, X)                     # training covariances
	L = nlg.cholesky(K)                      # cholesky factorization of cov (lower triangular matrix)
	alpha = gpr.solve_chol(L.transpose(),y)         # compute inv(K)*y
	return alpha

# same as gp_pred except that uses the precomputed alpha (what is returned from gp_pred_precompute_alpha())
def gp_pred_fast(logtheta, covfunc, X, alpha, Xstar):
	# (marginal) test predictions (Kss = self-cov; Kstar = corss-cov)
	[Kss, Kstar] = feval(covfunc, logtheta, X, Xstar)   # test covariances (Kss = self covariances, Kstar = cov between train and test cases)
	return np.dot(Kstar.transpose(),alpha)         # predicted means

# gt_poses_train (list of n_data homogeneous TF matrices)
# poses_train (list of n_data homogeneous TF matrices)
# state_train (np.array of shape (n_data x d))
def gp_correct_poses_precompute(gt_poses_train, poses_train, state_train):
	pose_error = calc_pose_error(gt_poses_train, poses_train)
	n_task_vars = 6
	alphas = []
	for i_task_var in range(n_task_vars):
		## data from a noisy GP
		X = state_train

		## DEFINE parameterized covariance funcrion
		covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
		## SET (hyper)parameters
		logtheta = np.array([np.log(1), np.log(1), np.log(np.sqrt(0.01))])

		#print 'hyperparameters: ', np.exp(logtheta)

		### sample observations from the GP
		y = pose_error[:,i_task_var]

		## PREDICTION precomputation
		alphas.append(gp_pred_precompute_alpha(logtheta, covfunc, X, y))
	return alphas

# alphas (list of n_task_vars as returned by gp_correct_poses_precompute())
# state_train (np.array of shape (n_data x d))
# poses_test (list of n_test homegeneous TF matrices)
# state_test (np.array of shape (n_test x d))
def gp_correct_poses_fast(alphas, state_train, poses_test, state_test):
	n_test = len(poses_test)
	n_task_vars = 6

	MU = np.empty((n_test, n_task_vars))

	for i_task_var in range(n_task_vars):
		## data from a noisy GP
		X = state_train

		## DEFINE parameterized covariance funcrion
		covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
		## SET (hyper)parameters
		logtheta = np.array([np.log(1), np.log(1), np.log(np.sqrt(0.1))])

		#print 'hyperparameters: ', np.exp(logtheta)

		### precomputed alpha
		alpha = alphas[i_task_var]

		### TEST POINTS
		Xstar = state_test

		## PREDICTION 
		MU[:,i_task_var] = gp_pred_fast(logtheta, covfunc, X, alpha, Xstar) # get predictions for unlabeled data ONLY
	
	est_gt_poses_test = [pose_to_tf(pose_diff).dot(pose) for (pose_diff, pose) in zip(MU, poses_test)]
	return est_gt_poses_test

# gt_poses_train (list of n_data homogeneous TF matrices)
# poses_train (list of n_data homogeneous TF matrices)
# state_train (np.array of shape (n_data x d))
# poses_test (list of n_test homegeneous TF matrices)
# state_test (np.array of shape (n_test x d))
def gp_correct_poses(gt_poses_train, poses_train, state_train, poses_test, state_test, hy_params=None):
	pose_error = calc_pose_error(gt_poses_train, poses_train)
	
	n_test = len(poses_test)
	n_task_vars = 6

	MU = np.empty((n_test, n_task_vars))
	S2 = np.empty((n_test, n_task_vars))

	for i_task_var in range(n_task_vars):
		  ## data from a noisy GP
		  X = state_train

		  ## DEFINE parameterized covariance funcrion
		  covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
		  ## SET (hyper)parameters
                  if hy_params==None:
                          logtheta = np.array([np.log(5), np.log(5), np.log(np.sqrt(0.5))])
                  else:
                          logtheta = hy_params[i_task_var,:]
		  #print 'hyperparameters: ', np.exp(logtheta)

		  ### sample observations from the GP
		  y = pose_error[:,i_task_var]

		  ### TEST POINTS
		  Xstar = state_test

		  ## PREDICTION 
		  print 'GP: ...prediction'
		  results = gpr.gp_pred(logtheta, covfunc, X, y, Xstar) # get predictions for unlabeled data ONLY
		  MU[:,i_task_var] = results[0]
		  S2[:,i_task_var:i_task_var+1] = results[1]
	
	est_gt_poses_test = [pose_to_tf(pose_diff).dot(pose) for (pose_diff, pose) in zip(MU, poses_test)]
	return est_gt_poses_test


def train_hyperparams(gt_poses_train, poses_train, state_train):
        """
        Returns the tuned hyper-params for 6 GPs.
        Hence, returns an 3x6 matrix (3 params per DOF).

        gt_poses_train, poses_train, state_train are as in the
        function gp_correct_poses.
        """
        covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
        pose_error  = calc_pose_error(gt_poses_train, poses_train)
        hparams     = np.empty((6,3))
        n_task_vars = 6

        for i_task_var in range(n_task_vars):
                X = state_train
                logtheta = np.array([-1,-1,-1])
                y = pose_error[:,i_task_var]
                hparams[i_task_var,:] = gpr.gp_train(logtheta, covfunc, X, np.reshape(y, (y.shape[0], 1)))
        print 'Tuned hyper-params: ', hparams
        return hparams


def print_formatted_error_stat(error):
	error_stat = ""
	for (mean, std) in zip(np.mean(error, axis=0).tolist(), np.std(error, axis=0).tolist()):
		error_stat += str(mean) + " +/- " + str(std) + "\t"
	print error_stat
