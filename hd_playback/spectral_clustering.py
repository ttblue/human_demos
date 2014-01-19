import h5py
import numpy as np
#from rapprentice import registration

from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
from sklearn.cluster import spectral_clustering
import cPickle as pickle
import argparse
import cv2, hd_rapprentice.cv_plot_utils as cpu
import os.path as osp
import time

from hd_rapprentice import registration
from hd_utils import clouds
from hd_utils.defaults import demo_files_dir

np.set_printoptions(precision=6, suppress=True)

def lerp (x, xp, fp, first=None):
	"""
	Returns linearly interpolated n-d vector at specified times.
	"""
	
	fp = np.asarray(fp)
	
	fp_interp = np.empty((len(x),0))
	for idx in range(fp.shape[1]):
		if first is None:
			interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx])).T
		else:
			interp_vals = np.atleast_2d(np.interp(x,xp,fp[:,idx],left=first[idx])).T
		fp_interp = np.c_[fp_interp, interp_vals]
	
	return fp_interp



def registration_cost(xyz0, xyz1):
# 	scaled_xyz0, _ = registration.unit_boxify(xyz0)
# 	scaled_xyz1, _ = registration.unit_boxify(xyz1)
	
	f,g = registration.tps_rpm_bij(xyz0, xyz1, rot_reg=1e-3, n_iter=50)
	cost = (registration.tps_reg_cost(f) + registration.tps_reg_cost(g))/2.0
	return cost


def extract_clouds(demofile):
	seg_num = 0
	leaf_size = 0.045
	keys = {}
	pclouds = []
	for demo_name in demofile:
		if demo_name != "ar_demo":
			for seg_name in demofile[demo_name]:
				if seg_name != 'done':
					keys[seg_num] = (demo_name, seg_name)
					pclouds.append(clouds.downsample(np.asarray(demofile[demo_name][seg_name]["cloud_xyz"]), leaf_size))
					print demo_name, seg_name
					seg_num += 1
	return keys, pclouds


def traj_cost(traj1, traj2, n, find_corr=False):
	"""
	Downsamples traj to have n points from start to end.
	"""
	
	ts1 = np.linspace(0,traj1.shape[0],n)
	ts2 = np.linspace(0,traj2.shape[0],n)
	
	xyz1 = lerp(ts1, range(traj1.shape[0]), traj1)
	xyz2 = lerp(ts2, range(traj2.shape[0]), traj2)
		
	if find_corr:
		return registration_cost(xyz1, xyz2)
	else:
		bend_c = 0.05
		rot_c = [1e-3, 1e-3, 1e-3]
		scale_c = 0.1
		f = registration.fit_ThinPlateSpline_RotReg(xyz1, xyz2, bend_c, rot_c, scale_c)
		g = registration.fit_ThinPlateSpline_RotReg(xyz2, xyz1, bend_c, rot_c, scale_c)
		return (registration.tps_reg_cost(f) + registration.tps_reg_cost(g))/2.0


def tot_cost (seg1, seg2):
	n = 20
	weights = np.array([[1,1]])

	costs = np.array([[registration_cost(seg1[0], seg2[0]),
			 traj_cost(seg1[1], seg2[1], n)+
			 traj_cost(seg1[2], seg2[2], n)]])

	return float(weights.dot(costs.T))


def similarity_matrix_segs(segs, n_segs = None):
	print "computing costs"
	if n_segs is None:
		n_segs = len(segs)
	cost_matrix = np.zeros((0, n_segs))
	for y in xrange(n_segs):
		new_seg = segs[y]
		ts = time.time()
		costs = [0.0]*y
		costs.extend(Parallel(n_jobs=3,verbose=51)(delayed(tot_cost)(seg, new_seg) for seg in segs[y:n_segs]))
		te = time.time()
		print y
		print "%f"%(te-ts)
		costs = np.array(costs)
		cost_matrix = np.vstack((cost_matrix, costs))
		
	cost_matrix = cost_matrix + cost_matrix.T - np.diag(np.diag(cost_matrix))
# 	for y in xrange(n_segs):
# 		for x in xrange(y+1, n_segs):
# 			cost_matrix[x][y] = cost_matrix[y][x]
	similarity_matrix = np.exp(-cost_matrix)

	return similarity_matrix


def similarity_matrix(pclouds, n_clouds = None):
	print "computing tps costs"
	if n_clouds is None:
		n_clouds = len(pclouds)
	cost_matrix = np.zeros((0, n_clouds))
	for y in xrange(n_clouds):
		new_xyz = pclouds[y]
		ts = time.time()
		costs = [0]*y	
		costs.extend(Parallel(n_jobs=4,verbose=51)(delayed(registration_cost)(c, new_xyz) for c in pclouds[y:n_clouds]))	
		te = time.time()
		print y
		print "%f"%(te-ts)
		costs = np.array(costs)
		cost_matrix = np.vstack((cost_matrix, costs))
	for y in xrange(n_clouds):
		for x in xrange(y+1, n_clouds):
			cost_matrix[x][y] = cost_matrix[y][x]
	similarity_matrix = np.exp(-cost_matrix)

	return similarity_matrix


def extract_segs(demofile):
	seg_num = 0
	leaf_size = 0.045
	keys = {}
	segs = []
	for demo_name in demofile:
		if demo_name != "ar_demo":
			for seg_name in demofile[demo_name]:
				if seg_name != 'done':
					keys[seg_num] = (demo_name, seg_name)
					seg = demofile[demo_name][seg_name]
					pc = clouds.downsample(np.asarray(seg['cloud_xyz']), leaf_size)
					segs.append((pc, np.asarray(seg['l']['tfms_s'])[:,0:3,3], np.asarray(seg['r']['tfms_s'])[:,0:3,3]))
					print demo_name, seg_name
					seg_num += 1
	return keys, segs


def main(demo_type, n_clusters, n_segs=None, use_clouds=False):
	demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
	iden = ''
	if n_segs is not None:
		iden = str(n_segs)
	if use_clouds:
		iden += '_clouds'
	smfile = "sim_matrix"+iden+".cp"
	try:
		with open(smfile, 'r') as f:
			sm = pickle.load(f)
		seg_num = 0
		keys = {}
		for demo_name in demofile:
			if demo_name != "ar_demo":
				for seg_name in demofile[demo_name]:
					if seg_name != 'done':
						keys[seg_num] = (demo_name, seg_name)
						print demo_name, seg_name
						seg_num += 1
						if n_segs is not None and seg_num >= n_segs: break 

	except IOError:
		if use_clouds:
			keys, clouds = extract_clouds(demofile)
			sm = similarity_matrix(clouds,n_segs)
		else:
			keys, segs = extract_segs(demofile)
			sm = similarity_matrix_segs(segs,n_segs)
		with open(smfile, 'wa') as f:
			pickle.dump(sm, f)
	print sm
	
	labels = spectral_clustering(sm, n_clusters = n_clusters, eigen_solver='arpack',assign_labels='discretize')
	names = {i:[] for i in xrange(args.num_clusters)}
	images = {i:[] for i in xrange(args.num_clusters)}
	
	print keys
	for i in xrange(len(labels)):
		label = labels[i]
		names[label].append(keys[i])
		images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))



	rows = []
	i = 0
	print "Press q to exit, left/right arrow keys to navigate"
	while True:
		print "Label %i"%(i+1)
		print names[i]
		import math
		ncols = 7
		nrows = int(math.ceil(1.0*len(images[i])/ncols))
		row = cpu.tile_images(images[i], nrows, ncols)
		rows.append(np.asarray(row))
		cv2.imshow("clustering result", row)
		kb = cv2.waitKey()
		if kb == 1113939:
			i = min(i+1,args.num_clusters-1)
		elif kb == 1113937:
			i = max(i-1,0)
		elif kb == 1048689:
			break
	return
	bigimg = cpu.tile_images(rows, len(rows), 50)
	cv2.imshow("clustering result", bigimg)
	print "press any key to continue"
	cv2.waitKey()




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo_type",help="Demo type.", type=str)
	parser.add_argument("--num_clusters", type=int)
	parser.add_argument("--num_segs", type=int, default=-1)
	parser.add_argument("--use_clouds", action="store_true", default=False)
	args = parser.parse_args()

	if args.num_segs < 0:
		ns = None
	else:
		ns = args.num_segs
	main(args.demo_type, args.num_clusters, ns, args.use_clouds)
