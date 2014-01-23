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

def best_two_in_cluster(cluster, sm):
	costs = np.zeros((len(cluster), len(cluster)))
	for i in xrange(len(cluster)):
		for j in xrange(len(cluster)):
			costs[i][j] = sm[cluster[i]][cluster[j]]
	sum_costs = np.sum(costs, axis=1)
	ranking = np.argsort(sum_costs)
	return cluster[ranking[0]], cluster[ranking[1]]


def find_best(demos, sm):
	best_demos = []
	for i in xrange(len(demos)):
		cluster = demos[i]
		one, two = best_two_in_cluster(cluster, sm)
		best_demos.append(one)
		best_demos.append(two)
	return best_demos

def main(demo_type, n_clusters):
	demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
	smfile = "similarity_matrix_tps_initial.cp"
	with open(smfile, 'r') as f: sm = pickle.load(f)
	seg_num = 0
	keys = {}
	for demo_name in demofile:
		if demo_name != "ar_demo":
			for seg_name in demofile[demo_name]:
				if seg_name == 'seg00':
					keys[seg_num] = (demo_name, seg_name)
					#print demo_name, seg_name
					seg_num += 1
	
	labels = spectral_clustering(sm, n_clusters = n_clusters, eigen_solver='arpack',assign_labels='discretize')
	names = {i:[] for i in xrange(args.num_clusters)}
	images = {i:[] for i in xrange(args.num_clusters)}
	demos = {i:[] for i in xrange(args.num_clusters)}
	
	for i in xrange(len(labels)):
		label = labels[i]
		names[label].append(keys[i])
		images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))
		demos[label].append(i)
	bests = find_best(demos, sm)
	print bests
	best_xyz = []
	for best in bests:
		best_xyz.append(np.asarray(demofile[keys[i][0]][keys[i][1]]["xyz"]))
	pickle.dump(best_xyz, 'initial.h5')

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
	args = parser.parse_args()

	main(args.demo_type, args.num_clusters)
