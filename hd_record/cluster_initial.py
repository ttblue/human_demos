import h5py
import numpy as np
#from rapprentice import registration

from joblib import Parallel, delayed
from sklearn.cluster import spectral_clustering
import cPickle as pickle
import argparse
import cv2, hd_rapprentice.cv_plot_utils as cpu
from mpl_toolkits.mplot3d import axes3d
import os.path as osp
import pylab

from hd_utils.clouds_utils import sample_random_rope
from hd_utils.defaults import demo_files_dir, hd_data_dir, similarity_costs_dir, \
	matrix_file, perturbation_file
from hd_utils.yes_or_no import yes_or_no

from extract_init import extract_init

np.set_printoptions(precision=6, suppress=True)

BASIC_DEMOS = [0,19,39,61]

def best_n_in_cluster(cluster, sm, n=2):
	costs = np.zeros((len(cluster), len(cluster)))
	for i in xrange(len(cluster)):
		for j in xrange(len(cluster)):
			costs[i][j] = sm[cluster[i]][cluster[j]]
	sum_costs = np.sum(costs, axis=1)
	ranking = np.argsort(-sum_costs)
	print ranking
	
	if n ==1:
		return cluster[ranking[0]]
	
	n = min(n, len(ranking))
	return [cluster[ranking[i]] for i in range(n)]


def find_best(demos, sm):
	best_demos = []
	for i in xrange(len(demos)):
		cluster = demos[i]
		if len(cluster) == 0: continue
		best_n_in_cluster(cluster, sm, n=1)
		best_demos.append(best_n_in_cluster(cluster, sm, n=1))
	return best_demos

def main(demo_type, n_base, n_perts, load_sm = False):
	demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
	if load_sm:
		sm_file = osp.join(hd_data_dir, similarity_costs_dir, matrix_file%demo_type)
		with open(sm_file, 'r') as f: sm = pickle.load(f)
	else:
		sm = extract_init(demo_type)

	seg_num = 0
	keys = {}
	for demo_name in demofile:
		if demo_name != "ar_demo":
			for seg_name in demofile[demo_name]:
				if seg_name == 'seg00':
					keys[seg_num] = (demo_name, seg_name)
					#print demo_name, seg_name
					seg_num += 1
	
	n_clusters = n_base - len(BASIC_DEMOS)
	labels = spectral_clustering(sm, n_clusters = n_clusters, eigen_solver='arpack',assign_labels='discretize')
	names = {i:[] for i in xrange(n_clusters)}
	images = {i:[] for i in xrange(n_clusters)}
	demos = {i:[] for i in xrange(n_clusters)}
	
	for i in xrange(len(labels)):
		label = labels[i]
		names[label].append(keys[i])
		images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))
		demos[label].append(i)

	rows = []
	i = 0
	inc = True
	
	print "Press q to exit, left/right arrow keys to navigate"
	while True:
		if len(images[i]) == 0:
			if i == n_clusters-1: inc = False
			elif i == 0: inc = True
				
			if inc: i = min(i+1,n_clusters-1)
			else: i = max(i-1,0)
			
			continue

		print "Label %i"%(i+1)
		print names[i]
		import math
		ncols = 7
		nrows = int(math.ceil(1.0*len(images[i])/ncols))

		row = cpu.tile_images(images[i], nrows, ncols)
		rows.append(np.asarray(row))
		cv2.imshow("clustering result", row)
		kb = cv2.waitKey()
		if kb == 1113939 or kb == 65361:
			i = min(i+1,n_clusters-1)
			inc = True
		elif kb == 1113937 or kb == 65361:
			i = max(i-1,0)
			inc = False
		elif kb == 1048689 or kb == 113:
			break
	
	bests = find_best(demos, sm)
	for i in BASIC_DEMOS:
		if i in bests:
			bests.remove(i)
	# add basic demos
	bests = BASIC_DEMOS+bests
	print bests

	best_xyz = []
	best_images = {i:None for i in bests}
	for best in bests:
		xyz = np.asarray(demofile[keys[best][0]][keys[best][1]]["cloud_xyz"])
		best_xyz.append(xyz)
		best_images[best] = np.asarray(demofile[keys[best][0]][keys[best][1]]["rgb"])
		

	print"Found %i clouds."%len(bests)
	
	print "These are the demos being saved."

	ncols = 10
	nrows = int(math.ceil(1.0*len(bests)/ncols))
	row = cpu.tile_images(best_images.values(), nrows, ncols)
	cv2.imshow("best", row)
	kb = cv2.waitKey()
	
	if not yes_or_no("Do these look fine to you?"):
		return
	
	remaining = n_base-len(bests)
	print "remaining:", remaining
	while remaining > 0:
		fig = pylab.figure()
		xyz = best_xyz[remaining-1]
		while True:
			fig.clf()
			perturbed_xyz = sample_random_rope(xyz, True)
			ax = fig.gca(projection='3d')
			ax.set_autoscale_on(False)
			ax.plot(perturbed_xyz[:,0], perturbed_xyz[:,1], perturbed_xyz[:,2], 'o')   
			fig.show()
			cv2.imshow("pert", best_images[bests[remaining-1]])
			kb = cv2.waitKey()
			if yes_or_no("Does this pert. look fine to you?"):
				best_xyz.append(perturbed_xyz)
				remaining -= 1
				break
		
	
	if n_perts != 0:		
		fig = pylab.figure()
		fig2 = pylab.figure()
		for i in xrange(len(best_xyz)):
			xyz = best_xyz[i]
			n_p = n_perts
			while n_p > 0:
				perturbed_xyz = sample_random_rope(xyz, True)
				ax = fig.gca(projection='3d')
				ax.set_autoscale_on(False)
				ax.plot(perturbed_xyz[:,0], perturbed_xyz[:,1], perturbed_xyz[:,2], 'o')   
				fig.show()
				if i < len(bests):
					cv2.imshow("pert", best_images[bests[i]])
					kb = cv2.waitKey()
				else:
					ax = fig2.gca(projection='3d')
					ax.set_autoscale_on(False)
					ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')   
					fig2.show()
				if not yes_or_no("Does this pert. look fine to you?"):
					best_xyz.append(perturbed_xyz)
					n_p -= 1
			
	pickle.dump(best_xyz, open(osp.join(demo_files_dir, perturbation_file), 'wa'))

	return



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo_type",help="Demo type.", type=str)
	parser.add_argument("--num_base_demos", type=int, default=100)
	parser.add_argument("--num_perts", type=int, default=0)
	parser.add_argument("--load",help="Load file?", action="store_true",default=False)
	args = parser.parse_args()

	main(args.demo_type, args.num_base_demos, args.num_perts, args.load)
