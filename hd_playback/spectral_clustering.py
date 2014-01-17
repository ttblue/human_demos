import h5py
import numpy as np
#from rapprentice import registration

from joblib import Parallel, delayed
import scipy.spatial.distance as ssd
from sklearn.cluster import spectral_clustering
import cPickle as pickle
import argparse
import cv2, hd_rapprentice.cv_plot_utils as cpu
from hd_rapprentice import registration
from hd_utils import clouds

parser = argparse.ArgumentParser()
parser.add_argument("--num_clouds", type=int)
parser.add_argument("--num_clusters", type=int)
args = parser.parse_args()

num_clouds = args.num_clouds


h5file = "/home/henrylu/henry_sandbox/human_demos_data/overhand.h5"

def registration_cost(xyz0, xyz1):
	scaled_xyz0, _ = registration.unit_boxify(xyz0)
	scaled_xyz1, _ = registration.unit_boxify(xyz1)
	f,g = registration.tps_rpm_bij(scaled_xyz0, scaled_xyz1, rot_reg=1e-3, n_iter=30)
	cost = registration.tps_reg_cost(f) + registration.tps_reg_cost(g)
	return cost

def similarity_matrix(pclouds):
	print "computing tps costs"
	cost_matrix = np.zeros((0, len(pclouds)))
	for y in xrange(len(pclouds)):
		new_xyz = pclouds[y]
		costs = Parallel(n_jobs=-1,verbose=100)(delayed(registration_cost)(c, new_xyz) for c in pclouds)
		print y
		costs = np.array(costs)
		cost_matrix = np.vstack((cost_matrix, costs))
	cost_matrix = 0.5 * (cost_matrix + cost_matrix.T)
	similarity_matrix = np.exp((1.0/cost_matrix))
	return similarity_matrix


def extract_clouds(demofile, num):
	seg_num = 0
	leaf_size = 0.4
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
					if seg_num >= num:
						return keys, pclouds

def main():
	demofile = h5py.File(h5file, 'r')
	smfile = "sim_matrix_"+str(num_clouds)+".cp"
	try:
		with open(smfile, 'r') as f:
			sm = pickle.load(f)
		seg_num = 0
		keys = {}
		done = False
		for demo_name in demofile:
			if done:
				break
			if demo_name != "ar_demo":
				for seg_name in demofile[demo_name]:
					if seg_name != 'done':
						keys[seg_num] = (demo_name, seg_name)
						print demo_name, seg_name
						seg_num += 1
						if seg_num >= num_clouds:
							done = True
							break
	except IOError:
		keys, clouds = extract_clouds(demofile, num_clouds)
		sm = similarity_matrix(clouds)
		with open(smfile, 'wa') as f:
			pickle.dump(sm, f)
	print sm
	labels = spectral_clustering(sm, n_clusters = args.num_clusters, eigen_solver='arpack')
	images = [[] for i in xrange(args.num_clusters)]
	print keys
	for i in xrange(len(labels)):
		label = labels[i]
		images[label].append(np.asarray(demofile[keys[i][0]][keys[i][1]]["rgb"]))



	rows = []
	for i in xrange(args.num_clusters):
		row = cpu.tile_images(images[i], 1, len(images[i]))
		rows.append(np.asarray(row))
		cv2.imshow("clustering result", row)
		print "press any key to continue"
		cv2.waitKey()
	return
	bigimg = cpu.tile_images(rows, len(rows), 50)
	cv2.imshow("clustering result", bigimg)
	print "press any key to continue"
	cv2.waitKey()




if __name__ == "__main__":
    main()