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



def main():
	demo_type = 'overhand'
	demofile = h5py.File(osp.join(demo_files_dir, demo_type, demo_type+'.h5'), 'r')
	f = open('sim_matrix1500.cp')
	dic = pickle.load(f)
	costs = dic['costs']
	cost_matrix = np.zeros((500, 500))
	for i in xrange(500):
		demo_name = 'd' + str(i+1) + 's0'
		for j in xrange(i, 500):
			compare_name = 'd' + str(j+1) + 's0'
			print demo_name, compare_name
			elements = costs[demo_name][compare_name]
			cost_matrix[i,j] = elements['tps'] + elements['traj']['l'] + elements['traj']['r'] + elements['traj_f']['l'] + elements['traj_f']['r']
			
			

	cost_matrix = cost_matrix + cost_matrix.T - np.diag(np.diag(cost_matrix))
	similarity_matrix = np.exp(-cost_matrix)
	with open('initial_similarity_matrix.cp', 'w') as fh: pickle.dump(similarity_matrix, fh)
		

if __name__ == "__main__":
        main()
