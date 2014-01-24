import numpy as np
#from rapprentice import registration
import cPickle as pickle
import os.path as osp

from hd_utils.defaults import hd_data_dir, similarity_costs_dir, matrix_file

def extract_init(demo_type, save=False):
	with open(osp.join(hd_data_dir, similarity_costs_dir, demo_type+'.costs')) as f:
		dic = pickle.load(f)
	costs = dic['costs']
	cost_matrix = np.zeros((500, 500))
	
	w_tps = 1.0
	w_traj = 0.5
	w_traj_f = 0.5
	
	for i in xrange(500):
		demo_name = 'd' + str(i+1) + 's0'
		print demo_name
		for j in xrange(i, 500):
			compare_name = 'd' + str(j+1) + 's0'
			elements = costs[demo_name][compare_name]

			tps = elements['tps']
			traj= elements['traj']['l'] + elements['traj']['r']
			traj_f = 0
			if 'traj_f' in elements:
				traj_f = elements['traj_f']['l'] + elements['traj_f']['r']
			cost_matrix[i,j] = w_tps*tps + w_traj*traj + w_traj_f*traj_f  

	cost_matrix = cost_matrix + cost_matrix.T - np.diag(np.diag(cost_matrix))
	similarity_matrix = np.exp(-cost_matrix)
	if save:
		with open(osp.join(hd_data_dir, similarity_costs_dir, matrix_file%demo_type), 'w') as fh: pickle.dump(similarity_matrix, fh)
	else:
		return similarity_matrix
		

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--demo_type",help="Demo type.", type=str)
	parser.add_argument("--save",help="Save file?", action="store_true",default=False)
	args = parser.parse_args()
	
	extract_init(args.demo_type, args.save)