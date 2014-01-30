"""
Script to split the indices etc. for parallel testing on the cloud.

This generates library of demos with different number of demos, initial state configuraion files
and the file of the command line arguments to be executed in parallel on the cloud.
"""
import cPickle as cp
import numpy as np
import math, sys
import os, os.path as osp
import h5py
from hd_utils.colorize import colorize

from hd_utils.defaults import data_dir, demo_files_dir, testing_commands_dir


init_perturbation_map = {'overhand_noik'  : 'overhand_noik',
			 'overhand_training': 'overhand_noik',
			 'overhand_post_training': 'overhand_noik',
			 'figure_eight_noik':'overhand_noik',
			 'figure_eight_training':'overhand_noik',
			 'figure_eight_post_training':'overhand_noik',
                         'square_knot'    : 'overhand_noik',
                         'overhand'       : 'overhand140',
                         'figure_eight'   : 'overhand140',
                         'slip_knot'      : 'overhand140',
                         'double_overhand': 'overhand140',
                         'clove_hitch'    : 'pile_hitch160',
                         'cow_hitch'      : 'pile_hitch160',
                         'pile_hitch'     : 'pile_hitch160'}

def get_rope_lengths(demo_type):
    if 'noik' in demo_type or 'training' in demo_type:
        return [140]
    elif 'hitch' in demo_type or demo_type in ['double_overhand', 'square_knot']:
        return [140, 160, 180]
    else:
        return [120,140,160]

def sample_rope_scaling(rope_lengths):
    return np.round(np.random.uniform(rope_lengths[0], rope_lengths[2])/(rope_lengths[1]+0.0), 2)


perturbations_dir = osp.join(data_dir, 'init_state_perturbations') 
perturbations_dir = "/home/ankush/sandbox444/human_demos/hd_evaluate/init_state_perturbations"

#demo_files_dir    = "/home/ankush/sandbox444/human_demos/hd_evaluate/sample_dat"
#data_dir          = "/home/ankush/sandbox444/human_demos/hd_evaluate/sample_dat"


def split_pertubations_into_two(perturb_fname, sizes = [50, 25, 4]):
    perturb_dat  = h5py.File(perturb_fname, 'r')
    num_perturbs = len(perturb_dat.keys())

    assert num_perturbs >= 90, colorize("not enough demos in the perturbation file : %s"%perturb_fname)

    subset1, subset2 = {}, {}
    for i in xrange(len(sizes)):
        if i==0:
            random_indices = range(num_perturbs)
            np.random.shuffle(random_indices)
            subset1[sizes[i]] = random_indices[:sizes[i]]
            subset2[sizes[i]] = random_indices[sizes[i]:]
        else:
            subset1[sizes[i]] = [subset1[sizes[i-1]][x] for x in np.random.choice(range(len(subset1[sizes[i-1]])), sizes[i], replace=False)]
            subset2[sizes[i]] = [subset2[sizes[i-1]][x] for x in np.random.choice(range(len(subset2[sizes[i-1]])), sizes[i], replace=False)] 

    perturb_dat.close()
    return subset1, subset2


def generate_testing_h5_files(demo_type, subset1, subset2, rope_lengths=[120,140,160]):
    """
    demo_type : the type of the knot : {overhand, square-knot, ...}
    subset1, subset2 : the splitting indices as returned by split_perturbations_into_two above
    """
    def copy_to_subset(parent_h5, child_h5, subset):
        for idx in subset:
            if idx >= len(parent_h5.keys()):
                continue
            pkey = parent_h5.keys()[idx]
            parent_h5.copy(pkey, child_h5)

    demo_h5_fnames = {l : osp.join(demo_files_dir, "%s%d"%(demo_type, l),  "%s%d.h5"%(demo_type,l)) for l in rope_lengths}
    demo_h5_files  = {}

    for h5_fname in demo_h5_fnames.values():
        if not osp.exists(h5_fname):
            print colorize(" %s : does not exist. Exiting..."%h5_fname, "red", True)
            sys.exit(-1)

    for l in rope_lengths:
        demo_h5_files[l] = h5py.File(demo_h5_fnames[l], 'r')

    out_dir = osp.join(data_dir, "testing_h5s", demo_type)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    
    num_lengths  = len(rope_lengths)
    subset_sizes = np.sort(np.array(subset1.keys()))

    cumm_subset_sizes = num_lengths * subset_sizes.astype(int)
    for i,cumm_size in enumerate(cumm_subset_sizes):
        for l in rope_lengths:
            subset1_h5_name, subset2_h5_name = [osp.join(out_dir, "size%d_set%d_len%d.h5"%(cumm_size, x, l)) for x in [1,2]]
            
            subset1_h5_file = h5py.File(subset1_h5_name, 'w')
            subset2_h5_file = h5py.File(subset2_h5_name, 'w')

            copy_to_subset(demo_h5_files[l], subset1_h5_file, subset1[np.sort(subset1.keys())[i]])            
            copy_to_subset(demo_h5_files[l], subset2_h5_file, subset2[np.sort(subset2.keys())[i]])

            subset1_h5_file.close()
            subset2_h5_file.close()
            print colorize("saved : %s"%subset1_h5_name, "green", True)            
            print colorize("saved : %s"%subset2_h5_name, "green", True)

    for h5file in demo_h5_files.values():
        h5file.close()

def generate_testing_h5_file_ik(demo_type, subset1, subset2):
    """
    demo_type : the type of the knot : {overhand, square-knot, ...}
    subset1, subset2 : the splitting indices as returned by split_perturbations_into_two above
    """
    def copy_to_subset(parent_h5, child_h5, subset):
        for idx in subset:
            if idx >= len(parent_h5.keys()):
                continue
            pkey = parent_h5.keys()[idx]
            parent_h5.copy(pkey, child_h5)

    rope_lengths = [140]
    demo_h5_fnames = {l : osp.join(demo_files_dir, "%s"%(demo_type),  "%s.h5"%(demo_type)) for l in [140]}
    demo_h5_files  = {}

    for h5_fname in demo_h5_fnames.values():
        if not osp.exists(h5_fname):
            print colorize(" %s : does not exist. Exiting..."%h5_fname, "red", True)
            sys.exit(-1)

    for l in rope_lengths:
        demo_h5_files[l] = h5py.File(demo_h5_fnames[l], 'r')

    out_dir = osp.join(data_dir, "testing_h5s", demo_type)
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    
    num_lengths  = len(rope_lengths)
    subset_sizes = np.array([np.sort(np.array(subset1.keys()))])

    cumm_subset_sizes = num_lengths * subset_sizes.astype(int)
    for i,cumm_size in enumerate(cumm_subset_sizes):
        for l in rope_lengths:
            subset1_h5_name, subset2_h5_name = [osp.join(out_dir, "size%d_set%d.h5"%(cumm_size, x)) for x in [1,2]]
            
            subset1_h5_file = h5py.File(subset1_h5_name, 'w')
            subset2_h5_file = h5py.File(subset2_h5_name, 'w')

            copy_to_subset(demo_h5_files[l], subset1_h5_file, subset1[np.sort(subset1.keys())[i]])            
            copy_to_subset(demo_h5_files[l], subset2_h5_file, subset2[np.sort(subset2.keys())[i]])

            subset1_h5_file.close()
            subset2_h5_file.close()
            print colorize("saved : %s"%subset1_h5_name, "green", True)            
            print colorize("saved : %s"%subset2_h5_name, "green", True)

    for h5file in demo_h5_files.values():
        h5file.close()



def generate_test_cmdline_params(demo_type, generate_h5s=False, sizes = [50, 25, 4], use_ik=False):
    """
    saves ~6000 sets of command line arguments.
    """
    rope_lengths     = get_rope_lengths(demo_type)
    perturb_fname    = init_perturbation_map[demo_type] + '_perturb.h5'
    perturb_fname    = osp.join(perturbations_dir, perturb_fname)
    perturb_file     = h5py.File(perturb_fname, "r") 
    subsets = split_pertubations_into_two(perturb_fname, sizes)

    if generate_h5s:
        if use_ik:
            generate_testing_h5_file_ik(demo_type, subsets[0], subsets[1])
        else:
            generate_testing_h5_files(demo_type, subsets[0], subsets[1], rope_lengths)

    cmdline_params = []
    cmdline_dir = testing_commands_dir
    if not osp.exists(cmdline_dir):
        os.mkdir(cmdline_dir)
    
    num_demos = len(rope_lengths) * np.sort(np.array(subsets[0].keys()).astype(int))
    
    num_perts = 5
    print num_perts
    
    for idx_ndemos, ndemos in enumerate(num_demos): ## x3
        for demo_set in [0,1]: ## x2
            for init_set in [0,1]: ## x2
                init_subset = subsets[init_set]
                for init_demo_idx in init_subset[np.max(init_subset.keys())]:      ## x50
                    init_config_data = perturb_file[perturb_file.keys()[init_demo_idx]]
                    init_demo_name   = perturb_file.keys()[init_demo_idx]
                    for init_perturb_name in init_config_data.keys()[0:num_perts]:                 ## x10
                        demo_data_h5_prefix = "size%d_set%d"%(ndemos, demo_set+1)
                        init_state_h5       = init_perturbation_map[demo_type]
			if len(rope_lengths) > 1: 
	                        rope_scaling_factor = sample_rope_scaling(rope_lengths)
			else:
				rope_scaling_factor = 1.0
                        results_fname       = osp.join(demo_type, "%d_demos"%ndemos, "initset%d_demoset%d"%(init_set+1, demo_set+1), "perturb_%s_%s.cp"%(init_demo_name, init_perturb_name))
                        cmdline_params.append([ndemos,
                                               demo_type,
                                               demo_data_h5_prefix,
                                               init_state_h5,
                                               str(init_demo_name),
                                               str(init_perturb_name),
                                               rope_scaling_factor,
                                               str(results_fname)])

    cmdline_file = osp.join(cmdline_dir, "%s_cmds.cp"%demo_type)
    cp.dump(cmdline_params, open(cmdline_file, 'w'))
    print colorize("wrote : %s"%cmdline_file, "yellow", True)
