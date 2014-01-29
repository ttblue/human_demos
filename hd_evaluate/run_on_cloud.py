## script to call the simulation on the cloud and save the results on the local machine

import cloud
from hd_utils.defaults import testing_commands_dir, testing_results_dir
import argparse
import cPickle as cp
import numpy as np
import os.path as osp
from hd_utils.colorize import colorize
import math
from hd_evaluate.call_run_test import run_sim_test


def save_results(results):
    for res in results:
        save_path = osp.join(testing_results_dir, res['state_save_fname'])
        with open(save_path, 'w') as f:
            cp.dump(res, f)


def call_on_cloud(cmd_params, core_type, num_batches, start_batch_num, end_batch_num):
    ntests = len(cmd_params)
    batch_size = int(math.ceil(ntests/(num_batches+0.0)))
    
    batch_edges = batch_size*np.array(xrange(num_batches))[start_batch_num : end_batch_num]
    
    for i in xrange(len(batch_edges)):
        if i==len(batch_edges)-1:
            cmds = cmd_params[batch_edges[i]:]
        else:
            cmds = cmd_params[batch_edges[i]:min(batch_edges[i+1], len(cmd_params))]

        jids = cloud.map(run_sim_test, cmds, _env='RSS-10k', _type=core_type)
        res  = cloud.result(jids)
        print colorize("got results for batch %d/%d "%(i, len(batch_edges)))
        save_results(res)

#### maybe make this a shell command and save to a file and use cloud.files...


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="P/NP App")
    parser.add_argument("--demo_type", type=str)
    parser.add_argument("--num_batches", help="run NUM_BATCHES chunks of tests one-by-one",type=int, default=10)
    parser.add_argument("--start_batch_num", type=int, default=1)
    parser.add_argument("--end_batch_num", type=int, default=100)
    parser.add_argument("--instance_type", type=str, default='f2')
    args = parser.parse_args()

    cmd_params_file = osp.join(testing_commands_dir, "%s_cmds.cp"%args.demo_type)
    cmd_params = cp.load(open(cmd_params_file,'r'))
    
    call_on_cloud(cmd_params, args.instance_type, args.num_batches, args.start_batch_num-1, args.end_batch_num)
    