## script to call the simulation on the cloud and save the results on the local machine
import cloud
from hd_utils.defaults import testing_commands_dir, testing_results_dir
import argparse
import cPickle as cp
import numpy as np
import os.path as osp, os
from hd_utils.colorize import colorize
import math
from hd_evaluate.call_run_test import run_sim_test


def save_results(results):
    for res in results:
        save_path = osp.join(testing_results_dir, res['state_save_fname'])

        ## make directories to save results in:
        save_dir = osp.dirname(save_path)
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

        with open(save_path, 'w') as f:
        #with open(osp.join('/home/ankush', osp.basename(save_path)), 'w') as f:
            cp.dump(res, f)


def call_on_cloud(cmd_params, core_type, num_batches, start_batch_num, end_batch_num):
    ntests = len(cmd_params)
    batch_size = int(math.ceil(ntests/(num_batches+0.0)))

    batch_edges = batch_size*np.array(xrange(num_batches))[start_batch_num : end_batch_num]
    print batch_edges
    for i in xrange(len(batch_edges)):
        if i==len(batch_edges)-1:
            cmds = cmd_params[batch_edges[i]:]
        else:
            cmds = cmd_params[batch_edges[i]:min(batch_edges[i+1], len(cmd_params))]
        print colorize("calling on cloud..", "yellow", True)
        try:
            jids = cloud.map(run_sim_test, cmds, _vol='rss_dat', _env='RSS3', _type=core_type)
            res  = cloud.result(jids)
            print colorize("got results for batch %d/%d "%(i, len(batch_edges)), "green", True)
            save_results(res)
        except Exception as e:
            print "Found exception %s. Not saving data for this demo."%e

#### maybe make this a shell command and save to a file and use cloud.files...
if __name__=='__main__':
    parser = argparse.ArgumentParser(description="P/NP App")
    parser.add_argument("--demo_type", type=str)
    parser.add_argument("--num_batches", help="run NUM_BATCHES chunks of tests one-by-one",type=int, default=8)
    parser.add_argument("--start_batch_num", type=int, default=1)
    parser.add_argument("--end_batch_num", type=int, default=8)
    parser.add_argument("--instance_type", type=str, default='c2')
    args = parser.parse_args()

    cmd_params_file = osp.join(testing_commands_dir, "%s_cmds.cp"%args.demo_type)
    cmd_params = cp.load(open(cmd_params_file,'r'))
    
    call_on_cloud(cmd_params, args.instance_type, args.num_batches, args.start_batch_num-1, args.end_batch_num)
    