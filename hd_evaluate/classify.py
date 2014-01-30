import argparse
import cPickle as cp
import numpy as np
import os.path as osp
import os
import subprocess


from hd_utils.utils import find_recursive
from hd_utils.defaults import testing_results_dir
from hd_utils.colorize import colorize

def call_pnp_app(demo_type):
    demo_testing_dir = osp.join(testing_results_dir, demo_type)
    sub    = ['12_demos', '75_demos', '150_demos']
    subsub = ['initset1_demoset1', 'initset1_demoset2', 'initset2_demoset1', 'initset2_demoset2']
    
    results = {}
    
    for s in sub:
        res_ss = {}
        for ss in subsub:
            snapshot_path = osp.join(demo_testing_dir, s, ss, 'snapshots')
            save_path     = '/tmp/%d.cp'%np.random.randint(10000000000)
            print colorize('doing : %s/%s'%(s,ss), "green", True)            
            subprocess.call('python /home/ankush/sandbox444/human_demos/hd_evaluate/pnpApp/pnpApp.py --snapshot_path=%s --save_path=%s'%(snapshot_path, save_path), shell=True)
            stats = cp.load(open(save_path,'r'))
            res_ss[ss] = stats
        results[s] = res_ss
    
    stats_file = osp.join(demo_testing_dir, 'stats.cp')
    
    with open(stats_file,'w') as f: 
        cp.dump(results, f)
    print colorize("all done", "red", True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify App")
    parser.add_argument("--demo_type", type=str)
    args = parser.parse_args()
    call_pnp_app(args.demo_type)
    