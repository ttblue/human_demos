import subprocess, signal
import argparse
import h5py
import os, os.path as osp
from hd_utils.defaults import demo_files_dir
import glob, cPickle as cp
from multiprocessing import Pool
from time import sleep
from hd_utils.colorize import colorize


def collect_test_results(cp_dir, output_cp_fname):   
    output_data = {}
    for fname in glob.glob(osp.join(cp_dir, "*.cp")):
        with open(fname, "r") as f:
            data = cp.load(f)
            demo_name = data['demo_name']
            perturb_name = data['perturb_name']
            states = data['seg_info']
            
            output_data[(demo_name, perturb_name)] = states            

    with open(output_cp_fname, "w") as output_f:
        cp.dump(output_data, output_f)   

def initial_setup(args):
    init_state_h5file = h5py.File(args.init_state_h5+".h5", "r")
    state_dir = osp.join(demo_files_dir, args.demo_type, osp.splitext(osp.basename(args.init_state_h5))[0])
    if osp.exists(state_dir):
        os.rmdir(state_dir)
    os.mkdir(state_dir)

    test_commands = []
    for demo_name in init_state_h5file.keys():
        for perturb_name in init_state_h5file[demo_name].keys():
            test_command = '''python hd_playback/run_test.py --init_state_h5=%s --demo_type=%s --demo_name=%s --perturb_name=%s --execution=0 --animation=1 --use_ar_init --simulation=1 --select=auto --max_steps_before_failure=8 --state_saver_dir=%s'''
            test_commands.append((test_command, demo_name, perturb_name))
    return state_dir, test_commands

def make_shell_calls(cmd_params):
    test_command, demo_name, perturb_name = cmd_params
    shell_cmd = test_command%(args.init_state_h5, args.demo_type, demo_name, perturb_name, state_dir)
    print colorize("Calling:\n\t"+shell_cmd, "green", True)
    return os.system(shell_cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="")
    parser.add_argument("--init_state_h5", type=str)
    parser.add_argument("--demo_type", type=str)
    parser.add_argument("--num_processes", type=int)
    args = parser.parse_args()
    
    state_dir, test_commands = initial_setup(args)

    pool =  Pool(processes=args.num_processes)
    res  = pool.map_async(make_shell_calls, test_commands)
    res.wait()

    output_h5_fname = osp.join(demo_files_dir, args.demo_type, osp.splitext(osp.basename(args.init_state_h5))[0] + "_test_result.cp")
    print colorize("collecting results...", "green", True)
    collect_test_results(state_dir, output_h5_fname)
