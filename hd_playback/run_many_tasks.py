import subprocess, signal
import argparse
import h5py
parser = argparse.ArgumentParser(usage="")

parser.add_argument("--init_state_h5", type=str)
parser.add_argument("--demo_type", type=str)

args = parser.parse_args()

init_state_h5file = h5py.File(args.init_state_h5+".h5", "r")

for demo_name in init_state_h5file.keys():
    for perturb_name in init_state_h5file[demo_name].keys():
        test_command = '''python hd_playback/run_test.py --init_state_h5=%s --demo_type=%s --demo_name=%s --perturb_name=%s --execution=0 --animation=1 --use_ar_init --simulation=1 --select=auto --max_steps_before_failure=8'''
        test_handle = subprocess.Popen(test_command%(args.init_state_h5, args.demo_type, demo_name, perturb_name), shell=True)
        test_handle.wait()
