import numpy as np
import cv2, argparse, h5py
import os.path as osp
import subprocess
import pdb

from hd_utils.defaults import demo_files_dir

usage = """
To compare the success of two versions of the do_task program on demos
of type DEMO_TYPE:
python test_crossings.py --demo_type=DEMO_TYPE
"""
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--single_demo", help="View and label a single demo", default=False, type=str)
parser.add_argument("--demo_start", help="Start of demo range.", default='0', type=str)
parser.add_argument("--demo_end", help="End of demo range.", default='100', type=str)


def main():
    args = parser.parse_args()
    subprocess_call = ["python", "do_task_floating.py"]
    subprocess_call.append("--demo_type="+args.demo_type)
    subprocess_call.append("--use_ar_init")
    subprocess_call.append("--select=auto")
    h5filename = osp.join(demo_files_dir, args.demo_type, args.demo_type + '.h5')
    hdf = h5py.File(h5filename, 'r+')
    crossings_failures = []
    baseline_failures = []
    for i in range(int(args.demo_start), int(args.demo_end)):
            #import pdb
            #pdb.set_trace()
        demo = hdf.keys()[i]
        fake_data_demo = "--fake_data_demo="+demo
        non_cross_call = "python do_task_floating.py --demo_type="+args.demo_type+" --fake_data_demo="+demo+" --fake_data_segment=seg00 --use_ar_init --select=auto --test_success --step=100"
        cross_call = "python do_task_floating.py --demo_type="+args.demo_type+" --fake_data_demo="+demo+" --fake_data_segment=seg00 --use_ar_init --select=auto --use_crossings --test_success --step=100"
        try:
            ncs = subprocess.call(non_cross_call.split())
        except:
            ncs = 1
        savefile = open("test_results.txt", 'a')
        if ncs != 0:
            baseline_failures.append(demo)
            savefile.write("Baseline failure: " + demo +"\n")
            print "baseline version failed"
        try:
            cs = subprocess.call(cross_call.split())
        except:
            cs = 1
        if cs != 0:
            crossings_failures.append(demo)
            savefile.write("Crossings failure: " + demo +"\n")
            print "crossings version failed"
        savefile.close()
    print "crossings failures:", crossings_failures, "\nbaseline failures:", baseline_failures
    savefile = open("test_results.txt", 'a')
    savefile.write("Crossings_failures:\n")
    for item in crossings_failures:
        savefile.write(str(item)+"\n")
    savefile.write("Baseline failures:\n")
    for item in baseline_failures:
        savefile.write(str(item)+"\n")
    savefile.close()
    return crossings_failures, baseline_failures
 

if __name__=="__main__":
    main()
