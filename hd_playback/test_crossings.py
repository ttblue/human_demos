import numpy as np
import cv2, argparse, h5py
import os.path as osp
import subprocess
import pdb
import IPython

from hd_utils.defaults import demo_files_dir

usage = """
To compare the success of two versions of the do_task program on demos
of type DEMO_TYPE:
python test_crossings.py --demo_type=DEMO_TYPE --name=<name of file to save results in>
"""
parser = argparse.ArgumentParser(usage=usage)

parser.add_argument("--demo_type", type=str)
parser.add_argument("--single_demo", help="View and label a single demo", default=False, type=str)
parser.add_argument("--demo_start", help="Start of demo range.", default='0', type=str)
parser.add_argument("--demo_end", help="End of demo range.", default='100', type=str)
parser.add_argument("--name", help="name to save file", default='test_results.txt', type=str)


def main():
    args = parser.parse_args()
    hdf = h5py.File(osp.join(demo_files_dir, args.demo_type, args.demo_type + '.h5'), 'r')
    crossings_failures = []
    baseline_failures = []
    for i in range(int(args.demo_start), int(args.demo_end)):
        demo = hdf.keys()[i]
        fake_data_demo = "--fake_data_demo="+demo
        call1 = "python do_task_merge.py --test_success --no_display --demo_type " + args.demo_type + " overhand_recovery overhand_fix --fake_data_demo=" + demo
        call1 += " --fake_data_segment=seg00 --step=100 --force_points --weight_pts --extra_settle=12 --shiftx=0 --choice_file="+args.name+"choice"
        call2 = "python do_task_merge.py --test_success --no_display --demo_type " + args.demo_type + " overhand_recovery overhand_fix --fake_data_demo=" + demo
        call2 += " --fake_data_segment=seg00 --step=100 --force_points --weight_pts --extra_settle=12 --shiftx=0 --choice_file="+args.name+"choice --init_perturb=1"

        savefile = open(osp.join("test_results", args.name), 'a')

        if i == int(args.demo_start):
            savefile.write(call1 + "\n")
            savefile.write(call2 + "\n")

        print "starting demo", demo

        try:
            out = subprocess.check_output(call1.split())
        except Exception as exc:
            err_msg = exc.output.split("\n")[-2]
            baseline_failures.append(demo)
            savefile.write("Unperturbed failure: " + demo + ": " + err_msg + "\n")
            print "Unperturbed failure: " + demo + ": " + err_msg + "\n"
            savefile.flush()

        print "finished first call"

        try:
            out = subprocess.check_output(call2.split())
        except Exception as exc:
            err_msg = exc.output.split("\n")[-2]
            crossings_failures.append(demo)
            savefile.write(" Perturbed failure: " + demo + ": " + err_msg + "\n")
            print " Perturbed failure: " + demo + ": " + err_msg + "\n"
        savefile.close()

        print "finished demo", demo

    print " Perturbed failures:", crossings_failures, "\nUnperturbed failures:", baseline_failures
    savefile = open(osp.join("test_results", args.name), 'a')

    savefile.write("Perturbed failures: " + str(len(crossings_failures)) + "\n")
    for item in crossings_failures:
        savefile.write("   " + str(item)+"\n")

    savefile.write("Unperturbed failures: " + str(len(baseline_failures)) + "\n")
    for item in baseline_failures:
        savefile.write("   " + str(item)+"\n")

    savefile.close()
    return crossings_failures, baseline_failures 

if __name__=="__main__":
    main()


