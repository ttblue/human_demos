import os, os.path as osp
import argparse

from hd_utils.defaults import demo_names, demo_files_dir, latest_demo_name
from hd_utils.colorize import redprint, yellowprint
from hd_utils.yes_or_no import yes_or_no
from hd_record.delete_demo import delete_demo

from visualize_demo import view_hydra_demo_on_rviz, view_demo_on_rviz, view_tracking_on_rviz


"""
Code to visualize demos in sequence, quickly.
"""
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_type",help="Type of demonstration")
    parser.add_argument("--demo_name",help="Name of demo", default='', type=str)
    parser.add_argument("--freq",help="Frequency of sampling.", default=30.0, type=float)
    parser.add_argument("--speed",help="Speed of demo.", default=1.0, type=float)
    parser.add_argument("--hydra_only",help="Use .traj file (kalman f/s data)", action='store_true',default=False)
    parser.add_argument("--use_traj",help="Use .traj file (kalman f/s data)", action='store_true',default=False)
    parser.add_argument("--main",help="If not using .traj file, which sensor is main?", default='h', type=str)
    parser.add_argument("--use_smoother",help="If using .traj file, filter or smoother?", action='store_true',default=False)
    parser.add_argument("--prompt",help="Prompt for each step.", action='store_true', default=False)
    parser.add_argument("--first",help="Lower bound for demo range.", default=0, type=int)
    parser.add_argument("--last",help="Upper bound for demo range.", default=-1, type=int)
    parser.add_argument("--prompt_delete",help="Prompt to delete after each demo.", action='store_true', default=False)
    parser.add_argument("--verbose", help="verbose", action='store_true', default=False)

    args = parser.parse_args()


    if args.demo_name != '':
        if args.use_traj:
            view_tracking_on_rviz(demo_type=args.demo_type, demo_name=args.demo_name,
                                  freq=args.freq, speed=args.speed, 
                                  use_smoother=args.use_smoother, prompt=args.prompt, verbose=args.verbose)
        else:
            if args.hydra_only:
                view_hydra_demo_on_rviz(demo_type=args.demo_type, demo_name=args.demo_name, 
                                        freq=args.freq, speed=args.speed, prompt=args.prompt, verbose=args.verbose)
            else:
                view_demo_on_rviz(demo_type=args.demo_type, demo_name=args.demo_name, 
                                  freq=args.freq, speed=args.speed, 
                                  main=args.main, prompt=args.prompt, verbose=args.verbose)
    
        if args.prompt_delete and yes_or_no('Do you want to delete %s?'%args.demo_name):
            delete_demo(args.demo_type, args.demo_name)
    else:
        first = args.first
        last = args.last
        
        demo_type_dir = osp.join(demo_files_dir, args.demo_type)
        
        if args.last == -1:
            latest_demo_file = osp.join(demo_type_dir, latest_demo_name)
            if osp.isfile(latest_demo_file):
                with open(latest_demo_file,'r') as fh:
                    last = int(fh.read()) + 1
            else:
                redprint("No demos!")

        
        for i in xrange (first, last+1):
            demo_name = demo_names.base_name%i
            demo_dir = osp.join(demo_files_dir, args.demo_type, demo_name)
            if osp.exists(demo_dir):
                if args.prompt:
                    raw_input('Hit enter for %s.'%demo_name)
                yellowprint("Visualizing: %s"%demo_name)
                if args.use_traj:
                    view_tracking_on_rviz(demo_type=args.demo_type, demo_name=demo_name,
                                          freq=args.freq, speed=args.speed, 
                                          use_smoother=args.use_smoother, prompt=args.prompt, verbose=args.verbose)
                else:
                    if args.hydra_only:
                        view_hydra_demo_on_rviz(demo_type=args.demo_type, demo_name=demo_name, 
                                                freq=args.freq, speed=args.speed, prompt=args.prompt, verbose=args.verbose)
                    else:
                        view_demo_on_rviz(demo_type=args.demo_type, demo_name=demo_name, 
                                          freq=args.freq, speed=args.speed, 
                                          main=args.main, prompt=args.prompt, verbose=args.verbose)

                if args.prompt_delete and yes_or_no('Do you want to delete %s?'%demo_name):
                    delete_demo(args.demo_type, demo_name)