import os, os.path as osp
import importlib
import numpy as np
import argparse
from defaults import demo_files_dir, demo_names, master_name
from extraction_utils import get_videos
from mpl_toolkits.mplot3d import axes3d
import pylab, yaml, h5py
from hd_utils.clouds_utils import sample_random_rope
from hd_utils.yes_or_no import yes_or_no
from hd_utils.extraction_utils import get_video_frames

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
parser.add_argument("--perturb_fname", help="File saved perturb point clouds", default="perturb")
parser.add_argument("--perturb_num", help="Number of random perturbations", type=int, default=1)
parser.add_argument("--max_perturb_attempt", help="Number of maximum attempts for perturbation", type=int, default=5)
parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()


cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)


demotype_dir = osp.join(demo_files_dir, args.demo_type)
perturb_h5file = osp.join(demotype_dir, args.perturb_fname+".h5")


if args.overwrite:
    if osp.exists(perturb_h5file):
        os.unlink(perturb_h5file)
    perturb_demofile = h5py.File(perturb_h5file)
else:
    if osp.exists(perturb_h5file):
        perturb_demofile = h5py.File(perturb_h5file, "r+")
    else:
        perturb_demofile = h5py.File(perturb_h5file)


task_dir = osp.join(demo_files_dir, args.demo_type)
task_file = osp.join(task_dir, master_name)
with open(task_file, "r") as fh: task_info = yaml.load(fh)

demos_info = task_info['demos']



fig = pylab.figure()

for demo_info in demos_info:

    demo_name = demo_info['demo_name']
    print demo_name
    demo_dir = osp.join(task_dir, demo_name)
    rgbd_dir = osp.join(demo_dir, demo_names.video_dir%(1))
    annotation_file = osp.join(demo_dir,"ann.yaml")
    with open(annotation_file, "r") as fh: annotations = yaml.load(fh)
    
    look_stamps = [seg_info['look'] for seg_info in annotations]
    rgb_imgs, depth_imgs= get_video_frames(rgbd_dir, look_stamps)
    
    if demo_name in perturb_demofile.keys():
        demo_group = perturb_demofile[demo_name]
    else:
        demo_group = perturb_demofile.create_group(demo_name)
        demo_group['rgb'] = rgb_imgs[0]
        demo_group['depth'] = depth_imgs[0]
        
    if 'perturbs' in demo_group.keys():
        perturb_group = demo_group['perturbs']
    else:
        perturb_group = demo_group.create_group('perturbs')
    
    n_perturb_existed = len(perturb_group.keys())
    
    xyz = cloud_proc_func(rgb_imgs[0], depth_imgs[0], np.eye(4)) 
    
    ax = fig.gca(projection='3d')
    ax.set_autoscale_on(False)
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')   
    fig.show()
    
    if yes_or_no("Do you want to add this original demo?"):
        perturb_group[str(n_perturb_existed)] = xyz
        print "add perturb demo %d"%(n_perturb_existed)
        n_perturb_existed += 1
    else:
        pass

    
    if yes_or_no("Do you want to skip perturbing this demo?"):
        print "skip perturbing demo"
        fig.clf()
        continue
    


    # start actual perturbation    
    n_perturbed = 0
    n_perturbed_attempt = 0

    
    while n_perturbed <= args.perturb_num and n_perturbed_attempt < args.max_perturb_attempt:    
        new_xyz = sample_random_rope(xyz, True)


        if yes_or_no("Are you happy with this perturbation?"):
            perturb_group[str(n_perturb_existed)] = new_xyz
            print "add perturb demo %d"%(n_perturb_existed)
            n_perturbed += 1
            n_perturb_existed += 1
      
        n_perturbed_attempt += 1      

    fig.clf()
    
    if yes_or_no("Stop perturbation?"):
        break
    
    
    
    
    
    
    
    
    
    
    
    
    







