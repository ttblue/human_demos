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
import clouds 

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--perturb_fname", help="File saved perturb point clouds")
parser.add_argument("--perturb_num", help="Number of random perturbations", type=int, default=1)
parser.add_argument("--max_perturb_attempt", help="Number of maximum attempts for perturbation", type=int, default=5)
parser.add_argument("--overwrite", action="store_true")

args = parser.parse_args()





demotype_dir = osp.join(demo_files_dir, args.demo_type)
h5file = osp.join(demotype_dir, args.demo_type+".h5")


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
    
num_h5_perturbs = len(perturb_demofile.keys())
    

demofile = h5py.File(h5file, 'r')

fig = pylab.figure()

demo_type_dir = osp.join(demo_files_dir, args.demo_type)
demo_master_file = osp.join(demo_type_dir, master_name)

with open(demo_master_file, 'r') as fh:
    demos_info = yaml.load(fh)

for demo in demos_info["demos"]:
    demo_name = demo["demo_name"]
    
    if demo_name in perturb_demofile.keys():
        demo_group = perturb_demofile[demo_name]
    else:
        demo_group = perturb_demofile.create_group(demo_name)
        
    n_perturb_existed = len(demo_group.keys()) # number of perturbations
    
    if 'hitch_pos' in demofile[demo_name]['seg00'].keys():
        has_hitch = True
    else:
        has_hitch = False
    
    xyz = demofile[demo_name]['seg00']["cloud_xyz"]
    
    ax = fig.gca(projection='3d')
    ax.set_autoscale_on(False)
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')   
    fig.show()

    if yes_or_no("Do you want to add this original demo?"):
        perturb_name = str(n_perturb_existed)
        perturb_group = demo_group.create_group(perturb_name)
        perturb_group['cloud_xyz'] = xyz
        
        if has_hitch:
            perturb_group['hitch_pos'] = demofile[demo_name]['seg00']['hitch_pos']
        
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
        
        if has_hitch:
            new_rope_xyz = sample_random_rope(demofile[demo_name]['seg00']['object'])
            new_rope_xyz = clouds.downsample(new_rope_xyz, .01)
            new_xyz = np.r_[new_rope_xyz, demofile[demo_name]['seg00']['hitch']]
        else:
            new_xyz = sample_random_rope(xyz, True)
            new_xyz = clouds.downsample(new_xyz, .01)

        if yes_or_no("Are you happy with this perturbation?"):
            perturb_name = str(n_perturb_existed)
            perturb_group = demo_group.create_group(perturb_name)
            perturb_group['cloud_xyz'] = new_xyz
            print "add perturb demo %d"%(n_perturb_existed)
            n_perturbed += 1
            n_perturb_existed += 1
      
        n_perturbed_attempt += 1      

    fig.clf()
    
    if yes_or_no("Stop perturbation?"):
        break        
