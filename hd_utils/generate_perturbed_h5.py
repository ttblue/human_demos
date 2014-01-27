import os, os.path as osp
import numpy as np
import argparse
from defaults import demo_files_dir
import pylab, h5py
from hd_utils.clouds_utils import sample_random_rope
from hd_utils.yes_or_no import yes_or_no
import clouds 
from mayavi import mlab


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
    perturb_demofile = h5py.File(perturb_h5file, "w")
else:
    if osp.exists(perturb_h5file):
        perturb_demofile = h5py.File(perturb_h5file, "r+")
    else:
        perturb_demofile = h5py.File(perturb_h5file, "w")
    
    

demofile = h5py.File(h5file, 'r')

demo_type_dir = osp.join(demo_files_dir, args.demo_type)


for demo in sorted(demofile.keys()):
    demo_name = demo
    
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
    xyz = np.squeeze(xyz)
    
    mlab.figure(0)
    mlab.clf()
    mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2], color=(1,0,0), scale_factor=.005)

    if yes_or_no("Do you want to add this original demo?"):
        perturb_name = str(n_perturb_existed)
        perturb_group = demo_group.create_group(perturb_name)
        
        perturb_group['cloud_xyz'] = xyz
        
        if has_hitch:
            hitch_pos = demofile[demo_name]['seg00']['hitch_pos']
            perturb_group['hitch_pos'] = np.squeeze(hitch_pos) 
        
        print "add perturb demo %d"%(n_perturb_existed)
        n_perturb_existed += 1
    else:
        pass       
    
    if yes_or_no("Do you want to skip perturbing this demo?"):
        print "skip perturbing demo"
        continue             
        
    # start actual perturbation    
    n_perturbed = 0
    n_perturbed_attempt = 0
    
        
    while n_perturbed <= args.perturb_num and n_perturbed_attempt < args.max_perturb_attempt:
        
        if has_hitch:
            new_object_xyz = sample_random_rope(demofile[demo_name]['seg00']['object'], True)
            new_object_xyz = clouds.downsample(new_object_xyz, .01)
            new_xyz = np.r_[new_object_xyz, demofile[demo_name]['seg00']['hitch']]
        else:
            new_xyz = sample_random_rope(xyz, True)
            new_xyz = clouds.downsample(new_xyz, .01)

        mlab.figure(0)
        mlab.clf()
        mlab.points3d(new_xyz[:,0], new_xyz[:,1], new_xyz[:,2], color=(1,0,0), scale_factor=.005)


        if yes_or_no("Are you happy with this perturbation?"):
            perturb_name = str(n_perturb_existed)
            perturb_group = demo_group.create_group(perturb_name)
            perturb_group['cloud_xyz'] = new_xyz
            print "add perturb demo %d"%(n_perturb_existed)
            n_perturbed += 1
            n_perturb_existed += 1
      
        n_perturbed_attempt += 1      

    
    if yes_or_no("Stop perturbation?"):
        break        
