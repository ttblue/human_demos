import os, os.path as osp
import importlib
import numpy as np
import argparse
from hd_utils.defaults import demo_files_dir, demo_names, master_name
from hd_utils.extraction_utils import get_videos
from mpl_toolkits.mplot3d import axes3d
import pylab, yaml, h5py
from hd_utils.clouds_utils import sample_random_rope
from hd_utils.yes_or_no import yes_or_no

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--demo_name", help="Name of demonstration", type=str, default='')
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
parser.add_argument("--perturb_fname", help="File saved perturb point clouds")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()





demotype_dir = osp.join(demo_files_dir, args.demo_type)
h5file = osp.join(demotype_dir, args.demo_type+".h5")


perturb_h5file = osp.join(demotype_dir, args.perturb_fname+".h5") 
print perturb_h5file
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

if args.demo_name == '':
    demo_type_dir = osp.join(demo_files_dir, args.demo_type)
    demo_master_file = osp.join(demo_type_dir, master_name)
    
    with open(demo_master_file, 'r') as fh:
        demos_info = yaml.load(fh)
    
    for demo in demos_info["demos"]:
        demo_name = demo["demo_name"]
        for seg_name in demofile[demo_name]:
            if seg_name == "done": continue
            print demo_name, seg_name
            xyz = demofile[demo_name][seg_name]["cloud_xyz"]
            xyz = np.squeeze(xyz)
            
            n_max_perturbed = 2
            n_perturbed = 0
            
            while n_perturbed < n_max_perturbed:
                new_xyz = sample_random_rope(xyz)
                
                ax = fig.gca(projection='3d')
                ax.set_autoscale_on(False)  
                ax.plot(new_xyz[:,0], new_xyz[:,1], new_xyz[:,2], 'o')            
                fig.show()
                
                if yes_or_no("Are you happy with this perturbation?"):
                    n_perturbed += 1
                    
                    perturb_demo_name = "perturb%i"%(num_h5_perturbs)
                    print "add perturb demo" + perturb_demo_name
                    demo_group = perturb_demofile.create_group(perturb_demo_name)
                    seg_group = demo_group.create_group(seg_name)
                    seg_group["cloud_xyz"] = new_xyz
                    num_h5_perturbs += 1
                fig.clf()
else:
    for seg_name in demofile[args.demo_name]:
        if seg_name == "done": continue
        print seg_name
        xyz = demofile[args.demo_name][seg_name]["cloud_xyz"]
        xyz = np.squeeze(xyz)
        print xyz.shape
        
        n_max_perturbed = 2
        n_perturbed = 0
        
        while n_perturbed < n_max_perturbed:
            new_xyz = sample_random_rope(xyz)
            
            ax = fig.gca(projection='3d')
            ax.set_autoscale_on(False)
            ax.plot(new_xyz[:,0], new_xyz[:,1], new_xyz[:,2], 'o')
        
            fig.show()
            
            if yes_or_no("Are you happy with this perturbation?"):
                n_perturbed += 1
                    
                perturb_demo_name = "perturb%i"%(num_h5_perturbs)
                print "add perturb demo" + perturb_demo_name
                demo_group = perturb_demofile.create_group(perturb_demo_name)
                seg_group = demo_group.create_group(seg_name)
                seg_group["cloud_xyz"] = new_xyz
                num_h5_perturbs += 1
            fig.clf()