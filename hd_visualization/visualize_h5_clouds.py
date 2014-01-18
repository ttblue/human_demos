import os.path as osp
import numpy as np
import argparse
from hd_utils.defaults import demo_files_dir
from mpl_toolkits.mplot3d import axes3d
import pylab
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration", type=str)
parser.add_argument("--h5_name", help="Name of h5", type=str, default='')
parser.add_argument("--demo_name", help="Name of demonstration", type=str, default='')
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
args = parser.parse_args()



demotype_dir = osp.join(demo_files_dir, args.demo_type)
if args.h5_name == '':
    h5file = osp.join(demotype_dir, args.demo_type+".h5")
else:
    h5file = osp.join(demotype_dir, args.h5_name+".h5")

demofile = h5py.File(h5file, 'r')

fig = pylab.figure()

if args.demo_name == '':
    for demo_name in demofile.keys():
        for seg_name in demofile[demo_name]:
            print demo_name, seg_name
            
            xyz = demofile[demo_name][seg_name]["cloud_xyz"]
            xyz = np.squeeze(xyz)
            print xyz.shape
        
            ax = fig.gca(projection='3d')
            ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
    
            
            fig.show()
            raw_input()
            fig.clf()
        
else:
    for seg_name in demofile[args.demo_name]:
        print seg_name
        xyz = demofile[args.demo_name][seg_name]["cloud_xyz"]
        xyz = np.squeeze(xyz)
        
        ax = fig.gca(projection='3d')
        ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
    
            
        fig.show()
        raw_input()
        fig.clf()

    
    


