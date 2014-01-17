import os.path as osp
import importlib
import numpy as np
import argparse
from hd_utils.defaults import demo_files_dir, demo_names
from hd_utils.extraction_utils import get_videos
from mpl_toolkits.mplot3d import axes3d
import pylab
import h5py
from hd_utils import clouds


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--demo_name", help="Name of demonstration")
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
args = parser.parse_args()


demotype_dir = osp.join(demo_files_dir, args.demo_type)
h5file = osp.join(demotype_dir, args.demo_type+".h5")

demofile = h5py.File(h5file, 'r')

for seg_name in demofile[args.demo_name]:
    print seg_name
    xyz = demofile[args.demo_name][seg_name]["cloud_xyz"]
    xyz = np.squeeze(xyz)
    print xyz.shape
    #xyz1 = clouds.remove_outliers(xyz, 1, 50)
    #xyz1 = xyz #clouds.downsample(xyz, 0.045)
    
    fig = pylab.figure()
    ax = fig.gca(projection='3d')
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
    
    fig.show()
    
    raw_input()

    
    


