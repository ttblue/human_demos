import os.path as osp
import importlib
import numpy as np
import argparse
from hd_utils.defaults import demo_files_dir, demo_names
from hd_utils.extraction_utils import get_videos
from mpl_toolkits.mplot3d import axes3d
import pylab
from hd_utils import clouds_utils, clouds


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--demo_name", help="Name of demonstration")
parser.add_argument("--cloud_proc_func", default="extract_red")
parser.add_argument("--cloud_proc_mod", default="hd_utils.cloud_proc_funcs")
args = parser.parse_args()


cloud_proc_mod = importlib.import_module(args.cloud_proc_mod)
cloud_proc_func = getattr(cloud_proc_mod, args.cloud_proc_func)


task_dir = osp.join(demo_files_dir, args.demo_type)
task_file = osp.join(task_dir, args.demo_name)

video_dir = osp.join(task_file, demo_names.video_dir%1)
rgbs, depths = get_videos(video_dir)

fig = pylab.figure()


for (rgb, depth) in zip(rgbs, depths):
    xyz = cloud_proc_func(np.asarray(rgb), np.asarray(depth), np.eye(4))
    
    ax = fig.gca(projection='3d')
    ax.plot(xyz[:,0], xyz[:,1], xyz[:,2], 'o')
    
    #clouds_utils.find_path_through_point_cloud(xyz, True)
    
    
    fig.show()
    raw_input()
    fig.clf()
    
    
    



