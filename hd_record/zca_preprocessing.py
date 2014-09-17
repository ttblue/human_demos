import numpy as np
import leveldb
import argparse
import os, os.path as osp
from os import listdir
import cv2
import yaml
import shutil
import caffe
from create_leveldb_utils import *
import cPickle as cp

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--input_db", type=str)
parser.add_argument("--compute_ZCA", action='store_true')
parser.add_argument("--image_normalize", action='store_true')
parser.add_argument("--ZCA_file", default='', type=str)
args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
db_dir = osp.join(task_dir, args.input_db)
new_db_dir = osp.join(task_dir, args.input_db+"_"+args.ZCA_file)


db = leveldb.LevelDB(db_dir)

if osp.exists(new_db_dir):
    shutil.rmtree(new_db_dir)
new_db = leveldb.LevelDB(new_db_dir)

batch = leveldb.WriteBatch()
for key in db.RangeIter(include_value = False):
    data = db.Get(key)
    batch.Put(key, data)
new_db.Write(batch, sync=True)
    

if args.compute_ZCA == True:
    #compute ZCA and apply
    data = collect_data_from_leveldb(new_db)
    if data.shape[1] > 10000:
        sample_data_ids = np.random.random_integers(0, data.shape[1] - 1, 10000)
        data = data[:, sample_data_ids]
    compute_ZCA_fast(data, args.image_normalize, osp.join(task_dir, args.ZCA_file))
    apply_ZCA_db_fast(new_db, args.image_normalize, osp.join(task_dir, args.ZCA_file))
else:
    #only apply ZCA
    apply_ZCA_db_fast(new_db, args.image_normalize, osp.join(task_dir, args.ZCA_file))
    
    
