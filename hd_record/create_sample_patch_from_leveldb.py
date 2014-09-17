#convert levelset data to matlab

import argparse
import os, os.path as osp
import numpy as np
from create_leveldb_utils import *
import scipy.io as sio

import leveldb
from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
parser.add_argument("--train_db", type=str)
parser.add_argument("--test_db", type=str)
parser.add_argument("--sample_patch_size", type=int)
parser.add_argument("--num_samples", type=int)
args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
train_db_dir = osp.join(task_dir, args.train_db)
test_db_dir = osp.join(task_dir, args.test_db)

train_db = leveldb.LevelDB(train_db_dir)
test_db = leveldb.LevelDB(test_db_dir)

train_db_out = leveldb.LevelDB(train_db_dir+"_sample_"+str(args.sample_patch_size))
test_db_out = leveldb.LevelDB(test_db_dir+"_sample_"+str(args.sample_patch_size))

sample_patches_leveldb(train_db, train_db_out, args.num_samples, args.sample_patch_size)
sample_patches_leveldb(test_db, test_db_out, args.num_samples, args.sample_patch_size)


