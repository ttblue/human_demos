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
args = parser.parse_args()

task_dir = osp.join(demo_files_dir, args.demo_type)
train_db_dir = osp.join(task_dir, args.train_db)
test_db_dir = osp.join(task_dir, args.test_db)

train_db = leveldb.LevelDB(train_db_dir)
test_db = leveldb.LevelDB(test_db_dir)

train_data = collect_data_from_leveldb(train_db, use_cv=False)
train_data = train_data.T
train_labels = collect_labels_from_leveldb(train_db)
test_data = collect_data_from_leveldb(test_db, use_cv=False)
test_data = test_data.T
test_labels = collect_labels_from_leveldb(test_db)

unique_labels = set(train_labels)
n_classes = len(unique_labels)
print "# of classes ", n_classes

train_mat_file = osp.join(task_dir, args.train_db+".mat")
test_mat_file = osp.join(task_dir, args.test_db+".mat")
sio.savemat(train_mat_file, {'data': train_data, 'labels': train_labels})
sio.savemat(test_mat_file, {'data': test_data, 'labels': test_labels})

