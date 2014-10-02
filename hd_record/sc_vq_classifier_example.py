import numpy as np
import leveldb
import argparse
import os, os.path as osp
from os import listdir
import cv2
import yaml
import shutil
import caffe
from caffe.io import caffe_pb2
from scipy import linalg
from create_leveldb_utils import *
import cPickle as cp
import h5py

from hd_utils.defaults import demo_files_dir, demo_names, master_name, verify_name

from scipy.io import loadmat

from sc_vq_classifier import *
import cPickle


parser = argparse.ArgumentParser()
parser.add_argument("--demo_type", help="Type of demonstration")
#parser.add_argument("--vq_result", help="File name for kmeans vq")
parser.add_argument("--train_db", type=str)
parser.add_argument("--test_db", type=str)
args = parser.parse_args()


task_dir = osp.join(demo_files_dir, args.demo_type)
ldbpath_train = osp.join(task_dir, args.train_db)
ldbpath_test = osp.join(task_dir, args.test_db)
ldb_train = leveldb.LevelDB(ldbpath_train)
ldb_test = leveldb.LevelDB(ldbpath_test)

train_labels = collect_labels_from_leveldb(ldb_train)
test_labels = collect_labels_from_leveldb(ldb_test)
train_images = collect_images_from_leveldb(ldb_train)
test_images = collect_images_from_leveldb(ldb_test)


#vq_result = loadmat(args.vq_result)
#svmLearner, trainXC_mean, trainXC_sd = sc_vq_train(train_images, train_labels, vq_result)
#training_result = {'svmLearner': svmLearner, 'trainXC_mean': trainXC_mean, 'trainXC_sd': trainXC_sd}
#with open('training_result_type_a.cp', 'wb') as fp:
#    cPickle.dump(training_result, fp)


#sc_vq_test(test_images, test_labels, vq_result, svmLearner, trainXC_mean, trainXC_sd)


#svmLearner, trainXC_mean, trainXC_sd, vq_result = sc_vq_train2(train_images, train_labels, 6, 0.25, 200, 400000)
svmLearner, trainXC_mean, trainXC_sd, vq_result = sc_vq_train2(train_images, train_labels, 6, 0.25, 300, 4000000)
training_result = {'svmLearner': svmLearner, 'trainXC_mean': trainXC_mean, 'trainXC_sd': trainXC_sd, 'vq_result': vq_result}


with open('training_result_type_b.cp', 'wb') as fp:
    cPickle.dump(training_result, fp)

sc_vq_test2(test_images, test_labels, vq_result, svmLearner, trainXC_mean, trainXC_sd)



