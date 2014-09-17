#convert levelset data to pylearn2 dataset

import argparse
import os, os.path as osp
import numpy as np
import pylearn2
from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from create_leveldb_utils import *

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





class levelsetData(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, which_set, train_db_dir, test_db_dir, axes=('b', 0, 1, 'c')):
        
        self.axes = axes
        dtype = 'uint8'
        
        
        train_db = leveldb.LevelDB(train_db_dir)
        test_db = leveldb.LevelDB(test_db_dir)
        
        self.img_shape = get_leveldb_image_shape(train_db)
        print self.img_shape
        self.img_size = np.prod(self.img_shape)
        
        train_data = collect_data_from_leveldb(train_db, use_cv=False)
        train_data = train_data.T
        train_labels = collect_labels_from_leveldb(train_db)
        unique_labels = set(train_labels)
        n_classes = len(unique_labels)
        self.n_classes = n_classes
        

        test_data = collect_data_from_leveldb(test_db, use_cv=False)
        test_data = test_data.T
        test_labels = collect_labels_from_leveldb(test_db)
        

        
        Xs = {'train': train_data, 'test': test_data}
        Ys = {'train': train_labels, 'test': test_labels}
        
        X = np.cast['float32'](Xs[which_set])
        y = Ys[which_set]
        
        if isinstance(y, list):
            y = np.asarray(y).astype(dtype)
        
        
        y = y.reshape((y.shape[0], 1))
            
        print train_data.shape
        print test_data.shape
        print X.shape
        print y.shape
        
        self.center = False
        self.rescale = False
        self.gcn = None
        
        view_converter = dense_design_matrix.DefaultViewConverter((self.img_shape[1], self.img_shape[2], self.img_shape[0]),
                                                                  axes)
        
        super(levelsetData, self).__init__(X=X, y=y, view_converter=view_converter, y_labels=n_classes)
        

train = levelsetData(which_set="train", train_db_dir=train_db_dir, test_db_dir=test_db_dir)
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000))

# Next we contrast normalize the patches. The default arguments use the
# same "regularization" parameters as those used in Adam Coates, Honglak
# Lee, and Andrew Ng's paper "An Analysis of Single-Layer Networks in
# Unsupervised Feature Learning"
pipeline.items.append(preprocessing.GlobalContrastNormalization(sqrt_bias=10., use_std=True))

# Finally we whiten the data using ZCA. Again, the default parameters to
# ZCA are set to the same values as those used in the previously mentioned
# paper.
pipeline.items.append(preprocessing.ZCA())

# Here we apply the preprocessing pipeline to the dataset. The can_fit
# argument indicates that data-driven preprocessing steps (such as the ZCA
# step in this example) are allowed to fit themselves to this dataset.
# Later we might want to run the same pipeline on the test set with the
# can_fit flag set to False, in order to make sure that the same whitening
# matrix was used on both datasets.
train.apply_preprocessor(preprocessor=pipeline, can_fit=True)

# Finally we save the dataset to the filesystem. We instruct the dataset to
# store its design matrix as a numpy file because this uses less memory
# when re-loading (Pickle files, in general, use double their actual size
# in the process of being re-loaded into a running process).
# The dataset object itself is stored as a pickle file.

path = pylearn2.__path__[0]
train_example_path = os.path.join(path, 'scripts', 'tutorials', 'grbm_smd_rope')
train.use_design_loc(os.path.join(train_example_path, 'preprocessed_train_design_rope.npy'))

train_pkl_path = os.path.join(train_example_path, 'preprocessed_train_rope.pkl')
serial.save(train_pkl_path, train)


   
        