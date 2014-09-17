import numpy as np
import leveldb
import caffe
from caffe.io import caffe_pb2
from scipy import linalg
from hd_utils.extraction_utils import get_video_frames
from pylearn2.datasets import preprocessing
from pylearn2.utils import serial
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix


# for datum patch: channel * height * width
# for cv2: height * width * channel

def datum2cv(patch_in):
    patch_out = patch_in.copy()
    patch_out = np.swapaxes(patch_out, 0, 2)
    patch_out = np.swapaxes(patch_out, 0, 1)
    return patch_out

def cv2datum(patch_in):
    patch_out = patch_in.copy()
    patch_out = np.swapaxes(patch_out, 0, 2)
    patch_out = np.swapaxes(patch_out, 1, 2)
    return patch_out
    
def get_leveldb_image_shape(db):
    for key in db.RangeIter(include_value = False):
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        patch = caffe.io.datum_to_array(patch)
        img_size = patch.shape
        break
    
    return img_size
    
    
def collect_labels_from_leveldb(db):
    labels = []
    for key in db.RangeIter(include_value = False):
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        labels.append(patch.label)
    
    return labels


def collect_images_from_leveldb(db, use_cv=True):
    patches = []
    for key in db.RangeIter(include_value = False):
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        patch = caffe.io.datum_to_array(patch)
        if use_cv:
            patch = datum2cv(patch)
        patches.append(patch)
        
    patches = np.asarray(patches)
        
    return patches


def collect_data_from_leveldb(db, use_cv=True):
    patches = []
    for key in db.RangeIter(include_value = False):
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        patch = caffe.io.datum_to_array(patch)
        if use_cv:
            patch = datum2cv(patch)
        patches.append(patch)
        
    patches = np.asarray(patches)
    count_per_patch = patches.shape[1] * patches.shape[2] * patches.shape[3]
    patches = patches.reshape(patches.shape[0], count_per_patch)
    patches = patches.T
    
    return patches


def generate_leveldb_per_channel(db, channel_db, channel_idx):

    batch = leveldb.WriteBatch()
    for key in db.RangeIter(include_value = False):
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        patch = caffe.io.datum_to_array(patch)
        
        patch = patch[channel_idx, :, : ]
        patch = np.expand_dims(patch, axis=0)
        datum = caffe.io.array_to_datum(patch)
        batch.Put(key, datum.SerializeToString())
    
    channel_db.Write(batch, sync=True)
    
def compute_ZCA_fast(X, normalize, ZCA_filename="zca"):
    zca_preprocessor = preprocessing.ZCA()
    zca_preprocessor.set_matrices_save_path(ZCA_filename+".npz")
    X = X.astype(np.float32)
    if normalize:
        X /= 255.0
    zca_preprocessor.fit(X.T)
    serial.save(ZCA_filename+".pkl", zca_preprocessor)
    

def apply_ZCA_fast(patches, normalize, zca_preprocessor):
    patches = patches.astype(np.float32)
    if normalize:
        patches /= 255.0
    dataset = DenseDesignMatrix(X = patches.T)    
    zca_preprocessor.apply(dataset)
    patches = dataset.get_design_matrix()
    return patches.T

def apply_ZCA_db_fast2(db, normalize, ZCA_filename="zca"):
    zca_preprocessor = preprocessing.ZCA()
    zca_preprocessor = serial.load(ZCA_filename+".pkl")
    
    patches = collect_data_from_leveldb(db)
    patches = apply_ZCA_fast(patches, normalize, zca_preprocessor)
    
    batch = leveldb.WriteBatch()
    i = 0
    for key in db.RangeIter(include_value = False):
        print key
        
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        label = patch.label
        patch = caffe.io.datum_to_array(patch)
        shape = patch.shape
        
        ZCA_patch = patches[:, i]
        ZCA_patch = ZCA_patch.reshape(shape[1], shape[2], shape[0])        
        ZCA_patch = cv2datum(ZCA_patch)
        ZCA_patch = caffe.io.array_to_datum(ZCA_patch, label)
        ZCA_patch = ZCA_patch.SerializeToString()
        batch.Put(key, ZCA_patch)
        i += 1
    db.Write(batch, sync=True)
    
    
def apply_ZCA_db_fast3(db, normalize, ZCA_filename="zca"):   
    zca_preprocessor = preprocessing.ZCA()
    zca_preprocessor = serial.load(ZCA_filename+".pkl")    
 
    batch = leveldb.WriteBatch()
    for key in db.RangeIter(include_value = False):
        print key
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        label = patch.label
        patch = caffe.io.datum_to_array(patch)
        patch = datum2cv(patch)
        shape = patch.shape
        patch = patch.reshape(shape[0] * shape[1] * shape[2], 1)
        ZCA_patch = apply_ZCA_fast(patch, normalize, zca_preprocessor)
        ZCA_patch = ZCA_patch.reshape(shape[0], shape[1], shape[2])        
        ZCA_patch = cv2datum(ZCA_patch)
        ZCA_patch = caffe.io.array_to_datum(ZCA_patch, label)
        ZCA_patch = ZCA_patch.SerializeToString()
        batch.Put(key, ZCA_patch)
    db.Write(batch, sync=True)
    
    
def apply_ZCA_db_fast(db, normalize, ZCA_filename="zca"):
    zca_preprocessor = preprocessing.ZCA()
    zca_preprocessor = serial.load(ZCA_filename+".pkl")     
    
    patches = collect_data_from_leveldb(db)
    
    n_patches = patches.shape[1]
    n_per_iteration = 10000
    n_iterations = np.ceil(n_patches / float(n_per_iteration))
    
    
    
    n = 0
    i = 0
    
    for key in db.RangeIter(include_value = False):
        if i >= n * n_per_iteration:
            if i > 0:
                db.Write(batch, sync=True)     
            batch = leveldb.WriteBatch()     
            patches_ = patches[:, n*n_per_iteration:min((n+1)*n_per_iteration, n_patches)]
            n = n + 1
            patches_ = apply_ZCA_fast(patches_, normalize, zca_preprocessor) 
            
        #print key
        
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        label = patch.label
        patch = caffe.io.datum_to_array(patch)
        shape = patch.shape
        
        ZCA_patch = patches_[:, i - (n-1) * n_per_iteration]
        ZCA_patch = ZCA_patch.reshape(shape[1], shape[2], shape[0])        
        ZCA_patch = cv2datum(ZCA_patch)
        ZCA_patch = caffe.io.array_to_datum(ZCA_patch, label)
        ZCA_patch = ZCA_patch.SerializeToString()
        batch.Put(key, ZCA_patch)
        i += 1
    db.Write(batch, sync=True)
        
    


    
    
#def apply_ZCA_db_fast(db, ZCA_filename="zca"):
#    zca_preprocessor = preprocessing.ZCA()
#    zca_preprocessor.set_matrices_save_path(ZCA_filename+".npz")
#    zca_preprocessor = serial.load(ZCA_filename+".pkl")
#    
#    batch = leveldb.WriteBatch()
#    for key in db.RangeIter(include_value = False):
#        print key
#        patch_string = db.Get(key)
#        patch = caffe_pb2.Datum.FromString(patch_string)
#        label = patch.label
#        patch = caffe.io.datum_to_array(patch)
#        shape = patch.shape
#        
#        #ZCA
#        patch = patch.reshape(shape[0] * shape[1] * shape[2])
#        patch = patch.astype(np.float32)
#        patch = zca_preprocessor.apply(patch.T)        
#        patch = patch.reshape([shape[0], shape[1], shape[2]])
#        patch = caffe.io.array_to_datum(patch)
#        patch = patch.SerializeToString()
#        batch.Put(key, patch)
#    db.Write(batch, sync=True)
    
# X should be in matrix mode (each column is the vectorization of each item)
def compute_ZCA(X, regularization=10**-5):
    ZCA_mean = np.mean(X, axis=1)
    [dim_X, n_X] = X.shape
    X_new = X - np.tile(ZCA_mean.reshape(dim_X, 1), (1, n_X))
    sigma = np.dot(X_new, X_new.T) / X_new.shape[1]
    U, S, V = linalg.svd(sigma)
    tmp = np.dot(U, np.diag(1/np.sqrt(S+regularization)))
    ZCA_rot = np.dot(tmp, U.T)
    return ZCA_mean, ZCA_rot
    
# patches should be in matrix mode (each column for each patch)
def apply_ZCA(patches, ZCA_mean, ZCA_rot):
    [dim_patch, num_patches] = patches.shape
    patches_new = patches - np.tile(ZCA_mean.reshape(dim_patch, 1), (1, num_patches))
    patches_new = np.dot(ZCA_rot, patches_new)
    return patches_new

def apply_ZCA_db(db, ZCA_mean, ZCA_rot):
    batch = leveldb.WriteBatch()
    for key in db.RangeIter(include_value = False):
        print key
        patch_string = db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        label = patch.label
        patch = caffe.io.datum_to_array(patch)
        patch = cv2datum(patch)
        
        shape = patch.shape
        patch = patch.reshape(shape[0] * shape[1] * shape[2])
        
        # ZCA 
        patch = patch - ZCA_mean
        patch = np.dot(ZCA_rot, patch)        
        patch = patch.reshape([shape[0], shape[1], shape[2]])
        patch = datum2cv(patch)
        patch = caffe.io.array_to_datum(patch)
        patch = patch.SerializeToString()
        batch.Put(key, patch)
    db.Write(batch, sync=True)
    
    
def sample_patches(image, patch_size, num_samples):
    (h, w, c) = image.shape
    starts_x = np.random.randint(0, h - patch_size, num_samples)
    starts_y = np.random.randint(0, w - patch_size, num_samples)
    
    patches = []
    for i in range(num_samples):
        patch = image[starts_x[i]:starts_x[i]+patch_size, starts_y[i]:starts_y[i]+patch_size, :]
        patches.append(patch)
    return patches

def sample_patches_leveldb(input_db, output_db, num_samples, patch_size):
    keys = [key for key in input_db.RangeIter(include_value = False)]
    num_images = len(keys)
    
    (c, h, w) = get_leveldb_image_shape(input_db)
    
    starts_x = np.random.randint(0, h - patch_size, num_samples)
    starts_y = np.random.randint(0, w - patch_size, num_samples)

    batch = leveldb.WriteBatch()
    for i in range(num_samples):
        image_id = i % num_images
        key = keys[image_id]
        
        patch_string = input_db.Get(key)
        patch = caffe_pb2.Datum.FromString(patch_string)
        label = patch.label
        patch = caffe.io.datum_to_array(patch)
        sub_patch = patch[:, starts_x[i]:starts_x[i]+patch_size, starts_y[i]:starts_y[i]+patch_size]
        sub_patch = caffe.io.array_to_datum(sub_patch, label)
        sub_patch = sub_patch.SerializeToString()
        batch.Put(key, sub_patch)
             
    output_db.Write(batch, sync=True)
    
    (c, h, w) = get_leveldb_image_shape(output_db)
    print c, h, w
    
    
    


def add_rgb_to_leveldb(video_dir, annotation, db, demo_name, patch_size, num_samples, start_id):
    frame_stamps = [seg_info["look"] for seg_info in annotation]
    rgb_imgs, depth_imgs = get_video_frames(video_dir, frame_stamps)
    
    batch = leveldb.WriteBatch()
    
    for (i_seg, seg_info) in enumerate(annotation):
        rgb_img = rgb_imgs[i_seg]
        (h, w, c) = rgb_img.shape
        
        patches = sample_patches(rgb_img, patch_size, num_samples)
        
        for (i, patch) in enumerate(patches):
            patch = cv2datum(patch)
            datum = caffe.io.array_to_datum(patch)
            patch_string = datum.SerializeToString()
            patch_unique_id = start_id
            start_id = start_id + 1
            batch.Put(str(patch_unique_id).zfill(7), patch_string)
            
    db.Write(batch, sync=True)
    
    return start_id