import numpy as np

def cloud_backto_xy(xyz, T_w_k):
    cx = 320. - .5
    cy = 240. - .5
    from hd_utils.defaults import asus_xtion_pro_f

    xyz_k = (xyz - T_w_k[:3,3][None,:]).dot(T_w_k[:3, :3])
    
    x = xyz_k[:, 0] / xyz_k[:, 2] * asus_xtion_pro_f + cx
    y = xyz_k[:, 1] / xyz_k[:, 2] * asus_xtion_pro_f + cy
    
    x = np.expand_dims(x, axis=1)
    y = np.expand_dims(y, axis=1)
    
    return np.concatenate((x, y), axis=1)


def predictCrossing2D(xy, image, net):
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    patch_size = params[0][2]
    offset = np.round(patch_size / 2.).astype(int)
    
    patches_indices = []
    for i in range(len(xy)):
        x_start = xy[i, 0] - offset
        y_start = xy[i, 1] - offset
        
        #patch = image[x_start:x_start+patch_size, y_start:y_start+patch_size, :]
        patch = image[y_start:y_start+patch_size, x_start:x_start+patch_size, :] / 255.0
        patches_indices.append((patch, i))
        
    patches, indices = zip(*patches_indices)
    
    n_patch_images = len(patches)
    n_iterations = np.ceil(n_patch_images / np.double(n_parallel))
    n_iterations = int(n_iterations)
    
    rope_crossing_predicts = []
    for i in range(n_iterations):
        start_id = n_parallel * i
        end_id = min(n_parallel * (i + 1), n_patch_images)
        scores = net.predict(patches[start_id:end_id], oversample=True)
        if end_id == n_patch_images:
            scores = scores[:end_id-start_id, :]
            
        print scores
            
        predicts = np.argmax(scores, axis=1)
        rope_crossing_predicts = np.concatenate((rope_crossing_predicts, predicts)).astype(int)
        
    return rope_crossing_predicts


def predictCrossing3D(xyz, image, net):    
    params = [v.data.shape for k, v in net.blobs.items()]
    n_parallel = params[0][0]
    patch_size = params[0][2]
    offset = np.round(patch_size / 2.).astype(int)

    T_w_k = np.eye(4)
    height, width, channel = image.shape
    xy = np.round(cloud_backto_xy(xyz, T_w_k)).astype(int)
        
    valid_mask = (xy[:, 0] - offset >= 0) & (xy[:, 0] - offset + patch_size <= width) & (xy[:, 1] - offset >= 0) & (xy[:, 1] - offset + patch_size <= height)
    xy = xy[valid_mask]
    
    rope_crossing_predicts_valid_points = predictCrossing2D(xy, image, net)
            
    rope_crossing_predicts = np.empty((len(xyz), 1))
    rope_crossing_predicts.fill(-1)
    rope_crossing_predicts[valid_mask] = np.expand_dims(rope_crossing_predicts_valid_points, axis=1)

    return rope_crossing_predicts

def learned_label_2_manual_label(labels):
    labels2 = []
    for i in range(len(labels)):
        if labels[i] == 0:
            label = -2 # error
        elif labels[i] == 1:
            label = 0 # end point
        elif labels[i] == 2:
            label = 0 # general rope segments
        else: # labels[i] == 3:
            label = 1 # crossing, but we don't know over or under crossing
        
        labels2.append(label)
        
    return labels2

def manual_label_2_learned_label(labels):
    labels2 = []
    for i in range(len(labels)):
        if i == 0 or i == len(labels) - 1: # end points
            label = 1
        elif labels[i] == 0: # general rope points 
            label = 2
        elif labels[i] == 1: # crossing
            label = 3
        elif labels[i] == -1: # crossing
            label = 3
        else: 
            label = 0 # will never happen
            
        labels2.append(label)
        
    return labels2
            



#def predictCrossing(xyz, image, net):
#    params = [v.data.shape for k, v in net.blobs.items()]
#    n_parallel = params[0][0]
#    patch_size = params[0][2]
#    offset = np.round(patch_size / 2.).astype(int)
#
#    
#    T_w_k = np.eye(4)
#    height, width, channel = image.shape
#    xy = np.round(cloud_backto_xy(xyz, T_w_k)).astype(int)
#        
#
#
#    valid_mask = (xy[:, 0] - offset >= 0) & (xy[:, 0] - offset + patch_size <= width) & (xy[:, 1] - offset >= 0) & (xy[:, 1] - offset + patch_size <= height)
#    xy = xy[valid_mask]
#    
#    patches_indices = []
#    for i in range(len(xy)):
#        x_start = xy[i, 0] - offset
#        y_start = xy[i, 1] - offset
#        
#        patch = image[x_start:x_start+patch_size, y_start:y_start+patch_size, :]
#        patches_indices.append((patch, i))
#        
#    patches, indices = zip(*patches_indices)
#    
#    n_patch_images = len(patches)
#    n_iterations = np.ceil(n_patch_images / np.double(n_parallel))
#    n_iterations = int(n_iterations)
#    
#    rope_crossing_predicts_valid_points = []
#    for i in range(n_iterations):
#        start_id = n_parallel * i
#        end_id = min(n_parallel * (i + 1), n_patch_images)
#        scores = net.predict(patches[start_id:end_id], oversample=True)
#        if end_id == n_patch_images:
#            scores = scores[:end_id-start_id, :]
#            
#
#            
#        predicts = np.argmax(scores, axis=1)
#        rope_crossing_predicts_valid_points = np.concatenate((rope_crossing_predicts_valid_points, predicts)).astype(int)
#        
#    rope_crossing_predicts = np.empty((len(xyz), 1))
#    rope_crossing_predicts.fill(-1)
#    rope_crossing_predicts[valid_mask] = np.expand_dims(rope_crossing_predicts_valid_points, axis=1)
#
#    return rope_crossing_predicts
