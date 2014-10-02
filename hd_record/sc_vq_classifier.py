import numpy as np

from decaf.layers.cpp import wrapper
from scipy.sparse import csr_matrix

from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt


# our network takes BGR images, so we need to switch color channels
def showimage(im):
    im = im.squeeze()
    plt.imshow(im)
    
# take an array of shape (n, height, width) or (n, height, width, channels)
#  and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    showimage(data)
    



def run_omp1(X, k, iterations):
    dictionary = np.random.randn(k, X.shape[1])
    var_dictionary = np.sqrt(np.sum(dictionary * dictionary, axis=1) + 1e-20)
    var_dictionary = np.expand_dims(var_dictionary, axis=1)
    
    dictionary = dictionary / var_dictionary
    
    for itr in range(iterations):
        print "Running GSVQ: iteration=%d..." % (itr)
        
        # do assignment + accumulation
        summation, counts = gsvq_step(X, dictionary)
        
        # reinit empty clusters
        I = np.where(np.sum(summation * summation, axis=1) < 0.001)[0]
        summation[I, :] = np.random.randn(I.shape[0], X.shape[1])
        
        var_dictionary = np.sqrt(np.sum(summation * summation, axis=1) + 1e-20)
        var_dictionary = np.expand_dims(var_dictionary, axis=1)
        dictionary = summation / var_dictionary
        
    return dictionary

def run_kmeans(X, k, iterations):
    kmeans_cluster = KMeans(k, max_iter=iterations)
    kmeans_cluster.fit(X)
    return kmeans_cluster.cluster_centers_
    

def gsvq_step(X, dictionary):
    summation = np.zeros(dictionary.shape)
    counts = np.zeros([dictionary.shape[0], 1])
    
    k = dictionary.shape[0]
    
    BATCH_SIZE = 2000
    
    for i in xrange(0, X.shape[0], BATCH_SIZE):
        lastInd = np.min([i+BATCH_SIZE, X.shape[0]])
        m = lastInd - i
        
        dots = np.dot(dictionary, X[i:lastInd, :].T)
        labels = np.argmax(abs(dots), axis=0)
        
        E = csr_matrix((np.ones(m), np.asarray([labels, np.array(range(m))])), dots.shape)
        
        counts = counts + sum(E.T)
        
        E = np.array(E.todense())
        dots = dots * E
        summation = summation + np.dot(dots, X[i:lastInd, :])
        
    return (summation, counts)


def extract_patches(images, psize, num_patches):
    num, height, width, channels = images.shape
    patches = np.zeros([num_patches, psize*psize * channels])
    for i in range(num_patches):
        if i%10000 == 0:
            print "Extracting patch: %d / %d" % (i, num_patches)
        start_x = np.random.randint(0, height - psize)
        start_y = np.random.randint(0, width - psize)
        image_id = np.random.randint(0, num)
        patch_img = images[image_id, start_x:start_x+psize, start_y:start_y+psize, :]
        patches[i, :] = patch_img.reshape(psize * psize * channels)
        
    # normalize for contrast
    mean_patches = np.mean(patches, axis=1)
    var_patches = np.sqrt(np.var(patches, axis=1) + 10)
    mean_patches = np.expand_dims(mean_patches, axis=1)
    var_patches = np.expand_dims(var_patches, axis=1)
    
    patches = (patches - mean_patches) / var_patches
    
    return patches

def ZCA_whitening(patches):
    C = np.cov(patches.T)
    M = np.mean(patches, axis=0)
    [w, v] = np.linalg.eig(C)
    P = v * np.diag(np.sqrt(1.0/(w + 0.1))) * v.T
    patches = np.dot(patches - M, P)
    
    return (patches, M, P)        


def im2col(features, psize, stride):
    if features.ndim != 4:
        raise ValueError('Input features should be 4-dimensional.')
    
    num, height, width, channels = features.shape
    new_height = (height - psize) / stride + 1
    new_width = (width - psize) / stride + 1
    channels = channels * psize * psize

    output = np.zeros([num, new_height, new_width, channels])
        
    wrapper.im2col_forward(features, output, psize, stride)
    
    return output



def extract_features(images, D, rfSize, input_dim, M, P, alpha):
    num_bases = D.shape[0]
    
    # extract features for all images
    channel_length = input_dim[0] * input_dim[1]
    num_images = images.shape[0]
    XC = np.zeros([num_images, num_bases * 2 * 4])
    for i in range(num_images):
        if (i+1)%1000 == 0:
            print "Extracting features: %d / %d"  % (i+1, num_images)
                        
        image = images[i:i+1, :, :, :].astype(float)
        
        patches = im2col(image, rfSize, 1)
        patches = patches.reshape([patches.shape[1] * patches.shape[2], patches.shape[3]])
        
        # normalize for contrast
        mean_patches = np.mean(patches, axis=1)
        var_patches = np.sqrt(np.var(patches, axis=1) + 10)
        mean_patches = np.expand_dims(mean_patches, axis=1)
        var_patches = np.expand_dims(var_patches, axis=1)
        
        patches = (patches - mean_patches) / var_patches
        # whiten
        patches = np.dot(patches - M, P)
        
        # compute activation
        z = np.dot(patches, D.T)
        patches = np.concatenate((np.maximum(z - alpha, 0), np.maximum(-z - alpha, 0)), axis=1)
        
        # patches is now the data matrix of activations for each patch
        
        # reshape to 2*num_bases-channel image
        prows = input_dim[0] - rfSize + 1
        pcols = input_dim[1] - rfSize + 1
        
        patches = patches.reshape(prows, pcols, num_bases * 2)
        
        # pool over quandrants
        halfr = int(prows / 2.)
        halfc = int(pcols / 2.)
        q1 = np.sum(np.sum(patches[0:halfr, 0:halfc, :], axis=0), axis=0)
        q2 = np.sum(np.sum(patches[halfr:, 0:halfc, :], axis=0), axis=0)
        q3 = np.sum(np.sum(patches[0:halfr, halfc:, :], axis=0), axis=0)
        q4 = np.sum(np.sum(patches[halfr:, halfc:, :], axis=0), axis=0)
        
        XC[i, :] = np.concatenate((q1, q2, q3, q4), axis=0)
    
    return XC


def sc_vq_train(images, labels, vq_result):
    dictionary = vq_result['dictionary']
    M = vq_result['M']
    P = vq_result['P']
    rfSize = vq_result['rfSize'][0, 0].astype(int)
    alpha = vq_result['alpha'][0, 0]
    input_dim = vq_result['input_dim'][0].astype(int)
    
    trainXC = extract_features(images, dictionary, rfSize, input_dim, M, P, alpha)
        
    L = 0.01
    
    trainXC_mean = np.mean(trainXC, axis=0)
    trainXC_sd = np.sqrt(np.var(trainXC, axis=0)+0.01)
    trainXCs = (trainXC - trainXC_mean) / trainXC_sd
    
    trainXCs = np.concatenate((trainXC, np.ones([trainXCs.shape[0], 1])), axis=1)
    
    svmLearner = LinearSVC(C=1/L)
    svmLearner.fit(trainXCs, labels)
    print svmLearner.score(trainXCs, labels)
        
    return (svmLearner, trainXC_mean, trainXC_sd)

def sc_vq_test(images, labels, vq_result, svmLearner, trainXC_mean, trainXC_sd):
    dictionary = vq_result['dictionary']
    M = vq_result['M']
    P = vq_result['P']
    rfSize = vq_result['rfSize'][0, 0].astype(int)
    alpha = vq_result['alpha'][0, 0]
    input_dim = vq_result['input_dim'][0].astype(int)
    

    testXC = extract_features(images, dictionary, rfSize, input_dim, M, P, alpha)
    testXCs = (testXC - trainXC_mean) / trainXC_sd
    
    testXCs = np.concatenate((testXC, np.ones([testXCs.shape[0], 1])), axis=1)
    print svmLearner.score(testXCs, labels)



def sc_vq_train2(images, labels, rfSize, alpha, num_bases, num_patches):
    
    input_dim = images[0].shape
    patches = extract_patches(images, rfSize, num_patches)
    patches, M, P = ZCA_whitening(patches)


    dictionary = run_omp1(patches, num_bases, 50)
    
    
    trainXC = extract_features(images, dictionary, rfSize, input_dim, M, P, alpha)
        
    L = 0.01
    
    trainXC_mean = np.mean(trainXC, axis=0)
    trainXC_sd = np.sqrt(np.var(trainXC, axis=0)+0.01)
    trainXCs = (trainXC - trainXC_mean) / trainXC_sd
    
    trainXCs = np.concatenate((trainXC, np.ones([trainXCs.shape[0], 1])), axis=1)
    
    svmLearner = LinearSVC(C=1/L)
    
    svmLearner.fit(trainXCs, labels)
    print svmLearner.score(trainXCs, labels)
    
    vq_result = {'dictionary': dictionary, 'M': M, 'P': P, 'rfSize': rfSize, 'alpha': alpha, 'input_dim': input_dim}
    
    return (svmLearner, trainXC_mean, trainXC_sd, vq_result)


def sc_vq_test2(images, labels, vq_result, svmLearner, trainXC_mean, trainXC_sd):
    dictionary = vq_result['dictionary']
    M = vq_result['M']
    P = vq_result['P']
    rfSize = vq_result['rfSize']
    alpha = vq_result['alpha']
    input_dim = vq_result['input_dim']
    

    testXC = extract_features(images, dictionary, rfSize, input_dim, M, P, alpha)
    testXCs = (testXC - trainXC_mean) / trainXC_sd
    
    testXCs = np.concatenate((testXC, np.ones([testXCs.shape[0], 1])), axis=1)
    print svmLearner.score(testXCs, labels)
    

    
    
    
    
    
    
    