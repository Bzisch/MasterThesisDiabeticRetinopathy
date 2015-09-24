from caffe.io import load_image
import os
import sys
import numpy as np
import multiprocessing as mp
import scipy
import time
from distance.distance import cdist as native_cdist
from lmdbWriter import open_csv
from skimage.util import view_as_windows

def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Extracts features from every image')
    parser.add_argument('-path', help='Path to directory where lmdb database is stored')
    parser.add_argument('-size', help='Patch size')
    parser.add_argument('-stride', help='Stride')
    args = parser.parse_args()
    return args


def im2col(Im, block, stride, style='sliding'):
    """block = (patchsize, patchsize)
        first do sliding
    """
    bx, by = block
    Imx, Imy = Im.shape
    Imcol = []
    for j in range(0, Imy, stride):
        for i in range(0, Imx, stride):
            if (i+bx <= Imx) and (j+by <= Imy):
                Imcol.append(Im[i:i+bx, j:j+by].T.reshape(bx*by))
            else:
                break
    return np.asarray(Imcol).T


def stack(labelDataPair, i, numEntries):

    flattened = np.transpose(labelDataPair[1]).flatten()
    if i % 5000 == 0:
        print "{0}/{1} images flattened".format(i, numEntries)

    return flattened


def extract_features(path, keys, centroids, rfSize, ImageDim, whitening, M, P, stride):
    #assert(nargin == 4 || nargin == 6)
    numCentroids = centroids.shape[0]
    numSamples = len(keys)
    numFeats = ImageDim[0]*ImageDim[1]*ImageDim[2]
    # compute features for all training images
    XC = np.zeros((numSamples, numCentroids*4))
    labels = np.zeros((numSamples, 1))
    j = 0
    for i in range(numSamples):

        total_start = time.time()
        print "Sample {0}".format(i)
        if (np.mod(i,2000) == 0):
            np.save('test_features_windowsize_{0}_iteration_{1}'.format(rfSize,i), XC)
            print 'Extracting features: ' + str(i) + '/' + str(numSamples)
        
        img = load_image(path + keys[i])
#        X = np.transpose(reader.next()[1]).flatten()
        X = img.flatten()

        # extract overlapping sub-patches into rows of 'patches'
        start = time.time()
        patches = np.vstack(
                (im2col(np.reshape(X[0:numFeats/3],ImageDim[0:2],'F'), (rfSize, rfSize), stride),
                im2col(np.reshape(X[numFeats/3:numFeats*2/3],ImageDim[0:2],'F'), (rfSize, rfSize), stride), im2col(np.reshape(X[numFeats*2/3:numFeats],ImageDim[0:2],'F'), (rfSize, rfSize), stride))).T
        end = time.time()
        
        w = view_as_windows(img, (rfSize,rfsize, 3), stride)
        w = w.reshape((w.shape[0]*w.shape[1],w.shape[3]*w.shape[4]*w.shape[5]))
        print np.array_equal(patches, w)
        from time import sleep
        for i in xrange(w.shape[0]):
            print w[i,0:20]
            print patches[i,500:520]
            sleep(1)

        print "Extract overlapping sub-patches time:{0}".format(end - start)

        # do preprocessing for each patch

    
        # normalize for contrast
        start = time.time()
        patchesMean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
        patchesVar = np.var(patches, axis=1, dtype=np.float32, ddof=1, keepdims=True)
        offsetMatrix = 10.0 * np.ones(patchesVar.shape)        
        patches = (patches - patchesMean) / np.sqrt(patchesVar + offsetMatrix)
        end = time.time()
        print "Preprocessing time:{0}".format(end - start)
        # whiten
        if (whitening):
            patches = np.dot((patches - M), P)
        # compute 'triangle' activation function
        start = time.time()
        z = native_cdist(patches, centroids)
        end = time.time()
        print "Triangle time:{0}".format(end - start)

        start = time.time()
        mu = np.tile(np.array([np.mean(z, axis = 1)]).T, (1, centroids.shape[0])) # average distance to centroids for each patch
        patches = np.maximum(mu - z, np.zeros(mu.shape))
        end = time.time()
        print "Distance calculation time:{0}".format(end - start)
        # patches is now the data matrix of activations for each patch

        # reshape to numCentroids-channel image
        start = time.time()
        prows = (ImageDim[0]-rfSize + 1*stride)/stride
        pcols = (ImageDim[1]-rfSize + 1*stride)/stride
        patches = np.reshape(patches, (prows, pcols, numCentroids),'F')
        end = time.time()
        print "Reshaping time:{0}".format(end - start)
        start = time.time()
        # pool over quadrants
        halfr = np.round(float(prows)/2)
        halfc = np.round(float(pcols)/2)
        q1 = np.array([np.sum(np.sum(patches[0:halfc, 0:halfr, :], axis = 1),axis = 0)])
        q2 = np.array([np.sum(np.sum(patches[halfc:patches.shape[0], 0:halfr, :], axis = 1),axis = 0)])
        q3 = np.array([np.sum(np.sum(patches[0:halfc, halfr:patches.shape[1], :], axis = 1),axis = 0)])
        q4 = np.array([np.sum(np.sum(patches[halfc:patches.shape[0], halfr:patches.shape[1], :], axis = 1),axis = 0)])
        end = time.time()
        print "Pooling time:{0}".format(end - start)

        # concatenate into feature vector
        XC[j,:] = np.vstack((q1,q2,q3,q4)).flatten()
        j += 1
        total_end = time.time()
        print "Iteration time:{0}".format(total_end - total_start)

    return XC


if __name__ == '__main__':
    
    arguments = main(sys.argv[1:])
    path = arguments.path
    rfsize = int(arguments.size)
    if arguments.stride:
        stride =int( arguments.stride)
    else:
        stride = 1

    #reader = LmdbReader(path)
    #numEntries = reader.info()['entries']
    ImageDim = (512,512,3)
    centroids = np.load('npy_data/train_centroids_windowsize_{0}.npy'.format(rfsize))
    M = np.load('npy_data/train_mean_windowsize_{0}.npy'.format(rfsize))
    P = np.load('npy_data/train_eigenvectors_windowsize_{0}.npy'.format(rfsize))
    

#    trainX = np.zeros((numEntries,ImageDim[0]*ImageDim[1]*ImageDim[2]))
#    trainY = np.zeros((numEntries,))
#    i = 0

#    while(1):

#       labelDataPair = reader.next()
#        if labelDataPair is False:
#            break
#        else:
#            trainY[i,] = labelDataPair[0]
#            trainX[i,:] = np.transpose(labelDataPair[1]).flatten()
#            i += 1
#            if i % 100 == 0:
#                print i
    import json 
    keys = os.listdir(path)
    with open('keys.txt', 'w') as f:
        json.dump(keys, f)
    features = extract_features(path, keys, centroids, rfsize, ImageDim, True, M, P, stride)
    np.save('train_features_windowsize_{0}'.format(rfsize), features)
    
#    print trainY
    
#    reader = LmdbReader(path)
#           print trainX.shape, trainY.shape
    
#    pool = mp.Pool()
#    results = [pool.apply(stack, args=(reader.next(), i, numEntries - 1)) for i in range(0, numEntries)]
#    print "Finished"
    
