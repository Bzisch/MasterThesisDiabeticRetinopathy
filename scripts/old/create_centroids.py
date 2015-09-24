__author__ = 'jan'
import sys
import numpy as np
from lmdbReader import LmdbReader
from kMeansFeatureExtractor import *
from sklearn.cluster import MiniBatchKMeans


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(description='Trains classifier')
    parser.add_argument('-path', help='Path to directory where images are stored')
    parser.add_argument('-stride', help='Stride for window extraction')
    #later add optional unsupervised algorithm parameter
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    rfSize = 16
    numCentroids=450
    whitening=True
    imageDim=(256, 256, 3)
    maxIter = 50
    batchSize = 500
    numPatchesByImage = 12
    #read the database in lmdb format
    arguments = main(sys.argv[1:])
    path = arguments.path
    if arguments.stride:
        stride = int(arguments.stride)
    else:
        stride = 1
    reader = LmdbReader(path)
    patches = reader.extractRandomPatches((rfSize, rfSize, 3), numPatchesByImage, stride)
    

#    data = reader.next()
#    size = reader.info()['entries']
#    patches = np.zeros((size*numPatchesByImage, rfSize*rfSize*3))
#    r = np.random.randint(0, imageDim[0] - rfSize, (numPatchesByImage*size,))
#    c = np.random.randint(0, imageDim[1] - rfSize, (numPatchesByImage*size,))

#    for i in range(0, size*numPatchesByImage):
#        print i        
#        trainX = np.transpose(data[1], (1,2,0))
#        if i % 1000 == 0:
#            print "Extracted {0}/{1} patches.".format(i, size*numPatchesByImage)
       
#        patch = trainX
#        patch = patch[r[i]:r[i]+rfSize,c[i]:c[i]+rfSize,:]
#        patches[i,:] = np.reshape(patch, (rfSize*rfSize*3,), 'F')
#        data = reader.next()

    
    #patch normalization
    patchesMean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
    patchesVar = np.var(patches, axis=1, dtype=np.float32, keepdims=True)
    offsetMatrix = 10.0 * np.ones(patchesVar.shape)
    patches = (patches - patchesMean) / np.sqrt(patchesVar + offsetMatrix)

    if whitening:
        C = np.cov(patches, y=None, rowvar=0, ddof=1).T
        M = np.mean(patches, axis=0, dtype=np.float32, keepdims=False)
        W, V = np.linalg.eig(C)
        P = np.dot(np.dot(V, np.diag(np.diag(np.sqrt(1./(np.diagflat(W) + 0.1))))), V.T)
        patches = np.dot((patches - M), P)

    estimator = KMeans(numCentroids, maxIter, batchSize)
    remain = patches.shape[0] % batchSize
    patches = patches[:patches.shape[0] - remain:]
    print patches.shape
    init, centroids = estimator.fit(patches)


#   estimator = MiniBatchKMeans(n_clusters=numCentroids, init='k-means++', max_iter=50, batch_size=1000, verbose=1, compute_labels=True, random_state=None, tol=0.0, max_no_improvement=10, init_size=None, n_init=10,     reassignment_ratio=0.5)
    
    np.save('centroids_windowsize_{0}'.format(rfSize), centroids)
    np.save('initialCentroids_windowsize_{0}'.format(rfSize), init)
    np.save('mean_windowsize_{0}'.format(rfSize), M)
    np.save('eigenvectors_windowsize_{0}'.format(rfSize), P)
    




