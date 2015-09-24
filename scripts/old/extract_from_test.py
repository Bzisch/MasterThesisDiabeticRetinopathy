import sys
from caffe.io import load_image
from skimage.util import view_as_windows
import os
import numpy as np
from kMeansFeatureExtractor import *

from lmdbWriter import open_csv

imageDim = (512,512,3)
rfSize = 16
numPatches = 50
images = os.listdir('../data/resized/trainOriginal')

labels_dict = open_csv('../data/trainLabels.csv')
values = labels_dict.values()
total_numPatches = values.count(0)*40 + (values.count(1) + values.count(2) + values.count(3) + values.count(4)) * 140
patches = np.zeros((total_numPatches, rfSize*rfSize*3))
whitening = True
maxIter = 50
batchSize = 1000
j = 0

values = labels_dict.values()

for each in images:
    if labels_dict[each.split('.')[0]] > 0:
        numPatches = 140
    else:
        numPatches = 40

    img = load_image('../data/resized/trainOriginal/' + each)
    windows = view_as_windows(img, (rfSize, rfSize, 3))
    r = np.random.randint(0, windows.shape[0] - windows.shape[3], (numPatches,))
    c = np.random.randint(0, windows.shape[1]- windows.shape[4], (numPatches,))
    for i in range(0, numPatches):
        patch = np.reshape(windows[r[i],c[i],0,:], windows.shape[3]*windows.shape[4]*windows.shape[5])
        patches[j,:] = patch
        if j % 100 == 0:
            print "Extracted {0}th patch of {1} patches totally".format(j,total_numPatches)        
        j += 1

numCentroids = int(np.sqrt(total_numPatches/2))

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
init, centroids = estimator.fit(patches)

np.save('train_centroids_windowsize_{0}'.format(rfSize), centroids)
np.save('train_initialCentroids_windowsize_{0}'.format(rfSize), init)
np.save('train_mean_windowsize_{0}'.format(rfSize), M)
np.save('train_eigenvectors_windowsize_{0}'.format(rfSize), P)

