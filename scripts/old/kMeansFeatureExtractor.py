__author__ = 'jan'

import numpy as np
from sklearn import grid_search
from sklearn.svm import LinearSVC
import sys
import scipy
import scipy.io as io
import time

class KMeans():


    def __init__(self, nClusters, maxIter, batchSize):

        self.nClusters = int(nClusters)
        self.maxIter = int(maxIter)
        self.batchSize = int(batchSize)


    def fit(self, X):

        x_2 = np.sum(np.square(X),axis = 1)
        centroids = np.random.randn(self.nClusters, X.shape[1])*0.1
        initialCentroids = np.copy(centroids)
        print X.shape
        for iteration in range(self.maxIter):

            loss = 0.0

            print "Iteration number: {0}/{1}".format(iteration, self.maxIter)
            
            start = time.time()
            c_2 = 0.5 * np.sum(np.square(centroids), axis=1)
            summation = np.zeros((self.nClusters, X.shape[1]))
            counts = np.zeros((self.nClusters,))
            end = time.time()
            print "SUmmation time {0}".format(end - start)
            
            for i in xrange(0,X.shape[0],self.batchSize):
                lastIndex=min(i+self.batchSize, X.shape[0])
                m = lastIndex - i
                val = np.amax(np.dot(centroids, X[i:lastIndex, :].T) - np.array(np.tile(np.array([c_2]).T, (1, self.batchSize))), axis = 0)
                labels = np.argmax(np.dot(centroids, X[i:lastIndex, :].T) - np.array(np.tile(np.array([c_2]).T, (1, self.batchSize))), axis = 0)
                loss = loss + np.sum(0.5*x_2[i:lastIndex] - val, axis = 0)

                # create a sparse matrix with ones denoting membership to a cluster
                rows = range(m)
                columns = labels
                data = np.ones((len(rows),))
                S = scipy.sparse.coo_matrix((data, (rows, columns)), shape=(self.batchSize, self.nClusters))

                summation = summation + S.T.dot(X[i:lastIndex, :])
                counts = counts + S.sum(axis = 0)
            
            start = time.time()
            centroids = np.array(summation/np.tile(counts.T, (1,X.shape[1])))
            #just zap empty centroids so they don't introduce NaNs everywhere.
            badIndex = np.where(counts == 0)[1]
            centroids[badIndex, :] = np.zeros((centroids.shape[1],))
            end = time.time()
            print "zaping time {0}".format(end - start)
            print "Current loss:{0}".format(loss)

    return initialCentroids, centroids





