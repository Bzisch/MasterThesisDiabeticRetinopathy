__author__ = 'jan'
import lmdb
import time
import caffe
from caffe.proto import caffe_pb2
import numpy as np
from skimage.util import view_as_windows
import matplotlib.image as mpimp
from sklearn.feature_extraction import image

class LmdbReader():


    def __init__(self, path):

	self.lmdb_env = lmdb.open(path)
        self.cursor = self.__set_cursor(path)
        self.datum = caffe_pb2.Datum()


    def __set_cursor(self, path):

        lmdb_txn = self.lmdb_env.begin()
        return lmdb_txn.cursor()

    
    def next(self):

        if self.cursor.next():

            key, value = self.cursor.item()
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            image = caffe.io.datum_to_array(datum)            
            label = datum.label
            return (label, image)

        else:
            return False


    def info(self):
        return self.lmdb_env.stat()

    
    def extractRandomPatches(self, patchSize, numPatches, stride=1):
        size = self.info()['entries']
        data = self.next()
        patches = np.zeros((numPatches*size, patchSize[0]*patchSize[1]*patchSize[2]))

        j = 0
        for z in range(0, size):                               
            start = time.time()
            img = np.transpose(data[1], (1,2,0))               
            windows = view_as_windows(img, patchSize, stride)
            r = np.random.randint(0, windows.shape[0] - windows.shape[3], (numPatches,))
            c = np.random.randint(0, windows.shape[1]- windows.shape[4], (numPatches,))
            for i in range(0, numPatches):
                patch = np.reshape(windows[r[i],c[i],0,:], windows.shape[3]*windows.shape[4]*windows.shape[5])                    
                patches[j,:] = patch
                if j % 100 == 0:
                    print "Extracted {0}th patch of {1} patches totally".format(j, numPatches*size)
                j += 1
            data = self.next()
            end = time.time()
            print (end - start)
        
        return  patches



