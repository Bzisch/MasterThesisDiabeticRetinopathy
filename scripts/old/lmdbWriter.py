from skimage.util import view_as_windows
import lmdb
import os
import re
import csv
import numpy as np
import caffe
import multiprocessing as mp
import caffe.io
from caffe.proto import caffe_pb2


        
def open_csv( path):
    labels_dict = {}
    with open(path,'rb') as f:
        reader = csv.reader(f)    
        for row in reader:
            labels_dict[row[0]] = float(row[1])

    return labels_dict

size = 256


train_csv = '../../data/data/training_0/training.csv'
test_csv = '../../data/data/training_0/testing.csv'
train_data = open_csv(train_csv)
test_data = open_csv(test_csv)

train_lmdb = lmdb.open('/home/nbanic/hardDrive/jbzik_kaggle_data/data/train_lmdb{0}'.format(size), map_size= len(train_data.keys()) * np.zeros((size, size)).nbytes

test_lmdb = lmdb.open('/home/nbanic/hardDrive/jbzik_kaggle_data/data/test_lmdb{0}'.format(size), map_size= len(test_data.keys()) * np.zeros((size, size)).nbytes

def preprocess(image):
    pass


def write(img_name,  i):
    
    label = train_data[img_name]
    image = caffe.io.load_image(img_name)
    image = preprocess(image)
    im_dat = caffe.io.array_to_datum(image.astype(float))
    im_dat.label = label
     
    with lmdb.begin(write=True) as in_txn:
        in_txn.put('{:08}'.format(i), im_dat.SerializeToString())
        print "Saving {0}/{1} images.".format(i, 140000)

    return i

if __name__ == '__main__':

    from joblib import Parallel, delayed
    
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply(write, args=(img_name, i)) for i, img_name in enumerate(train_data.keys())]
    
    


