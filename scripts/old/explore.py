import numpy as np
from lmdbWriter import open_csv
import matplotlib.image as mpimg
import os

labels = open_csv('../data/trainLabels.csv')
values = np.array(labels.values())
data = []

for i in range(0,5):
    data.append(np.zeros((values[values == i].shape[0], 512*512)))

print len(data)

class_counters = [0]*5
i = 0
for each in os.listdir('../data/resized/trainOriginal/'):
    if each.split('.jpeg')[0] in labels.keys():
        if values[i] > 0:
            class_counters[values[i]] += 1
            i += 1
            print class_counters
            continue
        

        data[values[i]][class_counters[values[i]], :] = mpimg.imread('../data/resized/train512/' + each)[:,:,2].flatten()
        class_counters[values[i]] += 1
        i += 1
        print class_counters

np.save('separated_data/class_0.npy', data[0])

#for i in range(1,5):
#    np.save('separated_data/class_{0}'.format(i), data[i])
