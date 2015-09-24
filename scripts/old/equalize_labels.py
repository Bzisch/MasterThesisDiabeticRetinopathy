from lmdbWriter import open_csv
import numpy as np
labels_dict = open_csv('../data/trainLabels.csv')

keys = []
for key in labels_dict.keys():
    s = key.split('_')
    keys.append(s[0])

keys = list(set(keys))


for key in keys:

    if labels_dict[ key + '_right'] > labels_dict[ key + '_left']:
        labels_dict[key + '_left'] = labels_dict[key + '_right']

    elif labels_dict[key + '_left'] > labels_dict[ key + '_right']:
        labels_dict[key + '_right'] = labels_dict[key + '_left']

trainY = np.zeros((len(labels_dict.keys()), 1))

for i, key in enumerate(labels_dict.keys()):
    trainY[i, 0] = labels_dict[key]


np.save('trainY_corrected.npy', trainY) 
