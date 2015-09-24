from sklearn.feature_selection import VarianceThreshold
from skimage.exposure import equalize_adapthist
from skimage.util import view_as_windows, view_as_blocks
from skimage.data import load
from scipy.signal import medfilt
import numpy as np
import matplotlib.image as mpimg
import matplotlib.cm as cm
from skimage.draw import circle_perimeter
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
import csv
import time
import os
from ..distance.distance import cdist as native_cdist


#calculates entropy of an image
#used as a feature for dataset clustering
def calc_entropy(img):
    from matplotlib.pylab import hist
    from scipy.stats import entropy
    hist, bins = np.histogram(img.ravel(), 256, [0,256])
    distribution = hist.astype(np.float32)/bins.shape[0]
    return entropy(distribution)


#clusters images into k clusters
def cluster_images(path, k, batch_size):
    """
    :param path: path to a folder with image files
    :return:
    """
    dir = os.listdir(path)
    images = np.zeros((len(dir), 3))

    for i, imgname in enumerate(dir):
        img = load(path + imgname)
        images[i, 0] = np.mean(img)
        images[i, 1] = np.var(img)
        images[i, 2] = calc_entropy(img)
        print str(i) + "/" + str(len(dir))


    estimator = MiniBatchKMeans(n_clusters=k, verbose=True, batch_size=batch_size)
    estimator.fit(images)
    from sklearn.externals import joblib
    joblib.dump(estimator, 'estimator.pkl')
    np.save('data.npy', images)


#select centroids with variance higher than average
def select_centroids(centroids):
    """
    :param centroids: learned centroids
    :return: new_centroids: (without centroids with variance < avg_variance(centroids))
    """
    sel = VarianceThreshold(threshold=np.var(centroids))
    new_centroids = sel.fit_transform(centroids.T)
    new_centroids = new_centroids.T
    return new_centroids


#create centroids image
#quadratic grid of centroids
def centroids_to_image(centroids, image_size, rfsize, g=0):
    """

    :param image_size: image is nxn
    :param rfsize: receptive field size
    :param g:
    :return: .png image with centroid images in a grid
    """
    margin = image_size / rfsize
    horizontal = []
    for i in xrange(margin):
        for j in xrange(margin):
            if j == 0:
                m_img = np.reshape(centroids[i*margin + j,:], (rfsize, rfsize))
            else:
                m_img = np.hstack((m_img, np.reshape(centroids[i*margin + j,:], (rfsize, rfsize))))
        horizontal.append(m_img)

    img = horizontal[0]
    for m_img in horizontal[1:]:
        img = np.vstack((img, m_img))

    mpimg.imsave('centroids_{0}{1}.png'.format(rfsize, g), img, cmap=cm.Greys_r)

#extract patches from an image
def extract_patches(path, numPatchesPerImage, patchSize):

    """
    :param path: path to a RGB fundus image
    :param numPatchesPerImage: number of patches to extract per image
    :param patchSize: patch is nxn size
    :return: patches: matrix with an image patch in each row
    """

    img = load(path)
    img = img[:,:,1]
    #contrast enhancemenet
    img = equalize_adapthist(img)
    windows = view_as_windows(img, (patchSize,patchSize))
    j = 0
    patches = np.zeros((numPatchesPerImage, patchSize*patchSize))
    while(j < numPatchesPerImage):
        
        sx = np.random.randint(0, windows.shape[0] - 1)  
        sy = np.random.randint(0, windows.shape[0] - 1)
        x = (patchSize/2 - 1) + sx
        y = (patchSize/2 - 1) + sy
        r = (img.shape[0]/2) - 1

        if np.sqrt((x - r) ** 2 + (y - r) **2 ) < r:
            patch = windows[sx, sy, :].flatten()            
            patches[j,:] = patch
            j += 1 
        else:
            if j > 0:
                j -= 1 
    
    return patches


#retina shape can be approximated with a circle of a radius = img.height/2 - 1
def get_perimeter(img):
    """
    :param img:
    :return:
    """
    cx = img.shape[0]/2
    cc, rr = circle_perimeter(cx - 1, cx - 1, 256)
    return rr, cc

#parallelized function for patch extraction
#sets the pool size to the number of cpus available
#actually usable only locally. not really portable to work
def extract(rfSize, path):
    """
    :param rfSize: receptive field size
    :param path:
    :return:
    """
    nums = {0:1, 1:5, 2:2, 3:10, 4:10}
    #labels_dict = open_csv()
    #currently a dummy variable, should contain labels of train examples
    labels_dict = {}

    for j in range(0, 5):
        if j == 0:
            images = labels_dict[j][:10000]

        numOfPatches = nums[j]
#        path = '/home/jbzik/Documents/Diplomski_Bzik/jbzik_kaggle_data/data/resized/trainOriginal/'

        patches = np.zeros((len(images)*numOfPatches, rfSize*rfSize))
        
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = [pool.apply_async(extract_patches, args=(path + images[i] + '.jpeg', numOfPatches, rfSize,)) for i in xrange(len(images))]
        

        for i in xrange(len(images)):
            patches[i*numOfPatches: (i*numOfPatches) + numOfPatches,:] = np.array(results[i].get())

            if i % 10000 == 0:
                print "Extracting {0}/{1} patch".format(i*nums[j], nums[j]*len(images))

        print "Finished"
        patches = patches[~np.isnan(patches).any(axis=1)]
        
        np.save('core/patches_{0}_{1}.npy'.format(j, rfSize), patches)


#@profile
def kmeans(rfsize):
    """
    :param rfsize: receptive field size
    :return: centroids (patches that represent some cluster of similar patches)
    """

    ###########################################################
    #in this part patches should be generated or read from a file to the variable patches
    #my code is not portable
    #it's tied to files that were generated locally
    patches0 = np.load('core/patches_0_{0}.npy'.format(rfsize))
    patches1 = np.load('core/patches_1_{0}.npy'.format(rfsize))
    patches = np.vstack((patches0, patches1))
    patches0 = np.load('core/patches_2_{0}.npy'.format(rfsize))
    patches1 = np.load('core/patches_3_{0}.npy'.format(rfsize))
    patches = np.vstack((patches, patches0, patches1))
    patches0 = np.load('core/patches_4_{0}.npy'.format(rfsize))
    patches = np.vstack((patches, patches0))
    ############################################################


    #data normalization (standardization)
    p_mean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
    p_var = np.var(patches, axis=1, dtype=np.float32, keepdims=True)
    off_matrix = 10.0 * np.ones(p_var.shape)
    patches = (patches - p_mean) / np.sqrt(p_var + off_matrix)

    #data whitening
    covariance_matrix = np.cov(patches, y=None, rowvar=0, ddof=1).T
    mean = np.mean(patches, axis=0, dtype=np.float32, keepdims=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    U = np.dot(np.dot(eigenvectors, np.diag(np.diag(np.sqrt(1./(np.diagflat(eigenvalues) + 0.1))))), eigenvectors.T)

    #load whitening parameters that were saved earlier
    #M = np.load('train_mean_windowsize.npy')
    #P = np.load('train_eigenvectors_windowsize.npy')
    patches = np.dot((patches - mean), U)


    #save whitening parameters for use later
    np.save('core/train_mean_windowsize', mean)
    np.save('core/train_eigenvectors_windowsize', U)

    #set n_clusters and estimate centroids using minibatch kmeans
    #https://algorithmicthoughts.wordpress.com/2013/07/26/machine-learning-mini-batch-k-means/
    n_clusters = int(np.sqrt(patches.shape[0]/2))
    estimator = MiniBatchKMeans(n_clusters=n_clusters, verbose=True, batch_size=1000, compute_labels=False)
    estimator.fit(patches)

    #save centroids
    np.save('core/centroids.npy', estimator.cluster_centers_)

def pool(q):
    """
    :param q:
    :return:
    """
    return np.array([np.sum(np.sum(q, axis=1), axis=0)])

#quadrant pooling
def pool_quadrant(patches, rooti, rootj, i, j, iz, jw):
    """
    :param patches:
    :param rooti:
    :param rootj:
    :param i:
    :param j:
    :param iz:
    :param jw:
    :return:
    """
    q1 = pool(patches[rooti:rooti+i, rootj:rootj+j, :])
    #q2 = pool(patches[rooti+i:iz, rootj:rootj+j, :])
    q3 = pool(patches[rooti:rooti+i, rootj+j:jw, :])
    #q4 = pool(patches[rooti+i:iz, rootj+j:jw, :])

    q = np.vstack((q1,q3)).flatten()
    return q


#generate features for an image based on learned centroids
def extract_features_img(path, centroids, rfSize, M, U, stride, normal_pooling=True):
    """
    :param path: path to RGB retina image
    :param centroids: learned centroids
    :param rfSize: receptive field size
    :param M: whitening parameter
    :param P: whitening parameter
    :param stride: parameter that defines the density of windows that are extracted from an image
    :param normal_pooling: if true:
                               divide in 4 regions and pool each one
                           else: divide in 16 regions and pool each one
    :return:feature_vector
    """

    img = load(path)
    try:
        img = img[:,:,1]
    except:
        return None

    #contrast enhancing
    img = equalize_adapthist(img)
    numFeats = img.shape[0] * img.shape[1]
    numCentroids = centroids.shape[0]

    #extract dense patches with predefined stride
    #smaller the stride, slower the function
    windows = view_as_windows(img, (rfSize, rfSize), stride)
    patches = np.reshape(windows, (windows.shape[0]*windows.shape[1], rfSize*rfSize))

    #data normalization
    p_mean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
    p_var = np.var(patches, axis=1, dtype=np.float32, ddof=1, keepdims=True)
    off_matrix = 10.0 * np.ones(p_var.shape)
    patches = (patches - p_mean) / np.sqrt(p_var + off_matrix)
    
    patches = np.dot((patches - M), U)
    
    #calculate distance from all patches to all centroids
    z = native_cdist(patches, centroids)
    
    #mean distance from each patch to all centroids
    #triangle activation function
    mu = np.tile(np.array([np.mean(z, axis = 1)]).T, (1, centroids.shape[0]))
    patches = np.maximum(mu - z, np.zeros(mu.shape))

    rows = (img.shape[0] - rfSize + stride)/stride
    columns = (img.shape[1] - rfSize + stride)/stride
        
    patches = np.reshape(patches, (rows, columns, numCentroids))

    #starting points
    #central point # of the patches "image"
    halfr = np.round(float(rows)/2)
    halfc = np.round(float(columns)/2)

    #pool quadrants
    if normal_pooling:       
        q1 = np.array([np.sum(np.sum(patches[0:halfc, 0:halfr, :], axis = 1),axis = 0)])
        q2 = np.array([np.sum(np.sum(patches[halfc:patches.shape[0], 0:halfr, :], axis = 1),axis = 0)])
        q3 = np.array([np.sum(np.sum(patches[0:halfc, halfr:patches.shape[1], :], axis = 1),axis = 0)])
        q4 = np.array([np.sum(np.sum(patches[halfc:patches.shape[0], halfr:patches.shape[1], :], axis = 1),axis =     0)])
        feature_vector = np.vstack((q1,q2,q3,q4)).flatten()

    else:
        
        quartr = np.round(float(rows)/4)
        quartc = np.round(float(columns)/2)
        q1 = pool_quadrant(patches, 0, 0, quartc, quartr, halfc, halfr)        
        q2 = pool_quadrant(patches, halfc, 0, quartc, quartr, patches.shape[0], halfr) 
        q3 = pool_quadrant(patches, 0, halfr, quartc, quartr, halfc, patches.shape[1])
        q4 = pool_quadrant(patches, halfc, halfr, quartc, quartr, patches.shape[0], patches.shape[1])
        feature_vector = np.vstack((q1, q2, q3, q4)).flatten()

               
    return feature_vector



#function is not really usable
#extracts features from all training images
def extract_features_all(rfSize, stride=False, train=True):
    images_dict = []
    if train:
        path = ''
    else:
        path =''
        images = []
        labels = []

        for folder in os.listdir(path):
            if 'train_sm' in folder:
                temp = os.listdir(path + folder)
                images += [path + folder + '/' + c for c in temp if c.split('.')[0] not in images_dict[0]]
        
                for l in temp:
                    label = l.split('.')[0]
                    if label in images_dict[1]:
                        labels.append(1)
                    elif label in images_dict[2]:
                        labels.append(2)
                    elif label in images_dict[3]:
                        labels.append(3)
                    elif label in images_dict[4]:
                        labels.append(4)
    
    if stride:
        stride = stride
    else:
        stride = rfSize/2

    centroids = np.load('core/{0}vs512/centroids16_selected.npy'.format(rfSize))
    M = np.load('core/{0}vs512/train_mean_windowsize.npy'.format(rfSize))
    P = np.load('core/{0}vs512/train_eigenvectors_windowsize.npy'.format(rfSize))
    
    if train:
        numOfExamples = 0
        images = []
        for i in xrange(5):
            numOfExamples += len(images_dict[i])
            images += images_dict[i]
    else:
#        images = os.listdir(path)
        numOfExamples = len(images)
    
    X = np.zeros((numOfExamples, centroids.shape[0]*8))
        
    for i in xrange(len(images)):
        s = time.time()
        if train:
            X[i,:] = extract_features_img(path + images[i] + '.jpeg', centroids, rfSize, M, P, stride, False)
        else:
            print images[i]
            X[i,:] = extract_features_img(images[i], centroids, rfSize, M, P, stride, False)
      

        #print X[i,:]
        print X[i,:].shape
        end = time.time()
        print "Sample {0}/{1}".format(i, len(images))
        print images[i]
        print end - s

    if train: 
        np.save('train_features_{0}_dense.npy'.format(rfSize), X)
    else:
        np.save('additional_features_{0}.npy'.format(rfSize), X)
        np.save('additional_labels.npy', np.array(labels))


##########################################################################
#Not sure if this part with second layer features works
#It should be the same process again
#It resulted with a mild increase of precision and recall of the final model
def extract_second_layer_centroids(features):
    
    patches = np.zeros((features.shape[0]*100, 100))

    for i in xrange(features.shape[0]):
        for j in xrange(50):
            patch = np.random.choice(features[i,:], 100)
            print patch
            patches[i*50 + j,] = patch
            print "{0}/{1}".format(i*50+j, features.shape[0]*50) 
    
    p_mean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
    p_var = np.var(patches, axis=1, dtype=np.float32, ddof=1, keepdims=True)
    off_matrix = 10.0 * np.ones(p_var.shape)
    patches = (patches - p_mean) / np.sqrt(p_var+ off_matrix)

    numCentroids = int(np.sqrt(patches.shape[0]/2))
    estimator = MiniBatchKMeans(numCentroids, verbose=True, batch_size=1000, compute_labels=False)
    patches = patches[~np.isnan(patches).any(axis=1)]
    estimator.fit(patches)


    np.save('second_layer_centroids.npy', estimator.cluster_centers_)

def extract_second_layer_features(centroids, features):
   
    s_features = np.zeros((features.shape[0], 4*centroids.shape[0]))

    for i in xrange(features.shape[0]):
        
        patches = np.zeros((features.shape[1]/100 + 1, 100))
        for j in range(0, features.shape[1], 100):
            if features[i, j:j+100].shape[0] == 100:
                patches[j/100,:] = features[i, j:j+100]
                
        patchesMean = np.mean(patches, axis=1, dtype=np.float32, keepdims=True)
        patchesVar = np.var(patches, axis=1, dtype=np.float32, keepdims=True)
        offsetMatrix = 10.0 * np.ones(patchesVar.shape)
        patches = (patches - patchesMean) / np.sqrt(patchesVar + offsetMatrix)

        z = native_cdist(patches, centroids)
        
        mu = np.tile(np.array([np.mean(z, axis = 1)]).T, (1, centroids.shape[0]))
        patches = np.maximum(mu - z, np.zeros(mu.shape))
        
        q = patches.shape[0]/4
        q1 = np.sum(patches[0:q,:], axis=0)       

        q_all = q1
        for j in range(1,4):
            q_all = np.hstack((q_all,np.sum(patches[j*q:(j+1)*q,:], axis=0)))

        s_features[i,:] = q_all

    np.save('second_layer_additional_features.npy', s_features)