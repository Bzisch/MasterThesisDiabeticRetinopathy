from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.externals import joblib
import random



            
def duplicate_classes(trainX, trainY, num, class_0, class_1):
    
    counts = [len(list(np.where(trainY == i)[0])) for i in range(0,5)]
    print "0s : {0}\n1s : {1}\n2s : {2}\n3s : {3}\n4s : {4}\n".format(counts[0] , counts[1], counts[2], counts[3], counts[4])            
    
    indexes = list(np.where(trainY == class_1)[0])
    
    print len(indexes)
    newX = np.zeros((num, trainX.shape[1]))
    newY = np.zeros((num, 1))
    zeros = 0
    ones = 0
    i = 0
    
    count = len(indexes)
    print count 
    while(zeros < num - count  or ones < count):
        
        if len(indexes) > 0:
            random_index = indexes.pop()           
        else:
            random_index = random.randint(0, trainX.shape[0] - 1)
        
        label = int(trainY[random_index, 0])               
        if label == class_1 and ones < count:
    
            newX[i,:] = trainX[random_index,:]
            newY[i,0] = 1
            ones += 1
            i += 1

        elif label == class_0 and zeros < num - count:
            newX[i,:] = trainX[random_index,:]
            newY[i,0] = 0
            zeros += 1
            i += 1
        
    from sklearn.utils import shuffle
    newX, newY  = shuffle(newX, newY)
    from sklearn.feature_selection import SelectKBest, chi2
    X_new = SelectKBest(chi2, k=500).fit_transform(newX, newY)
    return X_new, newY



if __name__ == '__main__':
    import os
    x_1 = np.load('features/test_features_windowsize_6.npy')
    x_2 = np.load('features/test_features_windowsize_8.npy')
    x_3 = np.load('features/test_features_windowsize_16.npy')
    x_4 = np.load('features/train_features_windowsize_16.npy')
    trainY = np.load('features/trainY_corrected.npy')
    
#  
    print "Feature vector dimensions: {0}".format(np.hstack((x_1, x_2, x_3, x_4)).shape[1])
    
    trainXC, y_train = duplicate_classes(np.hstack((x_1,x_2,x_3,x_4)), trainY, 2000, 0, 4)

    trainXC_mean = np.mean(trainXC, axis=0, dtype=np.float32, keepdims=False)
    trainXC_sd = np.sqrt(np.var(trainXC, axis=0, dtype=np.float32, ddof=1, keepdims=False)+0.01)
    trainXCs = (trainXC - trainXC_mean) / trainXC_sd
    x_train, x_test, y_train, y_test = train_test_split(trainXCs, y_train, test_size=0.25)
    
    classifier = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=100.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight='auto', verbose=1)
    classifier.fit(x_train, y_train)
#    joblib.dump(classifier, 'svm_incomplete_classifier.pkl')
    y_train_predicted = classifier.predict(x_train)
    y_test_predicted = classifier.predict(x_test)

    
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    train_accuracy = accuracy_score(y_train, y_train_predicted)
    test_accuracy = accuracy_score(y_test, y_test_predicted)

    print 'Train accuracy = ' + str(train_accuracy)
    print 'Test accuracy = ' + str(test_accuracy)

    
    print 'TRAIN: Classification report'
    print classification_report(y_train, y_train_predicted)
    print confusion_matrix(y_train, y_train_predicted)
    
    print 'TEST: Classification report'
    print classification_report(y_test, y_test_predicted) 
    print confusion_matrix(y_test, y_test_predicted)
   
    
    #print confusion_matrix(old_samples, prediction)
   # from time import sleep
   # for i in range(0, len(old_samples)):
   #     print old_samples[i], decision[i], prediction[i]
   #     sleep(1)
    #from lmdbWriter import open_csv
    #data_dict = open_csv('../data/trainLabels.csv')
    #files = os.listdir('../data/resized/train512')

    #full_classifier = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=100.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=1,    random_state=None)
    #full_classifier.fit(trainXCs, trainY)
#    joblib.dump(full_classifier, 'svm_classifier.pkl')

    #y_full_predicted = full_classifier.predict(trainXCs)
    #full_accuracy = accuracy_score(trainY, y_full_predicted)
    #print 'FULL TRAIN = ' + str(full_accuracy)

    #print 'FULL Classification report'
    #print classification_report(trainY, y_full_predicted)

    #print confusion_matrix(trainY, y_full_predicted)
    

