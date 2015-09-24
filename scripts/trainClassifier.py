##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################
################################COLLECTION OF FUNCTIONS THAT WILL BENEFIT NO ONE##########################
##########################################################################################################
##########################################################################################################
##########################################################################################################
##########################################################################################################

#from core.main_lib import open_csv
import numpy as np
from sklearn.cross_validation import *
from sklearn.metrics import *
from sklearn.svm import *
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.utils import shuffle
from sklearn.externals import joblib
from itertools import combinations
from sklearn.linear_model import LinearRegression

#normalization function
def normalize(data):
    mean = np.mean(data, axis = 0, dtype=np.float32, keepdims=False)
    sd = np.sqrt(np.var(data, axis = 0, dtype = np.float32, ddof=1, keepdims=False)+0.01)
    return (data - mean) / sd

#custom function for label creation
def make_labels(labels, i):

    if i == 4:
        return [i]*len(labels[i])
    else:
        return [i]*len(labels[i]) + make_labels(labels, i+1)


def create_data(class_0, class_1, numFeatures, all=False):

    if all:
        x0 = np.load('features16vs512/features_{0}.npy'.format(class_0))
        x1 = np.load('features16vs512/features_{0}.npy'.format(class_1))
        for i in range(class_1, 5):
            x1 = np.vstack((x1 , np.load('features16vs512/features_{0}.npy'.format(i))))

    elif class_0 == 0:

        x1 = np.load('features16vs512/features_{0}.npy'.format(class_1))
        x0 = np.load('features16vs512/features_{0}.npy'.format(class_0))
        x0 = x0[np.random.randint(x0.shape[0], size=int(5*x1.shape[0])),:]
    else:
        x1 = np.load('features16vs512/features_{0}.npy'.format(class_1))
        x0 = np.load('features16vs512/features_{0}.npy'.format(class_0))
    
    print "{0} vs {1}".format(class_0, class_1)
    print x0.shape, x1.shape
    X = np.vstack((x0,x1))
    y0 = np.zeros((x0.shape[0],))
    y1 = np.ones((x1.shape[0],))
    Y = np.concatenate((y0, y1))

    indices = list(np.where(np.isnan(X).any(axis=1) == True)[0])
    X = X[~np.isnan(X).any(axis=1)]
    Y = np.delete(Y, indices)

    X, Y = shuffle(X, Y)
    selector = SelectKBest(chi2, k=numFeatures)
    X = selector.fit_transform(X,Y)
    X = normalize(X)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.15)

    return trainX, testX, trainY, testY, selector


def train_and_test(trainX, testX, trainY, testY):

    classifier = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=100.0,  multi_class='ovr',fit_intercept=True, intercept_scaling=1, class_weight='auto', verbose=1) 
    classifier.fit(trainX, trainY)

    y_predicted = classifier.predict(testX)

    print accuracy_score(testY, y_predicted)
    print classification_report(testY, y_predicted)
    print confusion_matrix(testY, y_predicted)
    return classifier


def round_predictions(y_p):

    y_round = np.zeros((len(y_p),))
    for i in xrange(len(y_p)):                                                                   

        if y_p[i] < 0.76:
            y_round[i] = 0
        elif y_p[i] >= 0.76 and y_p[i] < 1.16:
            y_round[i] = 1
        elif y_p[i] >= 1.16 and y_p[i] < 1.7:
            y_round[i] = 2
        elif y_p[i] >= 1.7 and y_p[i] < 2.2:
            y_round[i] = 3
        elif y_p[i] >= 2.2:
            y_round[i] = 4
    return y_round

def test_train(clf, x, y, sel=False):

    indices = list(np.where(np.isnan(x).any(axis=1)))
    x = x[~np.isnan(x).any(axis=1)]
    y = np.delete(y, indices)
    if sel: 
        x = sel.transform(x)

    x = normalize(x)
    x, y = shuffle(x,y)
    y_pred = clf.predict(x)
    print accuracy_score(y, y_pred)
    print confusion_matrix(y, y_pred)
    print classification_report(y, y_pred)


def train(clf, x, y, sel=False, test=True):
    
    indices = list(np.where(np.isnan(x).any(axis=1)))
    x = x[~np.isnan(x).any(axis=1)]
    y = np.delete(y, indices)

    if sel:
        x = sel.fit_transform(x,y)

    x =  normalize(x)
    x, y  = shuffle(x,y)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

    if test:
        print x_train.shape
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        print accuracy_score(y_test, y_pred)
        print confusion_matrix(y_test, y_pred)
        print classification_report(y_test, y_pred)

        y_t_pred = clf.predict(x_train)
        print accuracy_score(y_train, y_t_pred)
        print confusion_matrix(y_train, y_t_pred)
        print classification_report(y_train, y_t_pred)

    else:
        clf.fit(x,y)
        y_pred = clf.predict(x)
        print accuracy_score(y, y_pred)
        print confusion_matrix(y, y_pred)
        print classification_report(y, y_pred)

    joblib.dump(clf, 'linear_svm_{0}0.pkl'.format(x.shape[1]))
    if sel:
        joblib.dump(sel, 'sel_{0}.pkl'.format(sel.k))

def test(clf, x, sel=False):

    indices = list(np.where(np.isnan(x).any(axis=1)))
    for i in indices:
        x[i] = x[i+1]
    x = normalize(x)

    if sel:
        x = sel.transform(x)
    y = clf.predict(x)
    import os
    names = os.listdir('')
    #clf.fit(x_train,y_train)
    y_p = clf.predict(x)
    import csv 
    with open('submission_{0}.csv'.format(x.shape[1]), 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'level'])
        for i in xrange(len(y_p)):
            img = names[i].split('.')[0]
            writer.writerow([img,int(y_p[i])])


if __name__ == '__main__':
    clf = LinearSVC(penalty='l2', loss='l2', dual=True, tol=0.0001, C=0.1,  multi_class='ovr',fit_intercept=True, intercept_scaling=1, verbose=1, max_iter=1000, class_weight='auto')
#    sel = SelectKBest(chi2, k=5000)
    #sel = joblib.load('linear_multilayer_100000/sel_100000.pkl')
    clf = joblib.load('linear_svm_99160.pkl')
    images = open_csv()

    y_additional = np.load('additional_labels.npy')
    features_additional_layer_1 = np.load('additional_features_16.npy')
    features_additional_layer_2 = np.load('second_layer_additional_features.npy')
    features_add = np.hstack((features_additional_layer_1, features_additional_layer_2))
    
    y = make_labels(open_csv(), 0)    
    features_layer1 = np.load('16vs512features/test_features_16_selected_dense.npy')
    features_layer2 = np.load('16vs512features/second_layer_test_features.npy')
    features = np.hstack((features_layer1, features_layer2))
    
    y_real = np.hstack((y, y_additional))
#    features = np.vstack((features, features_add))

#    train(clf, features, y_real, False, True)
    test(clf, features, False)
    #test_train(clf, features, y_real)
