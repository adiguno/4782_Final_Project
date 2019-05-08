import sklearn as sk 
from imblearn.over_sampling import SMOTE
import numpy as np 
import pickle
from collections import Counter
from sklearn.datasets import make_classification


if __name__ == "__main__":
    with open('features_labels.pkl', 'rb') as f:  
        features, labels = pickle.load(f)

    for x in np.nditer(labels, op_flags=['readwrite']):
        if x < 37:
            x[...] = 1
        else: 
            x[...] = 0
    labels = labels.astype(int)
    # print(labels.shape)
    labels = labels.flatten()
    # print(labels.shape)

    # print(features.shape)
    print('Original dataset shape %s' % Counter(labels))
    sm = SMOTE(random_state=42)
    x_res, y_res = sm.fit_resample(features, labels)
    # print(x_res.shape)
    print('SMOTE dataset shape %s' % Counter(y_res))

    term = 0
    preterm = 0
    for label in np.nditer(labels):
        if label == 0:
            term +=1
        else:
            preterm += 1
    print('preterm: {}'.format(preterm))
    print('term: {}'.format(term))
    term = 0
    preterm = 0
    for label in np.nditer(y_res):
        if label == 0:
            term +=1
        else:
            preterm += 1
    print('SMOTE preterm: {}'.format(preterm))
    print('SMOTE term: {}'.format(term))


    '''SAMPLE PROGRAM'''
    # X, y = make_classification(n_samples=1000,n_features=20,n_classes=2, class_sep=2,weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_clusters_per_class=1, random_state=10)
    # # print(X[0])
    # print(type(y[0]))
    # for i in range(10):
    #     print(y[i])
    # # print('x type: {}'.format(type(X)))
    # # print('Original dataset shape %s' % Counter(y))
    # print('X shape: {}'.format(X.shape))
    # print('y shape: {}'.format(y.shape))

    # # Original dataset shape Counter({1: 900, 0: 100})
    # sm = SMOTE(random_state=42)
    # X_res, y_res = sm.fit_resample(X, y)
    # # print('Resampled dataset shape %s' % Counter(y_res))
    # print('X_res shape: {}'.format(X_res.shape))
    # print('y_res shape: {}'.format(y_res.shape))
    # Resampled dataset shape Counter({0: 900, 1: 900})