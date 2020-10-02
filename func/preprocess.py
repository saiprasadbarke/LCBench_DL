'''This file contains all the functions for preprocessing the data'''

#Organise imports
import numpy as np
from sklearn import  preprocessing


def delete_constant_features_X(X):
    '''This function removes the constant features in the input''' 
    transformedX = []
    for config in X:
        modifiedConfig = [config["batch_size"],config["max_dropout"],config["max_units"], config["learning_rate"],config["momentum"],config["weight_decay"]]
        transformedX.append(modifiedConfig)
    return np.asarray(transformedX)


def reshape_op(y):
    '''This function is used to reshape the output'''
    return np.reshape(y, (-1,1))


def scale_features(X, method=None):
    '''This function is used perform feature scaling. Have implemented 2 methods: min-max normalization, Z-normalization'''
    if method == "znorm":
        scaler = preprocessing.StandardScaler()
        x_std = scaler.fit_transform(X)
        return x_std
    elif method == "minmax":
        min_max_scaler = preprocessing.MinMaxScaler()
        x_minmax = min_max_scaler.fit_transform(X)
        return x_minmax
    else:
        return X

def create_metafeatures_2d(metafeatures_list):
    '''Create 2d numpy array of metafeatures'''
    metafeatures_2d = []
    for metafeatures in metafeatures_list:
        metafeatures_1d = list(metafeatures.values())
        metafeatures_2d.append(metafeatures_1d)
    return np.asarray(metafeatures_2d)

def concatenate_metafeatures_features(features, metafeatures):
    '''concatenate features and metafeatures'''
    return np.append(features, metafeatures,axis=1)