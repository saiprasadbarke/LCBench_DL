'''This file contains all the functions for preprocessing the data'''

#Organise imports
import numpy as np
from sklearn import  preprocessing
from math import isnan

def delete_constant_features_X(X):
    '''This function removes the constant features in the input''' 
    transformedX = []
    for config in X:
        
        modifiedConfig = [config["num_layers"],config["learning_rate"],config["weight_decay"],config["batch_size"],config["max_dropout"],config["max_units"],config["momentum"]]
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
    
def parse_metafeatures_dict(metafeatures_dict, dataset_names):
    '''Parse metafeature json and create a dictionary'''
    metafeatures6datasets = dict()
    for dataset_name in dataset_names:
        dataset_dict = dict()
        iterator = iter(metafeatures_dict[dataset_name].items())
        for _ind in range(len(metafeatures_dict[dataset_name])):
            feature, value = iterator.__next__()
            dataset_dict[feature] = value
        metafeatures6datasets[dataset_name] = dataset_dict
    return metafeatures6datasets

def remove_nan_metafeatures(metafeatures_dict):
    '''Remove nan values from metafeature dicts'''
    nan_metafeatures = []
    new_dict = dict()
    dict_items = dict(metafeatures_dict["vehicle"]).items()

    for metafeature, value in dict_items:
        if isnan(value):
            nan_metafeatures.append(metafeature)

    for dataset_name, metafeature_dict in metafeatures_dict.items():
        for metafeature in nan_metafeatures:
            metafeature_dict.pop(metafeature)
        new_dict[dataset_name] = metafeature_dict
    return new_dict    

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