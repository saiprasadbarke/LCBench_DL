from math import isnan
import numpy as np

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

def create_metafeatures_array(metafeatures_dict, dataset_names):
    '''create metafeature array from metafeature dictonary'''
    metafeatures_list = []
    for dataset in dataset_names:
        for _ in range(2000):
            metafeatures_list.append(metafeatures_dict[dataset])
    return np.array(metafeatures_list)

