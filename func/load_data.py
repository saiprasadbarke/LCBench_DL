'''This file contains functions for loading the data and creating dataloaders'''
import os
import json

import numpy as np
from sklearn.model_selection import train_test_split

from preprocess import delete_constant_features_X, scale_features, reshape_op, remove_nan_metafeatures, create_metafeatures_2d, concatenate_metafeatures_features
from api import Benchmark

from torch import from_numpy
from torch.utils.data import DataLoader, TensorDataset

class TrainValSplitter():
    """Splits 25 % data as a validation split."""
    
    def __init__(self, dataset_names):
        self.ind_train, self.ind_val = train_test_split(np.arange(8000), test_size=0.25, stratify=dataset_names)
        
    def split(self, a):
        return a[self.ind_train], a[self.ind_val]

def read_data(bench, datasets):
    '''Reads configs and last value of the validation accuracy curve.'''
    #Since there are 2000 configs in each dataset we take the length of the first dataset only
    n_configs = bench.get_number_of_configs(datasets[0]) 
    data = [bench.query(dataset_name=d, tag="Train/val_accuracy", config_id=ind) for d in datasets for ind in range(n_configs)]
    configs = [bench.query(dataset_name=d, tag="config", config_id=ind) for d in datasets for ind in range(n_configs)]
    dataset_names = [d for d in datasets for ind in range(n_configs)]
    #We take the Y values as the last value of the Train/val_accuracy curves (Y has a single column)
    y = np.array([curve[-1] for curve in data])
    #the 1st return is a 2D array of configs, 2nd return is a 1D column vector of last values of Train/val_accuracy curves
    return np.array(configs), y, np.array(dataset_names)


def create_metafeatures_array(metafeatures_dict, dataset_names):
    '''Create metafeature array from metafeature dictonary'''
    metafeatures_list = []
    for dataset in dataset_names:
        for _ in range(2000):
            metafeatures_list.append(metafeatures_dict[dataset])
    return np.array(metafeatures_list)

def prepare_dataloaders(X_hp, X_mf, y, scaling = None, batch_size = None):
    '''This function is used to delete constant features, reshape data, scale data, concatenate hyperparameters and dataset metafeatures , convert data to tensors and prepare dataloaders'''
    X_hp_transformed = delete_constant_features_X(X_hp)
    X_mf_transformed = create_metafeatures_2d(X_mf)
    y_transformed = reshape_op(y)
    #print("X_hp_transformed:", X_hp_transformed.shape)
    #print("X_mf_transformed:", X_mf_transformed.shape)
    #print("y_transformed:", y_transformed.shape)
    #print() 
    X_hp_scaled = scale_features(X_hp_transformed, method=scaling)
    X_mf_scaled = scale_features(X_mf_transformed, method=scaling)
    y_scaled = scale_features(y_transformed, method=scaling)
    #print("X_hp_scaled:", X_hp_scaled.shape)
    #print("X_mf_scaled:", X_mf_scaled.shape)
    #print("y_scaled:", y_scaled.shape)
    #print() 
    X_features = concatenate_metafeatures_features(X_hp_scaled, X_mf_scaled)
    #print("X_features:", X_features.shape)
    #print()
    X_features_tensor = from_numpy(X_features.astype(np.float32))
    y_labels_tensor = from_numpy(y_scaled.astype(np.float32))
    #print("X_features_tensor:", X_features_tensor.shape)
    #print("y_labels_tensor:", y_labels_tensor.shape)
    #print() 
    dataset = TensorDataset(X_features_tensor, y_labels_tensor)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle= True, num_workers=2)

    return dataloader

def load_data_from_file(path_hyperparams, path_metafeatures):
    '''This function navigates to the locations pointed in the arguments and loads the data from the json files and returns the data as numpy arrays'''
    path = os.getcwd()
    path_hp = path + "/" + path_hyperparams
    path_mf = path + "/" + path_metafeatures
    print("Path to hyperparameters data: ", path_hp)
    print("Path to metafeatures data: ", path_mf)

    #create benchmark class object to access hyperparameters data
    bench = Benchmark(path_hp, cache=False)
    dataset_names_all = bench.get_dataset_names()
    print("Available datasets: ", dataset_names_all)
    train_datasets = ['adult', 'higgs', 'vehicle', 'volkert']
    test_datasets = ['Fashion-MNIST', 'jasmine']
    #load metafeatures and remove nan values
    with open(path_mf, "r") as f:
        metafeatures = json.load(f)
    metafeatures_without_nan = remove_nan_metafeatures(metafeatures)
    #split data into train and test set
    X_tv, y_tv, dataset_names_TV = read_data(bench, train_datasets)
    X_test, y_test, dataset_names_test = read_data(bench, test_datasets)
    X_metafeatures_TV = create_metafeatures_array(metafeatures_without_nan, train_datasets)
    X_metafeatures_test = create_metafeatures_array(metafeatures_without_nan, test_datasets)

    #Splitting data into training and validation.
    tv_splitter = TrainValSplitter(dataset_names=dataset_names_TV)
    X_train, X_val = tv_splitter.split(X_tv)
    y_train, y_val = tv_splitter.split(y_tv)
    dataset_names_train, dataset_names_val = tv_splitter.split(dataset_names_TV)
    X_metafeatures_train, X_metafeatures_val = tv_splitter.split(X_metafeatures_TV)

    print(f"Train-Validation-Test split: {len(dataset_names_train)}-{len(dataset_names_val)}-{len(dataset_names_test)}")
    return X_train, X_metafeatures_train, y_train, X_val, X_metafeatures_val, y_val, X_test, X_metafeatures_test, y_test
    