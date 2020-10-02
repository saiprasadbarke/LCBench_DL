import numpy as np
from sklearn.model_selection import train_test_split

def read_data(bench, datasets):
    #Since there are 2000 configs in each dataset we take the length of the first dataset only
    n_configs = bench.get_number_of_configs(datasets[0]) 
    data = [bench.query(dataset_name=d, tag="Train/val_accuracy", config_id=ind) for d in datasets for ind in range(n_configs)]
    configs = [bench.query(dataset_name=d, tag="config", config_id=ind) for d in datasets for ind in range(n_configs)]
    dataset_names = [d for d in datasets for ind in range(n_configs)]
    #We take the Y values as the last value of the Train/val_accuracy curves (Y has a single column)
    y = np.array([curve[-1] for curve in data])
    #the 1st return is a 2D array of configs, 2nd return is a 1D column vector of last values of Train/val_accuracy curves
    return np.array(configs), y, np.array(dataset_names)

class TrainValSplitter():
    """Splits 25 % data as a validation split."""
    
    def __init__(self, dataset_names):
        self.ind_train, self.ind_val = train_test_split(np.arange(8000), test_size=0.25, stratify=dataset_names)
        
    def split(self, a):
        return a[self.ind_train], a[self.ind_val]