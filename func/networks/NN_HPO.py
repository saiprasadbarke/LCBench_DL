import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from networks.MyMLP import MyMLP

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB

import logging
import pickle
logging.getLogger('hpbandster').setLevel(logging.DEBUG)

class PyTorchWorker(Worker):
    def __init__(self, input_size, output_size, train_loader, validation_loader, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self.input_size = input_size
        self.output_size = output_size

    def compute(self, config: CS.Configuration, budget, working_directory:str, *args, **kwargs) -> dict:
        """
        Simple MLP
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        myMLP = MyMLP(n_layers = config["num_layers"], dropout_rate =config["dropout_rate"] , n_inputs = self.input_size, n_outputs = self.output_size)
        model = myMLP.model
        criterion = nn.MSELoss()
        if config['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        for _epoch in range(int(budget)):
            loss = 0
            model.train()
            for inputs, labels in enumerate(self.train_loader):
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            train_loss = self.evaluate_loss(model, self.train_loader, criterion)
            validation_loss = self.evaluate_loss(model, self.validation_loader, criterion)
            test_loss = self.evaluate_loss(model, self.test_loader, criterion)

            return ({
                    'loss': validation_loss,
                    'info': {
                                'test accuracy': test_loss,
                                'train accuracy': train_loss,
                                'validation accuracy': validation_loss,
                                'model': str(model)
                            }
                    })

    def evaluate_loss(self, model: nn.Module, data_loader: DataLoader, criterion: nn.MSELoss) -> float:
        test_losses = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
            mean_loss = np.mean(test_losses)        
        return(mean_loss)


    @staticmethod
    def get_configspace() -> CS.ConfigurationSpace:
            """
            It builds the configuration space with the needed hyperparameters.
            It is easily possible to implement different types of hyperparameters.
            Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
            :return: ConfigurationsSpace-Object
            """
            cs = CS.ConfigurationSpace()
            lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
            # Add different optimizers as categorical hyperparameters.
            # SGD has a conditional parameter 'momentum'.
            optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
            sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
            cs.add_hyperparameters([lr, optimizer, sgd_momentum])
            # The hyperparameter sgd_momentum will be used,if the configuration
            # contains 'SGD' as optimizer.
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)
            #Number of layers in the MLP
            num_layers =  CSH.UniformIntegerHyperparameter('num_layers', lower=3, upper=7, default_value=2)
            cs.add_hyperparameters([num_layers])
            dropout_rate = CSH.UniformFloatHyperparameter('dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
            cs.add_hyperparameters([dropout_rate])
            return cs





