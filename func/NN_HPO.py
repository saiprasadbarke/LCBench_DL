import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np

from networks.MyMLP import MyMLP
from func.load_data import prepare_dataloaders
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

class PyTorchWorker(Worker):
    '''My BOHB worker'''
    def __init__(self, input_size, output_size, train_tuple, validation_tuple, test_tuple, **kwargs):
        super().__init__(**kwargs)
        '''Initialize the data sturctures , input and output sizes'''
        self.train_tuple = train_tuple
        self.validation_tuple = validation_tuple
        self.test_tuple = test_tuple
        self.input_size = input_size
        self.output_size = output_size

    def compute(self, config: CS.Configuration, budget, working_directory:str, *args, **kwargs) -> dict:
        """
        Simple MLP
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """
        myMLP = MyMLP(n_layers = config["num_layers"], dropout_rate =config["dropout_rate"] , n_inputs = self.input_size, n_outputs = self.output_size)
        model = myMLP.model
        print(model)
        criterion = nn.MSELoss()
        train_loader = prepare_dataloaders(X_hp=self.train_tuple[0], X_mf=self.train_tuple[1], y= self.train_tuple[2], X_scaling="minmax", y_scaling="minmax", batch_size=config["batch_size"], typeD = "tensor")
        validation_loader = prepare_dataloaders(X_hp=self.validation_tuple[0], X_mf=self.validation_tuple[1], y= self.validation_tuple[2], X_scaling="minmax",y_scaling="minmax",batch_size=config["batch_size"], typeD = "tensor")
        test_loader = prepare_dataloaders(X_hp=self.test_tuple[0], X_mf=self.test_tuple[1], y= self.test_tuple[2], X_scaling="minmax",y_scaling="minmax" ,batch_size=config["batch_size"], typeD = "tensor")
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        for _epoch in range(int(budget)):
            loss = 0
            model.train()
            for _idx , data in enumerate(train_loader):
                inputs, labels = data
                #print("Input shape: ", inputs.shape)
                #print("label shape: ", labels.shape)
                optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            train_loss = self.evaluate_loss(model, train_loader, criterion)
            validation_loss = self.evaluate_loss(model, validation_loader, criterion)
            test_loss = self.evaluate_loss(model, test_loader, criterion)

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
            for _idx , data in enumerate(data_loader):
                inputs, labels = data
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
            lr = CSH.UniformFloatHyperparameter(name='lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
            # Add different optimizers as categorical hyperparameters.
            # SGD has a conditional parameter 'momentum'.
            optimizer = CSH.CategoricalHyperparameter(name='optimizer', choices=['Adam', 'SGD'])
            sgd_momentum = CSH.UniformFloatHyperparameter(name='sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
            cs.add_hyperparameters([lr, optimizer, sgd_momentum])
            # The hyperparameter sgd_momentum will be used,if the configuration
            # contains 'SGD' as optimizer.
            cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
            cs.add_condition(cond)
            #Number of layers in the MLP
            num_layers =  CSH.UniformIntegerHyperparameter(name='num_layers', lower=3, upper=8)
            cs.add_hyperparameters([num_layers])
            batch_size = CSH.UniformIntegerHyperparameter(name='batch_size', lower=1, upper=10)
            cs.add_hyperparameters([batch_size])
            dropout_rate = CSH.UniformFloatHyperparameter(name='dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
            cs.add_hyperparameters([dropout_rate])
            return cs