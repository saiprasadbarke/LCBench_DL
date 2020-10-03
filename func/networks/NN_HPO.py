import os

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

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

    def compute(self, config, budget, working_directory, *args, **kwargs):
            """
            Simple example for a compute function using a feed forward network.
            It is trained on the MNIST dataset.
            The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
            """

            # device = torch.device('cpu')
            model = MNISTConvNet(num_conv_layers=config['num_conv_layers'],
                                                    num_filters_1=config['num_filters_1'],
                                                    num_filters_2=config['num_filters_2'] if 'num_filters_2' in config else None,
                                                    num_filters_3=config['num_filters_3'] if 'num_filters_3' in config else None,
                                                    dropout_rate=config['dropout_rate'],
                                                    num_fc_units=config['num_fc_units'],
                                                    kernel_size=3
            )

            criterion = torch.nn.CrossEntropyLoss()
            if config['optimizer'] == 'Adam':
                    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
            else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

            for epoch in range(int(budget)):
                    loss = 0
                    model.train()
                    for i, (x, y) in enumerate(self.train_loader):
                            optimizer.zero_grad()
                            output = model(x)
                            loss = F.nll_loss(output, y)
                            loss.backward()
                            optimizer.step()

            train_accuracy = self.evaluate_accuracy(model, self.train_loader)
            validation_accuracy = self.evaluate_accuracy(model, self.validation_loader)
            test_accuracy = self.evaluate_accuracy(model, self.test_loader)

            return ({
                    'loss': 1-validation_accuracy, # remember: HpBandSter always minimizes!
                    'info': {       'test accuracy': test_accuracy,
                                            'train accuracy': train_accuracy,
                                            'validation accuracy': validation_accuracy,
                                            'number of parameters': model.number_of_parameters(),
                                    }

            })

    def evaluate_accuracy(self, model, data_loader):
            model.eval()
            correct=0
            with torch.no_grad():
                    for x, y in data_loader:
                            output = model(x)
                            #test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
                            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                            correct += pred.eq(y.view_as(pred)).sum().item()
            #import pdb; pdb.set_trace()
            accuracy = correct/len(data_loader.sampler)
            return(accuracy)


    @staticmethod
    def get_configspace():
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
