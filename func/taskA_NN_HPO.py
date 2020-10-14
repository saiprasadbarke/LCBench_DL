import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np

from networks.taskA_networks import standard_CNN, standard_LSTM
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker

class PyTorchWorker(Worker):
    '''My BOHB worker'''
    def __init__(self, train_loader, validation_loader, test_loader, **kwargs):
        super().__init__(**kwargs)
        '''Initialize the data sturctures'''
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader

    def compute(self, config: CS.Configuration, budget, working_directory:str, *args, **kwargs) -> dict:
        """
        Standard CNN
        The input parameter "config" (dictionary) contains the sampled configurations passed by the bohb optimizer
        """

        # model_type : 0 -> LSTM | 1-> CNN

        # if model_type:
        #     num_filters_per_layer = [config[k] for k in sorted(config.keys()) if k.startswith('num_filters')]
        #     myCNN = standard_CNN(num_filters_per_layer)
        #     model = myCNN.model
        # else:
        myLSTM = standard_LSTM(num_lstm_layers = config["num_lstm_layers"], hidden_size =config["num_hidden_neurons"])
        model = myLSTM
        
        print(model)
        criterion = nn.MSELoss()
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=config['sgd_momentum'])

        for _epoch in range(int(budget)):
            loss = 0
            model.train()
            for _idx , data in enumerate(self.train_loader):
                inputs, labels = data
                #print("Input shape: ", inputs.shape)
                #print("label shape: ", labels.shape)
                #if not model_type:
                    #print("comes here")
                    #print(inputs.shape)
                    #print(type(inputs))
                #inputs = inputs.reshape(17,1, 1)
                    #print(inputs.shape)
                    #print(type(inputs))
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
            for _idx , data in enumerate(data_loader):
                inputs, labels = data
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_losses.append(loss.item())
            mean_loss = np.mean(test_losses)        
        return(mean_loss)


    @staticmethod
    def get_configspace() -> CS.Configuration:
        """ Define a conditional hyperparameter search-space.
    
        hyperparameters:
          num_filters_1   from    4 to   32 (int)
          num_filters_2   from    4 to   32 (int)
          num_filters_3   from    4 to   32 (int)
          num_conv_layers from    1 to    3 (int)
          lr              from 1e-6 to 1e-1 (float, log)
          sgd_momentum    from 0.00 to 0.99 (float)
          optimizer            Adam or  SGD (categoric)
          
        conditions: 
          include num_filters_2 only if num_conv_layers > 1
          include num_filters_3 only if num_conv_layers > 2
          include sgd_momentum  only if       optimizer = SGD
        """
        cs = CS.ConfigurationSpace()

        lr = CSH.UniformFloatHyperparameter(
            'lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        sgd_momentum = CSH.UniformFloatHyperparameter(
            'sgd_momentum', lower=0.0, upper=0.99, default_value=0.9, log=False)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD'])
        cs.add_hyperparameters([lr, optimizer, sgd_momentum])
        
        cond_opt = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond_opt)

        # if model_type:
        #     num_conv_layers = CSH.UniformIntegerHyperparameter(
        #         'num_conv_layers', lower=1, upper=3, default_value=2)
        #     num_filters_1 = CSH.UniformIntegerHyperparameter(
        #         'num_filters_1', lower=4, upper=64, default_value=16, log=True)
        #     num_filters_2 = CSH.UniformIntegerHyperparameter(
        #         'num_filters_2', lower=4, upper=64, default_value=16, log=True)
        #     num_filters_3 = CSH.UniformIntegerHyperparameter(
        #         'num_filters_3', lower=4, upper=64, default_value=16, log=True)
        #     cs.add_hyperparameters(
        #         [num_conv_layers, num_filters_1, num_filters_2, num_filters_3])
    
        #     cond_filters_2 = CS.GreaterThanCondition(num_filters_2, num_conv_layers, 1)
        #     cond_filters_3 = CS.GreaterThanCondition(num_filters_3, num_conv_layers, 2)
        #     cs.add_conditions([cond_filters_2, cond_filters_3])
        # else:
        num_lstm_layers = CSH.UniformIntegerHyperparameter(
                'num_lstm_layers', lower=1, upper=2, default_value=1)
        num_hidden_neurons = CSH.UniformIntegerHyperparameter(
                'num_hidden_neurons', lower=90, upper=110, default_value=100, log=True)
        #dropout_rate = CSH.UniformFloatHyperparameter(
         #       name='dropout_rate', lower=0.0, upper=0.9, default_value=0.5, log=False)
        cs.add_hyperparameters(
                [num_lstm_layers, num_hidden_neurons])
    
        return cs