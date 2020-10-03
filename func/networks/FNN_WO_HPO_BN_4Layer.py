#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# define the network class
class FNN_WO_HPO_BN_4Layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # call constructor from superclass
        super(FNN_WO_HPO_BN_4Layer, self).__init__()
        # define network layers
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, 4)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(4, 3)
        self.relu1 = nn.ReLU()
        self.fc4 = nn.Linear(3,output_size)
        
    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.bn1(output)
        output = self.sigmoid1(output)
        output = self.fc2(output)
        output = self.sigmoid2(output)
        output = self.fc3(output)
        output = self.relu1(output)
        output = self.fc4(output)
        return output



