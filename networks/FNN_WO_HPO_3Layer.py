#Imports

import torch.nn as nn

# define the network class
class FNN_WO_HPO_3Layer(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        # call constructor from superclass
        super(FNN_WO_HPO_3Layer, self).__init__()
        # define network layers
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.sigmoid1(output)
        output = self.fc2(output)
        output = self.sigmoid2(output)
        output = self.fc3(output)
        return output



