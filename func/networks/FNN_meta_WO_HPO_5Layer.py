#Imports
import torch.nn as nn

# define the network class
class FNN_meta_WO_HPO_5Layer(nn.Module):
    '''5 layer fully connected FNN. Hidden layer size from Autopytorch tabular paper'''
    def __init__(self, input_size, output_size):
        # call constructor from superclass
        super(FNN_meta_WO_HPO_5Layer, self).__init__()
        # define network layers
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 42)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(42, 29)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(29,16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16,3)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(3,1)
        
    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        output = self.relu4(output)
        output = self.fc5(output)
        return output
    
    



