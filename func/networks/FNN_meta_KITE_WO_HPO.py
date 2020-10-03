#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# define the network class
class FNN_meta_KITE_WO_HPO(nn.Module):
    '''Kite shaped fully connected FNN.'''
    def __init__(self, input_size, output_size):
        # call constructor from superclass
        super(FNN_meta_KITE_WO_HPO, self).__init__()
        # define network layers
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, 68)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(68, 81)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(81,94)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(94,107)
        self.dropout1 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(107,94)
        self.dropout2 = nn.Dropout(0.5)
        self.relu5 = nn.ReLU()
        self.fc6 = nn.Linear(94, 81)
        self.dropout3 = nn.Dropout(0.5)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Linear(81, 68)
        self.dropout4 = nn.Dropout(0.5)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Linear(68, 55)
        self.dropout5 = nn.Dropout(0.5)
        self.relu8 = nn.ReLU()
        self.fc9 = nn.Linear(55, 42)
        self.dropout6 = nn.Dropout(0.5)
        self.relu9 = nn.ReLU()
        self.fc10 = nn.Linear(42, 29)
        self.relu10 = nn.ReLU()
        self.fc11 = nn.Linear(29, 16)
        self.relu11 = nn.ReLU()
        self.fc12 = nn.Linear(16,3)
        self.relu12 = nn.ReLU()
        self.fc13 = nn.Linear(3,output_size)
        
    def forward(self, x):
        # define forward pass
        output = self.fc1(x)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        output = self.dropout1(output)
        output = self.relu4(output)
        output = self.fc5(output)
        output = self.dropout2(output)
        output = self.relu5(output)
        output = self.fc6(output)
        output = self.dropout3(output)
        output = self.relu6(output)
        output = self.fc7(output)
        output = self.dropout4(output)
        output = self.relu7(output)
        output = self.fc8(output)
        output = self.dropout5(output)
        output = self.relu8(output)
        output = self.fc9(output)
        output = self.dropout6(output)
        output = self.relu9(output)
        output = self.fc10(output)
        output = self.relu10(output)
        output = self.fc11(output)
        output = self.relu11(output)
        output = self.fc12(output)
        output = self.relu12(output)
        output = self.fc13(output)
        return output
    
    



