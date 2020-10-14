from collections import OrderedDict
import torch.nn as nn
import torch

class standard_CNN:
    '''Custom CNN class for BOHB'''
    def __init__(self, num_filters_per_layer):
        super().__init__()
        self.model = self._build_net(num_filters_per_layer)

    def _build_net(self, num_filters_per_layer) -> nn.Module:
        """Builds a deep convolutional model with various number of convolution
        layers for MNIST input using pytorch.
        
        for each element in num_filters_per_layer:
            convolution (conv_kernel_size, num_filters, stride=1, padding=0)
            relu
            max pool    (pool_kernel_size, stride=1)
        linear
        """
        assert len(num_filters_per_layer) > 0, "len(num_filters_per_layer) should be greater than 0"
        pool_kernel_size = 2
        conv_kernel_size = 3

        num_output_filters = 1
        output_size = 17 # number of features / length of sequence
        layers = OrderedDict()

        conv_out = 17
        
        for i, num_filters in enumerate(num_filters_per_layer):
            layers['conv' + str(i+1)] = nn.Conv1d(num_output_filters, num_filters, kernel_size=conv_kernel_size, dilation = 1)
            layers['relu' + str(i+1)] = nn.ReLU()
            layers['pool' + str(i+1)] = nn.MaxPool1d(pool_kernel_size, dilation = 1,  stride=1)
            conv_out = (conv_out - conv_kernel_size) +1 
            print("conv_out", conv_out)
            pool_output_size = conv_out  - (conv_kernel_size - 1)
            print("pool_output_size", pool_output_size)
            conv_out = pool_output_size
            num_output_filters = num_filters
                
        conv_output_size = int(num_output_filters * pool_output_size)
        print("final conv output", conv_output_size)
    
        layers['flatten'] = Flatten()
        layers['linear'] = nn.Linear(conv_output_size, 1)
        return nn.Sequential(layers)


class standard_LSTM(nn.Module):
    '''Custom LSTM class for BOHB'''
    def __init__(self, num_lstm_layers, hidden_size):
        super(standard_LSTM, self).__init__()

        #self.model = self.make_net(num_lstm_layers, hidden_size, dropout_rate)
   
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_lstm_layers)
        
        self.hidden_state = torch.zeros(num_lstm_layers, 1, hidden_size)
        self.cell_state = torch.zeros(num_lstm_layers, 1, hidden_size)
        self.hidden = (self.hidden_state, self.cell_state)
        
        #self.drop = nn.Dropout(p=dropout_rate)

        #self.fc1 = nn.Linear(INPUT_SIZE*HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc1 = nn.Linear(17*hidden_size, 1)
        #self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

        #print(self.model)

    def make_net(self, num_lstm_layers, hidden_size) -> nn.Module:
        """Builds a lstm  model ...
        """
        seq_len = 1
        in_size = 17

        #layers = OrderedDict()
        #layers = list()
        #layers['lstm'] = nn.LSTM(input_size=seq_len, hidden_size=hidden_size, num_layers=num_lstm_layers)

        #layers['relu'] = nn.ReLU()
        #layers['drop'] = nn.Dropout(p=dropout_rate)
        #layers['flatten'] = Flatten()
        #layers['linear'] = nn.Linear(in_size*hidden_size, 1)
        return nn.ModuleDict([
                ['lstm', nn.LSTM(input_size=seq_len, hidden_size=hidden_size, num_layers=num_lstm_layers)],
                ['flatten', Flatten()],
                ['linear', nn.Linear(in_size*hidden_size, 1)]
        ])

    def forward(self, x):
        #print("input size", INPUT_SIZE)
        #lstm_input = x.reshape(17,1, 1)

        lstm_input = x.reshape(17,1, 1)
        out, (hn, cn) = self.lstm(lstm_input, self.hidden)

        out = out.view(1, -1)
        #print(out.shape)
        #out = self.drop(self.relu(self.fc1(out)))
        out = self.fc1(out)
        
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)
