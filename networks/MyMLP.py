from collections import OrderedDict
import torch.nn as nn

class MyMLP:
    '''Custom MLP class for BOHB'''
    def __init__(self, n_layers, dropout_rate, n_inputs, n_outputs):
        super().__init__()
        self.model = self._build_net(n_layers, dropout_rate, n_inputs, n_outputs)

    def _build_net(self, n_layers, dropout_rate, n_inputs, n_outputs) -> nn.Module:
        layers = OrderedDict()
        neuron_list = number_neurons(n_layers)
        for i in range(n_layers):
            if i<n_layers-1:
                layers["fc" + str(i+1)] = nn.Linear(int(neuron_list[i]), int(neuron_list[i+1]))
                layers["dropout" + str(i+1)] = nn.Dropout(dropout_rate)
                layers["relu" + str(i+1)] = nn.ReLU()

        layers["fc" + str(n_layers)] = nn.Linear(int(neuron_list[-1]), 1)
        return nn.Sequential(layers)

def number_neurons(n_layers):
    '''Number of neurons in each layer'''
    if n_layers == 3:
        neuron_list = [55,37,19]
    elif n_layers == 4:
        neuron_list = [55,41,27,13]
    elif n_layers == 5:
        neuron_list = [55,44,33,22,11]
    elif n_layers == 6:
        neuron_list = [55,46,37,28,19,10]
    elif n_layers == 7:
        neuron_list = [55,47,39,31,23,15,7]
    elif n_layers == 8:
        neuron_list = [55,48,41,34,27,20,13,6]

    return neuron_list

if __name__ == "__main__":
    MyMLP = MyMLP(4, 0.5, 55, 1)
    model = MyMLP.model
    print(model)