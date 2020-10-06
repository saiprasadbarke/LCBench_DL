from collections import OrderedDict
import torch.nn as nn
from number_neurons import number_neurons, Capturing

class MyMLP:
    def __init__(self, n_layers, dropout_rate, n_inputs, n_outputs):
        super().__init__()
        self.model = self._build_net(n_layers, dropout_rate, n_inputs, n_outputs)

    def _build_net(self, n_layers, dropout_rate, n_inputs, n_outputs) -> nn.Module:
        layers = OrderedDict()
        with Capturing() as neuron_list:
            number_neurons(0, n_inputs, n_outputs, n_layers)

        for i in range(n_layers-2):
            layers["fc" + str(i+1)] = nn.Linear(int(neuron_list[i]), int(neuron_list[i+1]))
            layers["dropout" + str(i+1)] = nn.Dropout(dropout_rate)
            layers["relu" + str(i+1)] = nn.ReLU()

        layers["fc" + str(n_layers-1)] = nn.Linear(int(neuron_list[-2]), int(neuron_list[-1]))
        return nn.Sequential(layers)


if __name__ == "__main__":
    MyMLP = MyMLP(5, 0.5, 100, 5)
    model = MyMLP.model
    print(model)