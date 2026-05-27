import torch

#basic neural network with sigmoid
class sigmoidNN(torch.nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(sigmoidNN, self).__init__()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers = self.build_network()

    def build_network(self):
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No activation on the last layer
                layers.append(torch.nn.Sigmoid())
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)