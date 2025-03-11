from torch import nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.nntype = 'deep'
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            prev_size = size
        self.output_layer = nn.Linear(prev_size, output_size)
        self.activation = nn.PReLU()

    def forward(self, x):
        x: torch.Tensor
        x = self.flatten(x)
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        probs = nn.Softmax(dim=1)(x)
        return probs