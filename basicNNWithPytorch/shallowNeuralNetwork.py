from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.nntype = 'shallow'
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        x = self.flatten(x)
        probs = self.linear_stack(x)
        return probs