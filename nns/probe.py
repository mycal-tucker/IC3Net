import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64, num_layers=1):
        super(Probe, self).__init__()
        self.layers = nn.ModuleList()
        self.out_dim = output_dim
        prev_size = input_dim
        for layer_id in range(num_layers):
            next_dim = hidden_size if layer_id < num_layers - 1 else output_dim
            self.layers.append(nn.Linear(prev_size, next_dim))
            prev_size = hidden_size

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id < len(self.layers) - 1:
                x = x.clamp(min=0)
        return x


