import torch.nn as nn


class Probe(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=64):
        super(Probe, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        hidden = self.linear1(x).clamp(min=0)
        logits = self.linear2(hidden)
        return logits


