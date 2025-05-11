# models/mapping_network.py
import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()
        layers = [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2)]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))
        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
