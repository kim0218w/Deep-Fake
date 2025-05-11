# modules/style_modulation.py
import torch
import torch.nn as nn

class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim):
        super(StyleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.noise_weight = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.style_scale = nn.Linear(style_dim, out_channels)
        self.style_bias = nn.Linear(style_dim, out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, w):
        batch_size, _, height, width = x.shape

        # Style modulation
        scale = self.style_scale(w).view(batch_size, -1, 1, 1)
        bias = self.style_bias(w).view(batch_size, -1, 1, 1)

        # Apply style to conv weights (adaptive instance norm)
        x = self.conv(x)
        x = x * (scale + 1) + bias

        # Add noise
        noise = torch.randn(batch_size, 1, height, width, device=x.device)
        x = x + self.noise_weight * noise

        return self.activation(x)
