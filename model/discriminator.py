# models/discriminator.py
import torch
import torch.nn as nn

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        residual = self.skip(self.downsample(x))
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.downsample(x)
        return (x + residual) / torch.sqrt(torch.tensor(2.0, device=x.device))

class Discriminator(nn.Module):
    def __init__(self, resolution=256, channels={4: 512, 8: 512, 16: 512, 32: 256, 64: 128}):
        super(Discriminator, self).__init__()
        resolutions = sorted(channels.keys(), reverse=True)
        blocks = []
        for i in range(len(resolutions) - 1):
            blocks.append(DiscriminatorBlock(channels[resolutions[i]], channels[resolutions[i + 1]]))
        self.blocks = nn.Sequential(*blocks)

        self.final_conv = nn.Conv2d(channels[4], channels[4], 3, padding=1)
        self.final_act = nn.LeakyReLU(0.2)
        self.fc = nn.Linear(channels[4] * 4 * 4, 1)

    def forward(self, x):
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.final_act(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
