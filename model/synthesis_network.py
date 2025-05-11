# models/synthesis_network.py
import torch
import torch.nn as nn
from modules.style_modulation import StyleBlock

class SynthesisNetwork(nn.Module):
    def __init__(self, resolution=256, style_dim=512, channels={4: 512, 8: 512, 16: 512, 32: 256, 64: 128}):
        super(SynthesisNetwork, self).__init__()
        self.channels = channels
        self.style_dim = style_dim
        self.resolution = resolution

        self.constant_input = nn.Parameter(torch.randn(1, channels[4], 4, 4))
        self.block4 = StyleBlock(channels[4], channels[4], style_dim)

        self.blocks = nn.ModuleDict()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        resolutions = sorted([r for r in channels.keys() if r > 4])
        for res in resolutions:
            in_c = channels[res // 2]
            out_c = channels[res]
            self.blocks[str(res)] = StyleBlock(in_c, out_c, style_dim)

        self.to_rgb = nn.Conv2d(channels[max(channels)], 3, kernel_size=1)

    def forward(self, w):
        batch_size = w.shape[0]
        x = self.constant_input.expand(batch_size, -1, -1, -1)
        x = self.block4(x, w)
        for res in sorted(self.blocks.keys(), key=lambda r: int(r)):
            x = self.upsample(x)
            x = self.blocks[res](x, w)
        return self.to_rgb(x)
