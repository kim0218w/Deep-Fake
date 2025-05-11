import torch
import torch.nn as nn
import torch.nn.functional as F

class ModulatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, 
                 demodulate=True, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.demodulate = demodulate

        # 기본 weight (learnable)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, kernel_size, kernel_size)
        )

        # 스타일 벡터를 scale로 변환하는 affine layer
        self.style = nn.Linear(style_dim, in_channels)

    def forward(self, x, w):
        """
        x: (batch, in_channels, height, width)
        w: (batch, style_dim)
        """
        batch, in_c, h, w_ = x.shape

        # 1. 스타일로부터 modulation scale 생성 (batch, in_c)
        style = self.style(w).view(batch, 1, in_c, 1, 1)

        # 2. weight modulation: (batch, out_c, in_c, k, k)
        weight = self.weight * style

        # 3. weight demodulation
        if self.demodulate:
            d = torch.rsqrt((weight ** 2).sum([2, 3, 4]) + self.eps)
            weight = weight * d.view(batch, self.out_channels, 1, 1, 1)

        # 4. 그룹 convolution으로 처리
        x = x.view(1, batch * in_c, h, w_)
        weight = weight.view(batch * self.out_channels, in_c, self.kernel_size, self.kernel_size)
        out = F.conv2d(x, weight, padding=self.kernel_size // 2, groups=batch)
        out = out.view(batch, self.out_channels, h, w_)

        return out
