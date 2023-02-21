import torch
import torch.nn as nn


class FirstConv(nn.Module):
    """
    논문 architecture 상 stage 1에 해당하는 convolution layer
    """
    def __init__(self, in_channels, out_channels, strides, kernel_size):
        super().__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        )

    def forward(self, x):
        out = self.firstconv(x)

        return out


class NormConv(nn.Module):
    """
    논문 architecture 상 stage 2에 해당하는 convolution layer.
    MBConv와의 차이로는 별도 expand가 적용되지 않는 점과 inverted residual 내에서 1*1 convolution 연산이 생략됨.
    """
    def __init__(self, in_channels, out_channels, strides, kernel_size, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=kernel_size, stride=strides, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se_block = SEBlock(in_channels)

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
            Swish()
        )

        if (strides == 1) and (in_channels == out_channels):
            self.shortcut = True

        else:
            self.shortcut = False

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        out = self.residual(x)
        se_out = self.se_block(out)

        out = out * se_out
        out = self.pointwise(out)

        if self.shortcut:
            out = out + x

        return out


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernel_size, p=0.5):
        super().__init__()
        self.p = torch.tensor(p).float() if (in_channels == out_channels) else torch.tensor(1).float()

        self.expand = 6
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * self.expand,
                      stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels * self.expand, momentum=0.99, eps=1e-3),
            Swish(),

            nn.Conv2d(in_channels=in_channels * self.expand, out_channels=in_channels * self.expand,
                      stride=strides, kernel_size=kernel_size, padding=kernel_size // 2,
                      groups=in_channels * self.expand, bias=False),
            nn.BatchNorm2d(in_channels * self.expand, momentum=0.99, eps=1e-3),
            Swish()
        )

        self.se_block = SEBlock(in_channels=in_channels * self.expand)

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * self.expand, out_channels=out_channels,
                      stride=1, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
            Swish()
        )

        if (strides == 1) and (in_channels == out_channels):
            self.shortcut = True

        else:
            self.shortcut = False

    def forward(self, x):
        if self.training:
            if not torch.bernoulli(self.p):
                return x

        out = self.residual(x)

        se_out = self.se_block(out)
        out = out * se_out

        out = self.pointwise(out)

        if self.shortcut:
            out = out + x

        return out


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels, strides, kernel_size):
        super().__init__()

        self.lastconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
            Swish()
        )

    def forward(self, x):
        out = self.lastconv(x)

        return out


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))

        self.excitation = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=in_channels // reduction_ratio),
            Swish(),
            nn.Linear(in_features=in_channels // reduction_ratio, out_features=in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.squeeze(x)
        out = out.view(out.size(0), -1)
        out = self.excitation(out)
        out = out.view(out.size(0), out.size(1), 1, 1)

        return out


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x * self.sigmoid(x)

        return out
