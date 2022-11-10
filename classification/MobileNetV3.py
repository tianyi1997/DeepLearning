import torch
from torch import nn


class MobileNetV3(nn.Module):
    """
        MobileNetV3-Large
    """
    def __init__(self, n_class) -> None:
        super().__init__()
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )
        self.bottlenecks = nn.Sequential(
            # in, out, exp, kernel_size, stride, act, se
            Bottleneck(16, 16, 16, 3, 1, 'RE', False),
            Bottleneck(16, 24, 64, 3, 2, 'RE', False),
            Bottleneck(24, 24, 72, 3, 1, 'RE', False),
            Bottleneck(24, 40, 72, 5, 2, 'RE', True),
            Bottleneck(40, 40, 120, 5, 1, 'RE', True),
            Bottleneck(40, 40, 120, 5, 1, 'RE', True),
            Bottleneck(40, 80, 240, 3, 2, 'HS', False),
            Bottleneck(80, 80, 200, 3, 1, 'HS', False),
            Bottleneck(80, 80, 184, 3, 1, 'HS', False),
            Bottleneck(80, 80, 184, 3, 1, 'HS', False),
            Bottleneck(80, 112, 480, 3, 1, 'HS', True),
            Bottleneck(112, 112, 672, 3, 1, 'HS', True),
            Bottleneck(112, 160, 672, 5, 2, 'HS', True),
            Bottleneck(160, 160, 960, 5, 1, 'HS', True),
            Bottleneck(160, 160, 960, 5, 1, 'HS', True),
        )
        self.conv2d2 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1),
            nn.BatchNorm2d(960),
            nn.Hardswish()
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.conv2d3 = nn.Sequential(
            nn.Conv2d(960, 1280, kernel_size=1, stride=1),
            nn.Hardswish()
        )
        self.conv2d4 = nn.Conv2d(1280, n_class, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.bottlenecks(x)
        x = self.conv2d2(x)
        x = self.avg_pool(x)
        x = self.conv2d3(x)
        y = self.conv2d4(x)
        return y


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, nl_activation, need_se) -> None:
        super().__init__()
        if in_channels == out_channels and stride == 1:
            self.layers = ResidualBottleneck(in_channels, exp_size, kernel_size, nl_activation, need_se)
        else:
            self.layers = NonResidualBottleneck(in_channels, out_channels, exp_size, kernel_size, stride, nl_activation, need_se)

    def forward(self, x):
        y = self.layers(x)
        return y


class NonResidualBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, nl_activation, need_se, reduce=4) -> None:
        super().__init__()
        self.conv1x1_up = nn.Conv2d(in_channels, exp_size, (1, 1))
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.convDW = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, kernel_size//2, groups=exp_size),
            nn.Conv2d(exp_size, exp_size, kernel_size=1)
            )
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.conv1x1_down = nn.Conv2d(exp_size, out_channels, (1, 1))
        self.bn3 = nn.BatchNorm2d(out_channels)

        if need_se:
            self.selayer = SELayer(exp_size, reduce)
        else:
            self.selayer = nn.Identity()

        if nl_activation == 'HS':
            self.nl_activation = nn.Hardswish()
        elif nl_activation == 'RE':
            self.nl_activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation funciton.:', nl_activation)

    def forward(self, x):
        x = self.conv1x1_up(x)
        x = self.bn1(x)
        x = self.nl_activation(x)
        x = self.convDW(x)
        x = self.bn2(x)
        x = self.nl_activation(x)
        x = self.selayer(x)
        x = self.conv1x1_down(x)
        y = self.bn3(x)
        return y 



class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, exp_size, kernel_size, nl_activation, need_se, reduce=4) -> None:
        super().__init__()
        self.conv1x1_up = nn.Conv2d(in_channels, exp_size, (1, 1))
        self.bn1 = nn.BatchNorm2d(exp_size)
        self.convDW = nn.Sequential(
            nn.Conv2d(exp_size, exp_size, kernel_size, stride=1, padding=kernel_size//2, groups=exp_size),
            nn.Conv2d(exp_size, exp_size, 1)
            )
        self.bn2 = nn.BatchNorm2d(exp_size)
        self.conv1x1_down = nn.Conv2d(exp_size, in_channels, (1, 1))
        self.bn3 = nn.BatchNorm2d(in_channels)
        
        if need_se:
            self.selayer = SELayer(exp_size, reduce)
        else:
            self.selayer = nn.Identity()

        if nl_activation == 'HS':
            self.nl_activation = nn.Hardswish()
        elif nl_activation == 'RE':
            self.nl_activation = nn.ReLU()
        else:
            raise ValueError('Unknown activation funciton.:', nl_activation)

    def forward(self, x):
        res = self.conv1x1_up(x)
        res = self.bn1(res)
        res = self.nl_activation(res)
        res = self.convDW(res)
        res = self.bn2(res)
        res = self.nl_activation(res)
        res = self.selayer(res)
        res = self.conv1x1_down(res)
        res = self.bn3(res)
        print(x.shape, res.shape)

        y = torch.add(x , res)
        return y 

class SELayer(nn.Module):
    def __init__(self, in_channels, reduce) -> None:
        super().__init__()
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.squeeze_layer = nn.Sequential(
            nn.Linear(in_channels, in_channels//reduce),
            nn.ReLU(),
            nn.Linear(in_channels//reduce, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.global_avg_pooling(x).view((B, C))
        a = self.squeeze_layer(a).view((B, C, 1, 1))
        y = torch.mul(x, a)
        return y

if __name__ == '__main__':
    mobilenetv3 = MobileNetV3(1000)
    inputs = torch.rand((5, 3, 224, 224))
    mobilenetv3.eval()
    print(mobilenetv3(inputs).shape)