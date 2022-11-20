import torch
from torch import nn


def DepthwiseConv2d(in_channels, kernel_size, **kwarg):
    return nn.Conv2d(in_channels, in_channels, kernel_size, **kwarg, groups=in_channels)


class ShuffleNetV1(nn.Module):
    def __init__(self, num_classes, num_groups, scale_factor=1) -> None:
        super().__init__()
        group2channels = {
            1: 144,
            2: 200,
            3: 240,
            4: 272,
            8: 384
        }
        assert num_groups in group2channels, 'Unknown num_groups'
        self.num_classes = num_classes
        out_channels = int(group2channels[num_groups]*scale_factor)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ShuffleNetV1Unit(24, out_channels, 2, num_groups, grouped_conv1=False),
            *[ShuffleNetV1Unit(out_channels, out_channels, 1, num_groups) for _ in range(3)]
        )
        self.stage3 = nn.Sequential(
            ShuffleNetV1Unit(out_channels, 2*out_channels, 2, num_groups),
            *[ShuffleNetV1Unit(2*out_channels, 2*out_channels, 1, num_groups) for _ in range(7)]
        )
        self.stage4 = nn.Sequential(
            ShuffleNetV1Unit(2*out_channels, 4*out_channels, 2, num_groups),
            *[ShuffleNetV1Unit(4*out_channels, 4*out_channels, 1, num_groups) for _ in range(3)]
        )
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4*out_channels, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.globalpool(x)
        x = x.flatten(1)
        y = self.fc(x)
        return y


class ShuffleNetV1Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, num_groups, bottleneck_scale=0.25, grouped_conv1=True) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.downsample = stride != 1
        hidden_channels = int(bottleneck_scale*out_channels)
        self.gconv1x1_down = nn.Sequential( 
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, groups=(num_groups if grouped_conv1 else 1)),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.channelshuffle = ChannelShuffle(num_groups)
        self.dwconv3x3 = nn.Sequential(
            DepthwiseConv2d(hidden_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(hidden_channels)
        )
        if self.downsample:
            self.gconv1x1_up = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels-in_channels, kernel_size=1, groups=num_groups),
                nn.BatchNorm2d(out_channels-in_channels)
            )
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        else:
            self.gconv1x1_up = nn.Sequential(
                nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1, groups=num_groups),
                nn.BatchNorm2d(out_channels)
            )
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.gconv1x1_down(x)
        res = self.channelshuffle(res)
        res = self.dwconv3x3(res)
        res = self.gconv1x1_up(res)
        if self.downsample:
            y = self.relu(torch.concat([res, self.shortcut(x)], dim=1))
        else:
            y = self.relu(res + self.shortcut(x))
        return y


class ChannelShuffle(nn.Module):
    def __init__(self, num_groups) -> None:
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x):
        B, _, H, W = x.shape
        x = x.view(B, self.num_groups, -1, H, W)    # reshape (B, NG, H, W) --> (B, N, G, H, W)
        x = x.permute(0, 2, 1, 3, 4)    # transpose (B, N, G, H, W) --> (B, G, N, H, W)
        x = x.flatten(1, 2)     # flatten (B, G, N, H, W) --> (B, GN, H, W  )
        return x


if __name__ == '__main__':
    model = ShuffleNetV1(num_classes=1000, num_groups=8, scale_factor=0.25)
    model.eval()
    inputs = torch.rand((5, 3, 224, 224))
    print(model(inputs).shape)    