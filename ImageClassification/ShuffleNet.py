import torch
from torch import nn

class ShuffleNet(nn.Module):
    def __init__(self, num_classes, num_groups, scale_ratio=1) -> None:
        super().__init__()
        group2channels = {
            1: int(144*scale_ratio),
            2: int(200*scale_ratio),
            3: int(240*scale_ratio),
            4: int(272*scale_ratio),
            8: int(384*scale_ratio)
        }
        assert num_groups in group2channels, 'Unknown num_group'
        out_channels = int(group2channels[num_groups]*scale_ratio)
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=int(24*scale_ratio), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(int(24*scale_ratio)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ShuffleNetBlock(int(24*scale_ratio), out_channels, 2, num_groups, grouped_conv1=False),
            *[ShuffleNetBlock(out_channels, out_channels, 1, num_groups) for _ in range(3)]
        )
        self.stage3 = nn.Sequential(
            ShuffleNetBlock(out_channels, 2*out_channels, 2, num_groups),
            *[ShuffleNetBlock(2*out_channels, 2*out_channels, 1, num_groups) for _ in range(7)]
        )
        self.stage4 = nn.Sequential(
            ShuffleNetBlock(2*out_channels, 4*out_channels, 2, num_groups),
            *[ShuffleNetBlock(4*out_channels, 4*out_channels, 1, num_groups) for _ in range(3)]
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


class ShuffleNetBlock(nn.Module):
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

        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=stride, padding=1, groups=hidden_channels),
            nn.BatchNorm2d(hidden_channels)
        )
        if self.downsample:
            # TODO out_channels-in_channels must be divisible by groups (e.g. when g=8 s=0.5)
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

    def forward(self, x):
        B, _, H, W = x.shape
        res = self.gconv1x1_down(x)
        res = res.view(B, self.num_groups, -1, H, W)
        res = res.permute(0, 2, 1, 3, 4).flatten(1, 2)
        res = self.dwconv3x3(res)
        res = self.gconv1x1_up(res)
        if self.downsample:
            y = torch.concat([res, self.shortcut(x)], dim=1)
        else:
            y = res + self.shortcut(x)
        return y

if __name__ == '__main__':
    model = ShuffleNet(num_classes=1000, num_groups=3, scale_ratio=1)
    model.eval()
    inputs = torch.rand((5, 3, 224, 224))
    print(model(inputs).shape)    