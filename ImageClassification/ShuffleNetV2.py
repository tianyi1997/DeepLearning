import torch
from torch import nn


def DepthwiseConv2d(in_channels, kernel_size, **kwarg):
    return nn.Conv2d(in_channels, in_channels, kernel_size, **kwarg, groups=in_channels)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes, scale_factor=1) -> None:
        super().__init__()
        scale2channels = {
                        # FLOPs     Params
            0.5: 48,    # 41M       1.4M
            1:  116,    # 146M      2.3M
            1.5: 176,   # 299M      3.5M
            2:  244     # 244M      7.4M
        }
        self.num_classes = num_classes
        assert scale_factor in scale2channels, 'Unknown scale factor.'
        self.scale_factor = scale_factor
        conv5_out = (1024 if scale_factor < 2 else 2048)
        num_channels = scale2channels[scale_factor]
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage2 = nn.Sequential(
            ShuffleNetV2DownsampleUnit(24, num_channels),
            *[ShuffleNetV2BasicUnit(num_channels) for _ in range(3)]
        )
        self.stage3 = nn.Sequential(
            ShuffleNetV2DownsampleUnit(num_channels, 2*num_channels),
            *[ShuffleNetV2BasicUnit(2*num_channels) for _ in range(7)]
        )
        self.stage4 = nn.Sequential(
            ShuffleNetV2DownsampleUnit(2*num_channels, 4*num_channels),
            *[ShuffleNetV2BasicUnit(4*num_channels) for _ in range(3)]
        )
        self.conv5 = nn.Conv2d(4*num_channels, conv5_out, kernel_size=1)
        self.globalpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv5_out, self.num_classes)
        

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.globalpool(x).flatten(1)
        y = self.fc(x)
        return y


class ShuffleNetV2BasicUnit(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5) -> None:
        super().__init__()
        self.left_channels = int(split_ratio*in_channels)
        self.right_channels = in_channels - self.left_channels
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.right_channels, out_channels=self.right_channels, kernel_size=1),
            nn.BatchNorm2d(self.right_channels),
            nn.ReLU()
        )
        self.dwconv3x3 = nn.Sequential(
            DepthwiseConv2d(in_channels=self.right_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.right_channels)
        )
        self.pwconv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.right_channels, out_channels=self.right_channels, kernel_size=1),
            nn.BatchNorm2d(self.right_channels),
            nn.ReLU()
        )
        self.channelshuffle = ChannelShuffle(2)

    def forward(self, x):
        left, right = torch.split(x, self.left_channels, dim=1)
        right = self.pwconv1(right)
        right = self.dwconv3x3(right)
        right = self.pwconv2(right)
        y = self.channelshuffle(torch.concat([left, right], dim=1))
        return y
        
    
class ShuffleNetV2DownsampleUnit(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.left_dwconv3x3 = nn.Sequential(
            DepthwiseConv2d(in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.left_pwconv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()
        )
        self.right_pwconv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()        
        )
        self.right_dwconv3x3 = nn.Sequential(
            DepthwiseConv2d(in_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.right_pwconv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels//2, kernel_size=1),
            nn.BatchNorm2d(out_channels//2),
            nn.ReLU()        
        )
        self.channelshuffle = ChannelShuffle(2)

    def forward(self, x):
        left = self.left_dwconv3x3(x)
        left = self.left_pwconv(left)
        right = self.right_pwconv1(x)
        right = self.right_dwconv3x3(right)
        right = self.right_pwconv2(right)
        y = self.channelshuffle(torch.concat([left, right], dim=1))
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
    model = ShuffleNetV2(num_classes=1000, scale_factor=2)
    model.eval()
    inputs = torch.rand((5, 3, 224, 224))
    print(model(inputs).shape)    