import torch
from torch import nn

class ResNet50_FPN(nn.Module):
    def __init__(self, feature_dims=256) -> None:
        super().__init__()
        self.stage1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.stage2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNetBottleneck(64, 64, 256, False),
            *[ResNetBottleneck(256, 64, 256, False) for _ in range(2)]
        )
        self.stage3 = nn.Sequential(
            ResNetBottleneck(256, 128, 512, True),
            *[ResNetBottleneck(512, 128, 512, False) for _ in range(3)]            
        )
        self.stage4 = nn.Sequential(
            ResNetBottleneck(512, 256, 1024, True),
            *[ResNetBottleneck(1024, 256, 1024, False) for _ in range(5)]            
        )
        self.stage5 = nn.Sequential(
            ResNetBottleneck(1024, 512, 2048, True),
            *[ResNetBottleneck(2048, 512, 2048, False) for _ in range(2)]            
        )
        self.fp5 = nn.Conv2d(in_channels=2048, out_channels=feature_dims, kernel_size=1)
        self.fp4 = FPBlock(1024, feature_dims)
        self.fp3 = FPBlock(512, feature_dims)
        self.fp2 = FPBlock(256, feature_dims)
        self.downsample = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, c0):
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.fp5(c5)
        p4 = self.fp4(c4, p5)
        p3 = self.fp3(c3, p4)
        p2 = self.fp2(c2, p3)
        p6 = self.downsample(p5)
        return [p6, p5, p4, p3, p2]

class FPBlock(nn.Module):
    def __init__(self, low_channels, feature_dims) -> None:
        super().__init__()
        self.lateral_path = nn.Conv2d(
            in_channels=low_channels,
            out_channels=feature_dims,
            kernel_size=1
            )
        self.topdown_path = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3x3 = nn.Conv2d(
            in_channels=feature_dims,
            out_channels=feature_dims,
            kernel_size=3,
            padding=1
        )

    def forward(self, low, high):
        p = self.conv3x3(self.lateral_path(low) + self.topdown_path(high))
        return p

class ResNetBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,  downsample) -> None:
        super().__init__()
        stride = (2 if downsample else 1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1,
                stride=stride
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(out_channels)
        )
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        y = self.relu(res + self.shortcut(x))       
        return y

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50_FPN().to(device)
    inputs = torch.rand((5, 3, 512, 512)).to(device)
    model.eval()
    for p in model(inputs):
        print(p.shape)