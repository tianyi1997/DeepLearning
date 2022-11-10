import torch
from torch import nn

class ResNeXt(nn.Module):
    def __init__(self, n_classes, cardinality) -> None:
        super().__init__()
        self.cardinality = cardinality
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResNeXtBottleneck(64, 128, 256, 3, cardinality, False),
            *[ResNeXtBottleneck(256, 128, 256, 3, cardinality, False) for _ in range(2)]
        )
        self.conv3 = nn.Sequential(
            ResNeXtBottleneck(256, 256, 512, 3, cardinality, True),
            *[ResNeXtBottleneck(512, 256, 512, 3, cardinality, False) for _ in range(3)]
        )

        self.conv4 = nn.Sequential(
            ResNeXtBottleneck(512, 512, 1024, 3, cardinality, True),
            *[ResNeXtBottleneck(1024, 512, 1024, 3, cardinality, False) for _ in range(5)]
        )

        self.conv5 = nn.Sequential(
            ResNeXtBottleneck(1024, 1024, 2048, 3, cardinality, True),
            *[ResNeXtBottleneck(2048, 1024, 2048, 3, cardinality, False) for _ in range(2)]
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.global_avgpool(x).flatten(1)
        y = self.fc(x)
        return y

class ResNeXtBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, cardinality, downsample) -> None:
        super().__init__()
        self.conv1x1_down = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=1
                ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU()
            )
        stride = (2 if downsample else 1)
        self.groupconv = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels, 
                stride=stride, 
                kernel_size=kernel_size, 
                padding=kernel_size//2, 
                groups=cardinality
                ),
            nn.BatchNorm2d(num_features=hidden_channels),
            nn.ReLU()
            )
        self.conv1x1_up = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(num_features=out_channels),
            )
        if downsample == True or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
                )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv1x1_down(x)
        res = self.groupconv(res)
        res = self.conv1x1_up(res)
        x = self.shortcut(x)
        y = self.relu(x + res)
        return y


if __name__ == '__main__':
    model = ResNeXt(1000, 32)
    model.eval()
    inputs = torch.rand((5, 3, 224, 224))
    print(model(inputs).shape)