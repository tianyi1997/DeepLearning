import torch
import torch.nn as nn
import torch.nn.functional as F

class SEResNet(nn.Module):
    # ResNet18
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn0 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1)
        self.layer1 = nn.Sequential(SEResNetBlock(64, 64, False), SEResNetBlock(64, 64, False))
        self.layer2 = nn.Sequential(SEResNetBlock(64, 128, True), SEResNetBlock(128, 128, False))
        self.layer3 = nn.Sequential(SEResNetBlock(128, 256, True), SEResNetBlock(256, 256, False))
        self.layer4 = nn.Sequential(SEResNetBlock(256, 512, True), SEResNetBlock(512, 512, False))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_out_feature, bias=True)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = self.conv0(x)
        out = self.bn0(out)
        out = self.leakyrelu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.maxpool(out)
        out = self.leakyrelu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


class SEResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample, reduction=16):
        super(SEResNetBlock, self).__init__()

        self.leakyrelu0 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=((2, 2) if down_sample else (1, 1)), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)

        self.se = SELayer(out_channel, reduction)

        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.leakyrelu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.se(output)
        output += self.shortcut(x)
        output = self.leakyrelu2(output)
        return output


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x*y.expand_as(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, down_sample):
        super(ResNetBlock, self).__init__()

        self.leakyrelu0 = nn.LeakyReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), stride=((2, 2) if down_sample else (1, 1)), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.leakyrelu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1)
        self.leakyrelu2 = nn.LeakyReLU(inplace=True)

        if down_sample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=(1, 1), stride=(2, 2), bias=False),
                nn.BatchNorm2d(out_channel, eps=1e-05, momentum=0.1))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.leakyrelu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output += self.shortcut(x)
        output = self.leakyrelu2(output)
        return output