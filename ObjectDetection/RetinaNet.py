import torch
from torch import nn
from ResNet50_FPN import ResNet50_FPN


class RetinaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forwards(self, x):
        pass

class BoxRegressionSubnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forwards(self, x):
        pass

class ClassificationSubnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forwards(self, x):
        pass


class ResNet50_FPN(ResNet50_FPN):
    """
        ResNet50-FPN: backbone for RetinaNet
        Downsample is adjusted.
    """
    def __init__(self, feature_dims=256) -> None:
        super().__init__(feature_dims)
        self.downsample1 = nn.Conv2d(in_channels=feature_dims, out_channels=feature_dims, kernel_size=3, stride=2, padding=1)
        self.downsample2 = nn.Conv2d(in_channels=feature_dims, out_channels=feature_dims, kernel_size=3, stride=2, padding=1)

    def forward(self, c0):
        c1 = self.stage1(c0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        p5 = self.fp5(c5)
        p4 = self.fp4(c4, p5)
        p3 = self.fp3(c3, p4)
        p6 = self.downsample1(p5)
        p7 = self.downsample1(p6)
        return [p7, p6, p5, p4, p3]

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet50_FPN().to(device)
    inputs = torch.rand((5, 3, 512, 512)).to(device)
    model.eval()
    for p in model(inputs):
        print(p.shape)