import torch
from torch import nn
from ResNet50_FPN import ResNet50_FPN


class RetinaNet(nn.Module):
    def __init__(self, num_classes, num_anchors, in_channels, backbone) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        self.backbone = backbone
        self.cls_net = ClassificationSubnet(num_classes, num_anchors, in_channels)
        self.box_net = BoxRegressionSubnet(num_anchors, in_channels)

    def forward(self, x):
        feature_pyramid = self.backbone(x)
        all_cls = self.cls_net(feature_pyramid)
        all_box = self.box_net(feature_pyramid)
        return all_cls, all_box


class BoxRegressionSubnet(nn.Module):
    def __init__(self, num_anchors, in_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
        )
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4*num_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        all_box = []
        for p in x:
            bbox = self.bbox_reg(self.conv(p))
            all_box.append(bbox)
        return all_box


class ClassificationSubnet(nn.Module):
    def __init__(self, num_classes, num_anchors, in_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(in_channels, in_channels ,kernel_size=3, padding=1), nn.ReLU(),
        )
        self.cls_logits = nn.Sequential(
            nn.Conv2d(in_channels, num_classes*num_anchors, kernel_size=3, padding=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        all_cls = []
        for p in x:
            cls = self.cls_logits(self.conv(p))
            all_cls.append(cls)
        return all_cls


class ResNet50_FPN(ResNet50_FPN):
    """
        ResNet50-FPN: backbone for RetinaNet
        Downsample is adjusted.
    """
    def __init__(self, feature_dims) -> None:
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
        return p7, p6, p5, p4, p3


if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    model = RetinaNet(10, 9, 256, ResNet50_FPN(256)).to(device)
    inputs = torch.rand((5, 3, 512, 512)).to(device)
    model.eval()
    for p in model(inputs):
        for x in p:
            print(x.shape)