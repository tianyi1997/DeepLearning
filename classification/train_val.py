import torchvision
import numpy as np

from vision_transformer import VisionTransformer

dataset = torchvision.datasets.FashionMNIST(root='data', download=True)
model = VisionTransformer()
