import torch
from torch import nn
from torch.nn import functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma, alpha=1, reduce='mean') -> None:
        assert reduce in ['sum', 'mean']
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, x, y):
        if y.shape[0] == 1:
            y = y.view(-1, 1)
        p =  F.softmax(x, dim=1)
        p_t = torch.gather(p, 1, y)
        fl = - self.alpha * torch.pow((1 - p_t), self.gamma) * torch.log(p_t)
        if self.reduce == 'mean':
            fl = fl.mean()
        elif self.reduce == 'sum':
            fl = fl.sum()
        return fl


if __name__ == '__main__':
    num_classes = 10
    num_examples = 5
    x = torch.rand((num_examples, num_classes))
    y = torch.randint(0, num_classes - 1, (num_examples, 1))
    loss = FocalLoss(gamma=1)
    print(loss(x, y))
