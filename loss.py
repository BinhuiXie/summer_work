import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class DiscrepancyLoss2d(nn.Module):
    def __init__(self, weight=None):
        super(DiscrepancyLoss2d, self).__init__()
        self.weight = weight

    def forward(self, inputs1, inputs2):
        return torch.mean(torch.abs(F.softmax(inputs1) - F.softmax(inputs2)))
