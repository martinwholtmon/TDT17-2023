import torch
import torch.nn as nn
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
    def forward(self, x):
        pass
