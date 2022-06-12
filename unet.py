import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary

class UNet(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.ModuleList()
        self.deconv = nn.ModuleList()
        prev, next = in_channels, 64
        for _ in range(5):
            self.conv.append(nn.Conv2d(prev, next, 3, 2, 1))
            prev, next = next, next*2
        next = prev//2
        for _ in range(5):
            self.deconv.append(nn.ConvTranspose2d(prev*2, next, 3, 2, 1, 1))
            prev, next = next, next//2
        self.final = nn.Conv2d(prev, 1, 1)

    def forward(self, x):
        skip = []
        for layer in self.conv:
            x = F.relu(layer(x))
            skip.append(x)
        for i, layer in enumerate(self.deconv):
            y = torch.cat((x, skip[-i-1]), 1)
            x = F.relu(layer(y))
        x = self.final(x)
        return x

if __name__ == '__main__':
    unet = UNet(1)
    print(summary(unet, (10, 1, 1024, 1024)))
    print(unet(torch.zeros((10, 1, 1024, 1024))).shape)
