import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from .unet_parts import *
import torch
from torch.nn import Linear, Flatten
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = 16
        self.inc = (DoubleConv(n_channels, self.filters))
        self.down1 = (Down(self.filters, self.filters*2))
        self.down2 = (Down(self.filters*2, self.filters*4))
        self.down3 = (Down(self.filters*4, self.filters*8))
        factor = 2 if bilinear else 1
        self.down4 = (Down(self.filters*8, self.filters*16 // factor))
        self.up1 = (Up(self.filters*16, self.filters*8 // factor, bilinear))
        self.up2 = (Up(self.filters*8, self.filters*4 // factor, bilinear))
        self.up3 = (Up(self.filters*4, self.filters*2 // factor, bilinear))
        self.up4 = (Up(self.filters*2,self.filters, bilinear))
        self.outc = (OutConv(self.filters, n_classes))
        self.dense1 = Linear(self.filters**(2)*16*16 , self.filters**2, bias=True)
        self.dense2 = Linear(self.filters**2, self.filters**(2)*16*16, bias=True)
        self.flatten = Flatten(start_dim=1, end_dim=-1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        shape = x5.shape
        x6 = self.dense1(self.flatten(x5))
        x7 = self.dense2(x6)
        x8 = torch.reshape(x7, shape)
        x = self.up1(x8)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)
        return logits, x1, x2, x3, x4, x6

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)