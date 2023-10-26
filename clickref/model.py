import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Softmax, Sigmoid
import torch.nn.functional as F
from config import *


class Conv2d_new(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.relu = ReLU()

    def forward(self, x):
        x_relu = self.relu(x)
        x_new = self.conv(x_relu)

        return x + x_new


class Click_ref(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, filters=8):
        super(Click_ref, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.filters = filters
        self.relu = ReLU()
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()

        self.conv_corr1 = Conv2d(
            self.n_channels, self.filters * 8, kernel_size=(3, 3), padding=1
        )
        self.conv_corr2 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(5, 5), padding=2
        )
        self.conv_corr3 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(7, 7), padding=3
        )
        self.conv_corr4 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(9, 9), padding=4
        )
        self.conv_corr5 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(7, 7), padding=3
        )
        self.conv_corr6 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(5, 5), padding=2
        )
        self.conv_corr7 = Conv2d(
            self.filters * 8, self.n_classes, kernel_size=(3, 3), padding=1
        )

        self.conv_prob1 = Conv2d(
            self.n_channels, self.filters * 8, kernel_size=(3, 3), padding=1
        )
        self.conv_prob2 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(5, 5), padding=2
        )
        self.conv_prob3 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(7, 7), padding=3
        )
        self.conv_prob4 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(9, 9), padding=4
        )
        self.conv_prob5 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(7, 7), padding=3
        )
        self.conv_prob6 = Conv2d_new(
            self.filters * 8, self.filters * 8, kernel_size=(5, 5), padding=2
        )
        self.conv_prob7 = Conv2d(self.filters * 8, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x, stardist):
        x1_corr = self.conv_corr1(x)
        x2_corr = self.conv_corr2(x1_corr)
        x3_corr = self.conv_corr3(x2_corr)
        x4_corr = self.conv_corr4(x3_corr)
        x5_corr = self.conv_corr5(x4_corr)
        x6_corr = self.conv_corr6(x5_corr)
        x7_corr = self.softmax(self.conv_corr7(x6_corr))

        x1_mask = self.conv_prob1(x)
        x2_mask = self.conv_prob2(x1_mask)
        x3_mask = self.conv_prob3(x2_mask)
        x4_mask = self.conv_prob4(x3_mask)
        x5_mask = self.conv_prob5(x4_mask)
        x6_mask = self.conv_prob6(x5_mask)
        x7_mask = self.sigmoid(self.conv_prob7(x6_mask))
        x7_mask = torch.concat([x7_mask, x7_mask, x7_mask], dim=1)

        output = torch.log(
            torch.clip(stardist * (1.0 - x7_mask) + x7_mask * x7_corr, 1e-6, 1 - 1e-6)
        )

        return output
