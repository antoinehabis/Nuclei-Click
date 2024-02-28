import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Softmax, Sigmoid
import torch.nn.functional as F
from config import *


class Clickref(nn.Module):
    def __init__(self, n_channels, n_classes, filters=8):
        super(Clickref, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.filters = filters
        self.relu = torch.nn.ReLU()
        self.softmax = Softmax(dim=1)
        self.sigmoid = Sigmoid()
        self.conv_image1 = Conv2d(
            self.n_channels, 3 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )
        self.conv_image2 = Conv2d(
            4 * self.filters,
            6 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_image3 = Conv2d(
            8 * self.filters,
            12 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

        self.conv_image4 = Conv2d(
            16 * self.filters,
            24 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_mask1 = Conv2d(
            32 * self.filters,
            16 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_mask2 = Conv2d(
            16 * self.filters,
            16 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

        self.conv_mask3 = Conv2d(
            16 * self.filters,
            8 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

        self.conv_mask4 = Conv2d(
            8 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_mask5 = Conv2d(
            8 * self.filters, 4 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_mask6 = Conv2d(
            4 * self.filters, 1, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_click1 = Conv2d(
            7, self.filters, kernel_size=(3, 3), padding=1, bias=True
        )
        self.conv_click2 = Conv2d(
            self.filters,
            2 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_click3 = Conv2d(
            2 * self.filters,
            4 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_click4 = Conv2d(
            4 * self.filters,
            8 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

        self.conv_corr1 = Conv2d(
            32 * self.filters,
            16 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_corr2 = Conv2d(
            16 * self.filters,
            16 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )
        self.conv_corr3 = Conv2d(
            16 * self.filters,
            8 * self.filters,
            kernel_size=(3, 3),
            padding=1,
            bias=True,
        )

        self.conv_corr4 = Conv2d(
            8 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_corr5 = Conv2d(
            8 * self.filters, 4 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_corr6 = Conv2d(
            4 * self.filters, 3, kernel_size=(3, 3), padding=1, bias=True
        )

        self.maxpool = torch.nn.MaxPool2d((2, 2), stride=2, padding=0)
        self.upsample1 = torch.nn.Upsample(size=(256 // 4, 256 // 4), mode="bilinear")
        self.upsample2 = torch.nn.Upsample(size=(256 // 2, 256 // 2), mode="bilinear")
        self.upsample3 = torch.nn.Upsample(size=(256, 256), mode="bilinear")

    def forward(self, image, click, baseline):

        x1_image = self.conv_image1(image)
        x1_click = self.conv_click1(torch.cat((click, baseline), dim=1))

        cat1 = torch.cat((x1_image, x1_click), dim=1)
        x1_image = self.maxpool(self.relu(cat1))

        x2_image = self.conv_image2(x1_image)
        x2_click = self.conv_click2(self.maxpool(self.relu(x1_click)))
        cat2 = torch.cat((x2_image, x2_click), dim=1)
        x2_image = self.maxpool(self.relu(cat2))

        x3_image = self.conv_image3(x2_image)
        x3_click = self.conv_click3(self.maxpool(self.relu(x2_click)))
        cat3 = torch.cat((x3_image, x3_click), dim=1)
        x3_image = self.maxpool(self.relu(cat3))

        x4_image = self.conv_image4(x3_image)
        x4_click = self.conv_click4(self.maxpool(self.relu(x3_click)))
        cat4 = torch.cat((x4_image, x4_click), dim=1)
        x4_image = self.relu(cat4)

        ### correction branch

        x_corr1 = self.upsample1(self.relu(self.conv_corr1(x4_image)))
        x_corr2 = self.relu(self.conv_corr2(x_corr1 + cat3))
        x_corr3 = self.upsample2(self.relu(self.conv_corr3(x_corr2)))
        x_corr4 = self.relu(self.conv_corr4(x_corr3 + cat2))
        x_corr5 = self.upsample3(self.relu(self.conv_corr5(x_corr4)))
        x_corr6 = torch.tanh(self.conv_corr6(x_corr5 + cat1))

        return x_corr6
