import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, Softmax, Sigmoid
import torch.nn.functional as F
from config import *


# class Conv2d_new(nn.Module):
#     """Upscaling then double conv"""

#     def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.padding = padding
#         self.bias = bias
#         self.conv = Conv2d(
#             self.in_channels,
#             self.out_channels,
#             kernel_size=self.kernel_size,
#             padding=self.padding,
#             bias = self.bias
#         )
#         self.relu = torch.nn.LeakyReLU()

#     def forward(self, x):
#         x_relu = self.relu(x)
#         x_new = self.conv(x_relu)

#         return x + x_new


class Click_ref(nn.Module):
    def __init__(self, n_channels, n_classes, filters = 8):
        super(Click_ref, self).__init__()


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
            16 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )
        

        self.conv_mask4 = Conv2d(
            8 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_mask5= Conv2d(
            8 * self.filters, 4 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_mask6= Conv2d(
            4 * self.filters, 1, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_click1 = Conv2d(
            4, self.filters, kernel_size=(3, 3), padding=1, bias=True
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
            16 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )
        

        self.conv_corr4 = Conv2d(
            8 * self.filters, 8 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_corr5= Conv2d(
            8 * self.filters, 4 * self.filters, kernel_size=(3, 3), padding=1, bias=True
        )

        self.conv_corr6= Conv2d(
            4 * self.filters, 3, kernel_size=(3, 3), padding=1, bias=True
        )

        self.maxpool = torch.nn.MaxPool2d((3,3), stride=2,padding =1)
        self.upsample1 = torch.nn.Upsample(size=(256//4,256//4), mode='bilinear')
        self.upsample2 = torch.nn.Upsample(size=(256//2,256//2), mode='bilinear')
        self.upsample3 = torch.nn.Upsample(size=(256,256), mode='bilinear')

    def forward(self, image, click, stardist):

        x1_image = self.conv_image1(image)
        x1_click = self.conv_click1(click)

        cat1 = torch.cat((x1_image, x1_click),dim=1)
        x1_image = self.maxpool(self.relu(cat1))
   
        x2_image = self.conv_image2(x1_image)
        x2_click = self.conv_click2(self.maxpool(self.relu(x1_click)))
        cat2 = torch.cat((x2_image, x2_click),dim=1)
        x2_image = self.maxpool(self.relu(cat2))

        x3_image = self.conv_image3(x2_image)
        x3_click = self.conv_click3(self.maxpool(self.relu(x2_click)))
        cat3 = torch.cat((x3_image, x3_click),dim=1)
        x3_image = self.maxpool(self.relu(cat3))

        x4_image = self.conv_image4(x3_image)
        x4_click = self.conv_click4(self.maxpool(self.relu(x3_click)))
        cat4 = torch.cat((x4_image, x4_click),dim=1)
        x4_image = self.relu(cat4)

        ### correction branch

        x_corr1 = self.upsample1(self.relu(self.conv_corr1(x4_image)))
        x_corr2 = self.relu(self.conv_corr2(x_corr1 + 0.))
        x_corr3 = self.upsample2(self.relu(self.conv_corr3(x_corr2)))
        x_corr4 = self.relu(self.conv_corr4(x_corr3 + 0.))
        x_corr5 = self.upsample3(self.relu(self.conv_corr5(x_corr4)))
        x_corr6 = self.softmax(self.conv_corr6(x_corr5 + 0.))


        ### mask branch
        x_mask1 = self.upsample1(self.relu(self.conv_mask1(x4_image)))
        x_mask2 = self.relu(self.conv_mask2(x_mask1 + cat3))
        x_mask3 = self.upsample2(self.relu(self.conv_mask3(x_mask2)))
        x_mask4 = self.relu(self.conv_mask4(x_mask3 + cat2))
        x_mask5 = self.upsample3(self.relu(self.conv_mask5(x_mask4)))
        x_mask6 = self.sigmoid(self.conv_mask6(x_mask5 + cat1))   
        output = stardist * (1.0 - x_mask6) + x_mask6 * x_corr6

        return output

