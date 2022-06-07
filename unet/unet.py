# code of our PyTorch implementation of the U-Net model
# author: Pinglei He

import torch
import torch.nn as nn


class UNet(nn.Module):

    @classmethod
    def _conv(cls, in_channels: int, out_channels: int):
        # down sampling block with two convolutions
        # e.g. 1 channel -> Conv2D -> 128 channels -> Conv2D -> 128 channels
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, dtype=torch.double),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, dtype=torch.double),
            nn.ReLU()
        )

    @classmethod
    def _up_conv(cls, in_channels: int, out_channels: int):
        # up sampling block with transposed convolutions
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, dtype=torch.double)

    @classmethod
    def _crop(cls, input_tensor, output_tensor_like):
        in_dim = input_tensor.size()[2]  # number of pixels per border
        out_dim = output_tensor_like.size()[2]  # number of pixels per border
        delta = (in_dim - out_dim) // 2  # number of pixels to crop per border
        return input_tensor[..., delta:delta + out_dim, delta:delta + out_dim]

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_down_0 = self._conv(self.in_channels, 64)
        self.conv_down_1 = self._conv(64, 128)
        self.conv_down_2 = self._conv(128, 256)
        self.conv_down_3 = self._conv(256, 512)
        self.conv_down_4 = self._conv(512, 1024)

        self.up_conv_4_3 = self._up_conv(1024, 512)
        self.conv_up_3 = self._conv(1024, 512)

        self.up_conv_3_2 = self._up_conv(512, 256)
        self.conv_up_2 = self._conv(512, 256)

        self.up_conv_2_1 = self._up_conv(256, 128)
        self.conv_up_1 = self._conv(256, 128)

        self.up_conv_1_0 = self._up_conv(128, 64)
        self.conv_up_0 = self._conv(128, 64)

        self.out = nn.Conv2d(64, self.out_channels, kernel_size=1, stride=1, dtype=torch.double)

    def forward(self, x):
        # encoder
        x0 = self.conv_down_0(x)
        x1 = self.conv_down_1(self.max_pool(x0))
        x2 = self.conv_down_2(self.max_pool(x1))
        x3 = self.conv_down_3(self.max_pool(x2))
        x4 = self.conv_down_4(self.max_pool(x3))
        # decoder
        x = self.up_conv_4_3(x4)
        x3 = self._crop(x3, x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv_up_3(x)
        x = self.up_conv_3_2(x)
        x2 = self._crop(x2, x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up_2(x)

        x = self.up_conv_2_1(x)
        x1 = self._crop(x1, x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up_1(x)

        x = self.up_conv_1_0(x)
        x0 = self._crop(x0, x)
        x = torch.cat([x0, x], dim=1)
        x = self.conv_up_0(x)
        x = self.out(x)

        return x
