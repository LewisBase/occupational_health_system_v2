# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-16 10:50:09
@Author: Liu Hengjiang
@File: model\conv_layer\cnnmodel.py
@Software: vscode
@Description:
        卷积层定义
"""
import torch
import torch.nn as nn
from typing import Union


class CNNModel(nn.Module):
    def __init__(self,
                 conv1_in_channels: int = 1,
                 conv1_out_channels: int = 32,
                 conv1_kernel_size: Union[int, tuple] = 3,
                 pool1_kernel_size: Union[int, tuple] = 2,
                 pool1_stride: int = 2,
                 conv2_out_channels: int = 64,
                 conv2_kernel_size: Union[int, tuple] = 5,
                 pool2_kernel_size: Union[int, tuple] = 2,
                 pool2_stride: int = 2,
                 dropout_rate: float = 0.1,
                 adaptive_pool_output_size: tuple = (1, 1),
                 linear1_out_features: int = 32,
                 linear2_out_features: int = 1):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=conv1_in_channels,
                      out_channels=conv1_out_channels,
                      kernel_size=conv1_kernel_size),
            nn.MaxPool2d(kernel_size=pool1_kernel_size, stride=pool1_stride),
            nn.Conv2d(in_channels=conv1_out_channels,
                      out_channels=conv2_out_channels,
                      kernel_size=conv2_kernel_size),
            nn.MaxPool2d(kernel_size=pool2_kernel_size, stride=pool2_stride),
            nn.Dropout2d(p=dropout_rate),
            nn.AdaptiveMaxPool2d(output_size=adaptive_pool_output_size))
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv2_out_channels, out_features=linear1_out_features),
            nn.ReLU(),
            nn.Linear(in_features=linear1_out_features,
                      out_features=linear2_out_features),
        )

    def forward(self, x):
        x = self.conv(x)
        y = self.dense(x)
        return y


if __name__ == "__main__":
    model = CNNModel()
    data = torch.randn(size=(1, 1, 400, 80))  # batch_size, in_channel, H, W
    model(data)
