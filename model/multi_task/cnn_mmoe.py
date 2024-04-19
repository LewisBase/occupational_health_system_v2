# -*- coding: utf-8 -*-
"""
@DATE: 2024-04-16 10:52:04
@Author: Liu Hengjiang
@File: model\multi_task\cnn_mmoe.py
@Software: vscode
@Description:
        卷积+MMoE神经网络
"""
import torch
import torch.nn as nn
from typing import Union

from model.conv_layer.cnn import CNNModel
from model.multi_task.mmoe import MMoELayer


class ConvMMoEModel(nn.Module):
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
                 linear2_out_features: int = 16,
                 num_experts: int = 3,
                 num_tasks: int = 3,
                 expert_hidden_units: int = 32,
                 gate_hidden_units: int = 32
                 ):
        super(ConvMMoEModel, self).__init__()
        self.cnn = CNNModel(conv1_in_channels=conv1_in_channels,
                            conv1_out_channels=conv1_out_channels,
                            conv1_kernel_size=conv1_kernel_size,
                            pool1_kernel_size=pool1_kernel_size,
                            pool1_stride=pool1_stride,
                            conv2_out_channels=conv2_out_channels,
                            conv2_kernel_size=conv2_kernel_size,
                            pool2_kernel_size=pool2_kernel_size,
                            pool2_stride=pool2_stride,
                            dropout_rate=dropout_rate,
                            adaptive_pool_output_size=adaptive_pool_output_size,
                            linear1_out_features=linear1_out_features,
                            linear2_out_features=linear2_out_features)
        self.mmoe = MMoELayer(input_size=linear2_out_features,
                              num_experts=num_experts,
                              num_tasks=num_tasks,
                              expert_hidden_units=expert_hidden_units,
                              gate_hidden_units=gate_hidden_units)

    def forward(self, x):
        x = self.cnn(x)
        y = self.mmoe(x)
        return y

        
if __name__ == "__main__":
    cnn_mmoe = ConvMMoEModel()
    data = torch.randn(size=(1,1,400,22))
    cnn_mmoe(data)
    