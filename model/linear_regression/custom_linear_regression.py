# -*- coding: utf-8 -*-
"""
@DATE: 2024-07-05 16:23:12
@Author: Liu Hengjiang
@File: model\linear_regression\custom_linear_regression.py
@Software: vscode
@Description:
        构建自定义的多层次线性回归模型，用于复杂噪声声压级峰度校正方法中的分段校正
"""

import torch
import torch.nn as nn
from typing import Union


class SegmentAdjustTestModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(SegmentAdjustTestModel, self).__init__()
        self.linear1 = nn.Linear(in_features=in_features,
                                 out_features=1)
        self.linear2 = nn.Linear(in_features=1,
                                 out_features=out_features)
        
    def forward(self, x):
        x = self.linear1(x)
        x = x[:,0,:] + x[:,1,:]
        y = self.linear2(x)
        return y
    
    
class CustomLayer(nn.Module):

    def __init__(self, input_size):
        super(CustomLayer, self).__init__()
        self.weight_0 = nn.Parameter(torch.rand(1, input_size))
        nn.init.uniform_(self.weight_0, a=0, b=1)
        self.weight_1 = nn.Parameter(torch.rand(1, input_size))
        nn.init.uniform_(self.weight_1, a=0, b=1)
        # self.weight = nn.Parameter(torch.rand(1, input_size))
        # nn.init.uniform_(self.weight, a=0, b=1)

    def forward(self, x):
        # 自定义运算
        col_0 = x[:, 0, :]
        col_1 = x[:, 1, :]
        col_1 = torch.where(col_1 <= 0, torch.tensor(1e-10, device=col_1.device), col_1)
        out = col_0 * self.weight_0 + torch.log10(col_1/3) * self.weight_1
        # out = col0 + torch.log10(col1/3) * self.weight
        LAeq_out = 10 * torch.log10(torch.mean(10**(out/10), dim=1, keepdim=True))
        return LAeq_out

    
class CustomLayerMono(nn.Module):
    def __init__(self):
        super(CustomLayerMono, self).__init__()
        self.hat_lambda = nn.Parameter(torch.rand(1))
        nn.init.uniform_(self.hat_lambda, a=1, b=6)

    def forward(self, x):
        # 自定义运算
        col_0 = x[:, 0, :]
        col_1 = x[:, 1, :]
        col_1 = torch.where(col_1 <= 0, torch.tensor(1e-10, device=col_1.device), col_1)
        out = col_0 + torch.log10(col_1/3) * self.hat_lambda
        LAeq_out = 10 * torch.log10(torch.mean(10**(out/10), dim=1, keepdim=True))
        return LAeq_out


class SegmentAdjustModel(nn.Module):

    def __init__(self, in_features: int, out_features: int):
        super(SegmentAdjustModel, self).__init__()
        # self.custom_layer = CustomLayer(input_size=in_features)
        self.custom_layer = CustomLayerMono()
        self.linear = nn.Linear(in_features=1,
                                out_features=out_features)
        self._initialize_weights()

    def forward(self, x):
        x = self.custom_layer(x)
        y = self.linear(x)
        return y

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.linear.weight, a=0, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.linear.bias, 0)


if __name__ == "__main__":
    seg_adjust = SegmentAdjustModel(in_features=480, out_features=1)
    params = seg_adjust.state_dict()
    data = torch.rand(size=(128 , 2, 480))
    res = seg_adjust(data)
