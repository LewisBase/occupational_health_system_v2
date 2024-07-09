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


class CustomLayer(nn.Module):

    def __init__(self, input_size):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.rand(1, input_size))
        self.batch_norm = nn.BatchNorm1d(input_size)
        nn.init.uniform_(self.weight, a=0, b=1)

    def forward(self, x):
        # 自定义运算
        col1 = x[:, 0, :]
        col2 = x[:, 1, :]
        out = col1 + torch.log10(col2/3) * self.weight
        # out = self.batch_norm(out)
        LAeq_out = 10 * torch.log10(torch.mean(10**(out/10), dim=1, keepdim=True))
        return LAeq_out


class SegmentAdjustModel(nn.Module):

    def __init__(self, in_feautres: int, out_feautres: int):
        super(SegmentAdjustModel, self).__init__()
        self.custom_layer = CustomLayer(input_size=in_feautres)
        self.linear = nn.Linear(in_features=1,
                                out_features=out_feautres)

    def forward(self, x):
        x = self.custom_layer(x)
        y = self.linear(x)
        return y


if __name__ == "__main__":
    seg_adjust = SegmentAdjustModel(in_feautres=480, out_feautres=1)
    params = seg_adjust.state_dict()
    data = torch.rand(size=(128 , 2, 480))
    res = seg_adjust(data)
