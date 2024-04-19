# -*- coding: utf-8 -*-
"""
@DATE: 2024-01-05 23:03:12
@Author: Liu Hengjiang
@File: model\multi_task\mmoe.py
@Software: vscode
@Description:
        MMOE模型
"""

import torch
import torch.nn as nn


class MMoELayer(nn.Module):
    def __init__(self,
                 input_size: int = 64,
                 num_experts: int = 3,
                 num_tasks: int = 3,
                 expert_hidden_units: int = 32,
                 gate_hidden_units: int = 32):
        super(MMoELayer, self).__init__()

        self.num_experts = num_experts
        self.num_tasks = num_tasks

        self.expert_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, expert_hidden_units),
                nn.ReLU(),
                nn.Linear(expert_hidden_units, expert_hidden_units),
                nn.ReLU(),
            ) for _ in range(num_experts)
        ])

        self.gate_nets = nn.ModuleList([
            nn.Sequential(nn.Linear(input_size, gate_hidden_units), nn.ReLU(),
                          nn.Linear(gate_hidden_units, num_experts),
                          nn.Softmax(dim=1)) for _ in range(num_tasks)
        ])

        self.task_nets = nn.ModuleList(
            [nn.Linear(expert_hidden_units, 1) for _ in range(num_tasks)])

    def forward(self, x):
        expert_outputs = []
        for expert_net in self.expert_nets:
            expert_output = expert_net(x)
            expert_outputs.append(expert_output)

        gate_outputs = []
        for gate_net in self.gate_nets:
            gate_output = gate_net(x)
            gate_outputs.append(gate_output)

        task_outputs = []
        for i in range(self.num_tasks):
            expert_gates = torch.stack([gate_outputs[i]], dim=1)
            # expert_gates = torch.stack(
            #     [gate_outputs[i] for _ in range(self.num_experts)], dim=2)
            expert_outputs_tensor = torch.stack(expert_outputs, dim=2)
            weighted_expert_outputs = expert_gates.expand_as(expert_outputs_tensor) * expert_outputs_tensor
            # weighted_expert_outputs = expert_gates * expert_outputs_tensor
            task_output = torch.sum(weighted_expert_outputs, dim=2)
            task_output = self.task_nets[i](task_output)
            task_outputs.append(task_output)

        return task_outputs


class MMoEembedding(nn.Module):
    """MMoE for CTCVR problem

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 user_feature_dict,
                 item_feature_dict,
                 emb_dim=128,
                 n_expert=3,
                 mmoe_hidden_dim=128,
                 hidden_dim=[128, 64],
                 dropouts=[0.5, 0.5],
                 output_size=1,
                 expert_activation=None,
                 num_task=2):
        """MMOE model input patameters
        存在exoert_activation时可支持分类任务

        Args:
            user_feature_dict (_type_): _description_
            item_feature_dict (_type_): _description_
            emb_dim (int, optional): _description_. Defaults to 128.
            n_expert (int, optional): _description_. Defaults to 3.
            mmoe_hidden_dim (int, optional): _description_. Defaults to 128.
            hidden_dim (list, optional): _description_. Defaults to [128,64].
            dropouts (list, optional): _description_. Defaults to [0.5,0.5].
            output_size (int, optional): _description_. Defaults to 1.
            expert_activation (_type_, optional): _description_. Defaults to None.
            num_task (int, optional): _description_. Defaults to 2.
        """
        super(MMoEembedding, self).__init__()
        # check input parameters
        if user_feature_dict is None or item_feature_dict is None:
            raise Exception(
                "input parameter user_feature_dict and item_feature_dict must be not None"
            )
        if isinstance(user_feature_dict, dict) is False or isinstance(
                item_feature_dict, dict) is False:
            raise Exception(
                "input parameter user_feature_dict and item_feature_dict must be dict"
            )

        self.user_feature_dict = user_feature_dict
        self.item_feature_dict = item_feature_dict
        self.expert_activation = expert_activation
        self.num_task = num_task

        # embedding 的初始化
        user_cate_feature_nums, item_cate_feature_nums = 0, 0
        for user_cate, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_cate_feature_nums += 1
                setattr(self, user_cate, nn.Embedding(num[0], emb_dim))
        for item_cate, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_cate_feature_nums += 1
                setattr(self, item_cate, nn.Embedding(num[0], emb_dim))

        # user embedding + item embedding
        hidden_size = emb_dim * (
            user_cate_feature_nums + item_cate_feature_nums) + (
                len(self.user_feature_dict) - user_cate_feature_nums) + (
                    len(self.item_feature_dict) - item_cate_feature_nums)

        # experts
        self.experts = torch.nn.Parameter(torch.rand(hidden_size,
                                                     mmoe_hidden_dim,
                                                     n_expert),
                                          requires_grad=True)
        self.experts.data.normal_(0, 1)
        self.experts_bias = torch.nn.Parameter(torch.rand(
            mmoe_hidden_dim, n_expert),
                                               requires_grad=True)
        # gates
        self.gates = nn.ParameterList([
            torch.nn.Parameter(torch.rand(hidden_size, n_expert),
                               requires_grad=True) for _ in range(num_task)
        ])
        for gate in self.gates:
            gate.data.normal_(0, 1)
        self.gates_bias = nn.ParameterList([
            torch.nn.Parameter(torch.rand(n_expert), requires_grad=True)
            for _ in range(num_task)
        ])

        # 独立任务的DNN结构
        for i in range(self.num_task):
            setattr(self, f"task_{i + 1}_dnn", nn.ModuleList())
            hid_dim = [mmoe_hidden_dim] + hidden_dim
            for j in range(len(hid_dim) - 1):
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"dnn_hidden_{j}", nn.Linear(hid_dim[j], hid_dim[j + 1]))
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"dnn_batchnorm_{j}", nn.BatchNorm1d(hid_dim[j + 1]))
                getattr(self, f"task_{i + 1}_dnn").add_module(
                    f"dnn_dropout_{j}", nn.Dropout(dropouts[j]))
            getattr(self, f"task_{i + 1}_dnn").add_module(
                "task_last_layer", nn.Linear(hid_dim[-1], output_size))

    def forward(self, x):
        assert x.size()[1] == len(self.item_feature_dict) + len(
            self.user_feature_dict)
        # embedding
        user_embed_list, item_embed_list = list(), list()
        for user_feature, num in self.user_feature_dict.items():
            if num[0] > 1:
                user_embed_list.append(
                    getattr(self, user_feature)(x[:, num[1]].long()))
            else:
                user_embed_list.append(x[:, num[1]].unsqueeze(1))
        for item_feature, num in self.item_feature_dict.items():
            if num[0] > 1:
                item_embed_list.append(
                    getattr(self, item_feature)(x[:, num[1]].long()))
            else:
                item_embed_list.append(x[:, num[1]].unsqueeze(1))

        # embedding 融合
        user_embed = torch.cat(user_embed_list, axis=1)
        item_embed = torch.cat(item_embed_list, axis=1)

        # hidden layer
        hidden = torch.cat([user_embed, item_embed],
                           axis=1).float()  # batch * hidden_size

        # mmoe
        # batch * mmoe)hidden_size * num_experts
        experts_out = torch.einsum("ij, jkl -> ikl", hidden, self.experts)
        experts_out += self.experts_bias
        if self.expert_activation is not None:
            experts_out = self.expert_activation(experts_out)

        gates_out = list()
        for idx, gate in enumerate(self.gates):
            gate_out = torch.einsum("ab, bc -> ac", hidden,
                                    gate)  # batch * num_experts
            if self.gates_bias:
                gate_out += self.gates_bias[idx]
            gate_out = nn.Softmax(dim=-1)(gate_out)
            gates_out.append(gate_out)

        outs = list()
        for gate_output in gates_out:
            expanded_gate_output = torch.unsqueeze(
                gate_output, 1)  # batch * 1 * num_experts
            # batch * mmoe_hidden_szie * num_experts
            weighted_expert_output = experts_out * expanded_gate_output.expand_as(
                experts_out)
            # batch * mmoe_hidden_size
            outs.append(torch.sum(weighted_expert_output, 2))

        # task tower
        task_outputs = list()
        for i in range(self.num_task):
            x = outs[i]
            for mod in getattr(self, f"task_{i+1}_dnn"):
                x = mod(x)
            task_outputs.append(x)

        return task_outputs


if __name__ == "__main__":
    import numpy as np

    a = torch.from_numpy(
        np.array([[1, 2, 4, 2, 0.5, 0.1], [4, 5, 3, 8, 0.6, 0.43],
                  [6, 3, 2, 9, 0.12, 0.32], [9, 1, 1, 1, 0.12, 0.45],
                  [8, 3, 1, 4, 0.21, 0.67]]))
    user_cate_dict = {
        "user_id": (11, 0),
        "user_list": (12, 3),
        "user_num": (1, 4)
    }
    item_cate_dict = {
        "item_id": (8, 1),
        "item_cate": (6, 2),
        "item_num": (1, 5)
    }
    mmoe = MMoEembedding(user_cate_dict, item_cate_dict)
    outs = mmoe(a)
    print(outs)
    data = torch.rand(size=(1, 64))
    model = MMoELayer()
    model(data)
