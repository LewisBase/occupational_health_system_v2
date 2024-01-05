
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class MMoE(nn.Module):
    def __init__(self, num_experts, num_tasks, num_factors, hidden_units, dropout_rate):
        super(MMoE, self).__init__()
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.num_factors = num_factors
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # User Embedding Layer
        self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=num_factors)
        
        # Expert Layer
        self.expert_layers = nn.ModuleList([nn.Linear(num_factors, hidden_units) for _ in range(num_experts)])
        
        # Gate Layer
        self.gate_layers = nn.ModuleList([nn.Linear(num_factors, num_experts) for _ in range(num_tasks)])
        
        # Task Tower Layers
        self.task_towers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, 1)
        ) for _ in range(num_tasks)])

    def forward(self, x):
        # User Embedding
        user_emb = self.user_embedding(x['user'])
        
        # Expert Layer
        expert_outputs = []
        for expert_layer in self.expert_layers:
            expert_outputs.append(F.relu(expert_layer(user_emb)))
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Gate Layer
        gate_outputs = []
        for gate_layer in self.gate_layers:
            gate_outputs.append(F.softmax(gate_layer(user_emb), dim=-1))
        gate_outputs = torch.stack(gate_outputs, dim=1)
        
        # Expert Outputs x Gate Outputs
        expert_gate = torch.matmul(gate_outputs.unsqueeze(-1), expert_outputs.unsqueeze(1))
        expert_gate = expert_gate.squeeze(-1)
        
        # Task Tower Layers
        task_outputs = []
        for i in range(self.num_tasks):
            task_emb = expert_gate[:, i, :]
            task_outputs.append(self.task_towers[i](task_emb))
        task_outputs = torch.cat(task_outputs, dim=1)

        return task_outputs