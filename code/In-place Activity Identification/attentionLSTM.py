# -*- coding: utf-8 -*-
"""
Created on Tue May 23 22:05:38 2023

@author: User
"""

import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs):
        energy = torch.tanh(encoder_outputs)
        attention_scores = torch.matmul(energy, self.attention_weights)
        attention_weights = self.softmax(attention_scores.squeeze(-1))
        context_vector = torch.matmul(encoder_outputs.transpose(1, 2), attention_weights.unsqueeze(-1)).squeeze(-1)
        return context_vector

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        context_vector = self.attention(h_n)
        flattened = self.flatten(context_vector)
        fc1_output = torch.relu(self.fc1(flattened))
        fc2_output = torch.relu(self.fc2(fc1_output))
        output = self.fc3(fc2_output)
        output = self.softmax(output)
        return output

batch_size = 50
sequence_length = 1
input_size = 200 * 2
hidden_size = 64
num_layers = 2
output_size = 6

model = LSTMAttentionModel(input_size, hidden_size, num_layers, output_size)
input_data = torch.randn(batch_size, sequence_length, input_size)
output = model(input_data)


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainable parameters: ", total_params)


