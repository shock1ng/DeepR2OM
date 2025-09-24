# -*- coding: utf-8 -*-
# @Time : 2024/4/6 14:54
# @Author : JohnnyYuan
# @File : classifier.py
import torch.nn as nn
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, width = x.size()

        # Compute query, key, and value
        proj_query = self.query_conv(x).view(batch_size, -1, width).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch_size, -1, width)
        energy = torch.bmm(proj_query, proj_key)

        attention = F.softmax(energy, dim=2)
        attention = nn.Dropout(p=0.5)(attention)
        proj_value = self.value_conv(x).view(batch_size, -1, width)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels,width)

        # Apply scaling factor
        out = self.gamma * out + x
        return out

class classifier(nn.Module):
    def __init__(self, args = None): # 4层encode构成一个bert
        super(classifier, self).__init__()
        self.args = args
        #### 开始定义下游任务 ####
        self.drop = nn.Dropout(0.5)
        self.GRU = nn.GRU(input_size=648, hidden_size=512, dropout=0.5,
                                 num_layers=2, batch_first=True, bidirectional=True)
        self.LSTM = nn.LSTM(input_size=648, hidden_size=512, dropout=0.5,
                                 num_layers=3, batch_first=True, bidirectional=True)
        self.CNN = nn.Sequential(
            nn.Conv1d(1,16,kernel_size=3,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(2),

        )

        self.att123 = SelfAttention(32)

        self.CNN2 = nn.Sequential(
            nn.Conv1d(32, 16, kernel_size=3, padding=2),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(3440, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = x.view(self.args['batch_size'], 1, -1)
        x = self.CNN(x)
        x = self.att123(x)
        y = self.CNN2(x)
        # x = self.drop(x)
        # x, _ = self.LSTM (x)
        # y = self.classifier(x)

        return y