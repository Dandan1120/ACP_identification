from torch import nn
import torch
import torch.nn as nn
import torch.nn.init as init

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_rate)  

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x) 
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# class MLPClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
#         super(MLPClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 增加一个隐藏层
#         self.fc3 = nn.Linear(hidden_dim, output_dim)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout_rate)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)  # 批归一化层

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)  # 批归一化
#         x = self.dropout(x)  # Dropout 放在 ReLU 之前
#         x = self.relu(x)

#         x = self.fc2(x)
#         x = self.dropout(x)  # Dropout 放在 ReLU 之前
#         x = self.relu(x)

#         x = self.fc3(x)
#         x = self.dropout(x)  # Dropout 放在 ReLU 之前
#         x = self.sigmoid(x)
#         return x


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)

        if input_dim != hidden_dim:
            self.shortcut = nn.Linear(input_dim, hidden_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.dropout(x)
        x = self.relu(x)

        # x = x + identity
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        # x = self.sigmoid(x)
        return x
