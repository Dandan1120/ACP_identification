from torch import nn
import torch

class FeatureFusion_attention(nn.Module):
    def __init__(self, seq_feature_dim, pc_feature_dim, hidden_dim, num_heads):
        super(FeatureFusion_attention, self).__init__()
        self.seq_feature_dim = seq_feature_dim
        self.pc_feature_dim = pc_feature_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()
        # 线性变换
        self.seq_proj = nn.Linear(seq_feature_dim, hidden_dim)
        self.pc_proj = nn.Linear(pc_feature_dim, hidden_dim)

        # 使用 PyTorch 的 MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, seq_features, pc_features):
        # 线性变换
        seq_features = self.seq_proj(seq_features)  # (batch_size, hidden_dim)
        pc_features = self.pc_proj(pc_features)  # (batch_size, hidden_dim)
        pc_features = self.relu(pc_features)#激活

        # 拼接特征
        features = torch.stack([seq_features, pc_features], dim=1)  # (batch_size, 2, hidden_dim)
        features = features.transpose(0, 1)  # (2, batch_size, hidden_dim)

        # 注意力机制
        fused_features, _ = self.attention(features, features, features)  # (2, batch_size, hidden_dim)

        # 调整形状回 (batch_size, 2, hidden_dim)
        fused_features = fused_features.transpose(0, 1)  # (batch_size, 2, hidden_dim)

        # 取平均
        fused_features = fused_features.mean(dim=1)  # (batch_size, hidden_dim)

        return fused_features
    
class FeatureFusion_gate(nn.Module):
    def __init__(self, seq_feature_dim, pc_feature_dim, hidden_dim):
        super(FeatureFusion_gate, self).__init__()
        self.seq_feature_dim = seq_feature_dim
        self.pc_feature_dim = pc_feature_dim
        self.hidden_dim = hidden_dim
        self.relu = nn.ReLU()

        # 线性变换
        self.seq_proj = nn.Linear(seq_feature_dim, hidden_dim)
        self.pc_proj = nn.Linear(pc_feature_dim, hidden_dim)

        # 门控机制的线性层
        self.gate = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_features, pc_features):
        # 线性变换
        seq_features = self.seq_proj(seq_features)  # (batch_size, hidden_dim)
        pc_features = self.pc_proj(pc_features)  # (batch_size, hidden_dim)

        # 计算门控值
        combined_features = torch.cat([seq_features, pc_features], dim=1)
        gate_value = self.gate(combined_features)  # (batch_size, 1)

        # 门控融合
        gated_fused_features = gate_value * seq_features + (1 - gate_value) * pc_features

        return gated_fused_features



