import torch
import torch.nn as nn
import numpy as np
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_filters, filter_sizes, dropout_rate):
        super(CNNEncoder, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, emb_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, emb_dim, seq_len]
        
        # 卷积 + 池化
        conved = [torch.relu(conv(embedded)) for conv in self.convs]  # [batch_size, num_filters, seq_len - fs + 1]
        pooled = [torch.max(conv, dim=2)[0] for conv in conved]  # [batch_size, num_filters]
        # pooled = [torch.max(conv,kernel_size=conv.size(2))[0] for conv in conved]
        # 拼接所有卷积层的输出
        cat = self.dropout(torch.cat(pooled, dim=1))  # [batch_size, num_filters * len(filter_sizes)]
        
        return cat


# vocab_size = 21  # 20种氨基酸 + 1个填充符号
# emb_dim = 128    # 词嵌入维度（可调）
# num_filters = 100  # 每个卷积层的卷积核数量
# filter_sizes = [4,6,8]  # 卷积核大小（可调）
# dropout_rate = 0.40  # Dropout 概率（可调）

# # 初始化 CNN 编码器
# model = CNNEncoder(vocab_size, emb_dim, num_filters, filter_sizes, dropout_rate)




