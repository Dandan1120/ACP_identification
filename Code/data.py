import torch 
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# 将氨基酸序列转换为数值特征
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
char_to_int = {char: i+1 for i, char in enumerate(amino_acids)}#i是索引 char是氨基酸

def sequence_to_ints(sequence):
    return [char_to_int[char] for char in sequence if char in char_to_int]

# 填充+截断序列
def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        #填充序列
        padded_sequence = sequence + [0] * (max_length - len(sequence))
    else:
        # 截断序列只取前五十
        padded_sequence = sequence[:max_length]
    return padded_sequence

class ProteinDataset(Dataset):
    def __init__(self, sequences, csv_features, labels):
        self.sequences = sequences
        self.csv_features = csv_features
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.csv_features[idx], self.labels[idx]

# data = pd.read_csv('ACP2_alternate_protein_properties.csv')

# #建议用gpu，不然会很慢
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
# print(f"Using device: {device}")

# #划分训练集测试集，比例8：2
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=50)

# #填充+截断氨基酸序列+将氨基酸序列转换为数字索引
# max_length = 50 #允许容纳的氨基酸序列最大长度
# train_padded_sequences = train_data['Seq'].apply(sequence_to_ints).apply(lambda x: pad_sequence(x, max_length)).tolist()
# train_labels = train_data['Label'].values

# # 打印处理后的数据
# print("\n处理后的训练集序列（前5条）：")
# for i, seq in enumerate(train_padded_sequences[:5]):
#     print(f"序列 {i+1}: {seq}")

# print("\n处理后的训练集标签（前5条）：")
# print(train_labels[:5])

# # 打印原始数据和处理后的数据对比
# print("\n原始数据（前5条）：")
# print(train_data[['Seq', 'Label']].head())
