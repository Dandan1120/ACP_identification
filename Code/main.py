from data import *
from con_attention import enc_classifier
from feture_fusion import *
from mlp import MLPClassifier
from train_test import *
from torch import nn
import torch.optim as optim
from Transformer import TransformerModel
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from cnn import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import seaborn as sns
#一样，换数据集了记得把data地址也改掉！！！
data = pd.read_csv('ACP2_alternate_protein_properties.csv')

#建议用gpu，不然会很慢
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")

#划分训练集测试集，比例8：2
train_data, test_data = train_test_split(data, test_size=0.2, random_state=54)

#填充+截断氨基酸序列+将氨基酸序列转换为数字索引
max_length = 50 #允许容纳的氨基酸序列最大长度
train_padded_sequences = train_data['Seq'].apply(sequence_to_ints).apply(lambda x: pad_sequence(x, max_length)).tolist()
train_labels = train_data['Label'].values
test_padded_sequences = test_data['Seq'].apply(sequence_to_ints).apply(lambda x: pad_sequence(x, max_length)).tolist()
test_labels = test_data['Label'].values

# train_pc = train_data[["MolecularWeight", "IsoelectricPoint", "InstabilityIndex", "GRAVY", "ExtinctionCoefficient1", "ExtinctionCoefficient2", "AliphaticIndex"]]
# test_pc = test_data[["MolecularWeight", "IsoelectricPoint", "InstabilityIndex", "GRAVY", "ExtinctionCoefficient1", "ExtinctionCoefficient2", "AliphaticIndex"]]
train_pc = train_data[["MolecularWeight", "IsoelectricPoint","InstabilityIndex","GRAVY", "ExtinctionCoefficient1", "ExtinctionCoefficient2", "AliphaticIndex"]]
test_pc = test_data[["MolecularWeight", "IsoelectricPoint", "InstabilityIndex","GRAVY", "ExtinctionCoefficient1", "ExtinctionCoefficient2", "AliphaticIndex"]]
scaler = StandardScaler()
# scaler = MinMaxScaler()
# train_pc_normalized = scaler.fit_transform(train_pc)
# test_pc_normalized = scaler.transform(test_pc)
from sklearn.preprocessing import RobustScaler
robust_scaler = RobustScaler()
train_pc_normalized  = robust_scaler.fit_transform(train_pc)
test_pc_normalized = robust_scaler.transform(test_pc)


# 张量
train_pc = torch.tensor(train_pc_normalized, dtype=torch.float32)
test_pc = torch.tensor(test_pc_normalized, dtype=torch.float32)

#转换为tensor
X_train = torch.tensor(train_padded_sequences, dtype=torch.long)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test= torch.tensor(test_padded_sequences, dtype=torch.long)
y_test = torch.tensor(test_labels, dtype=torch.long)


#创建数据集
train_dataset = ProteinDataset(X_train, train_pc,y_train)
test_dataset = ProteinDataset(X_test,train_pc, y_test)

#加载数据集

batch_size=64
train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 模型部分
# 定义模型参数
# vocab_size = len(char_to_int)  # 氨基酸词汇表大小
# emb_dim = 64 # 嵌入维度
# n_encoder = 6  # 编码器层数
# h = 8  # 多头注意力头数
# d_ff = 128  # 前馈网络维度
# dropout_rate_enc = 0.2 # Dropout 概率
dropout_rate_cl = 0.1

# 初始化模型
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
pretrained_model_name = "facebook/esm2_t6_8M_UR50D"

# 设置随机种子
seed = 50
set_seed(seed)
# model = enc_classifier(
#     n_encoder=n_encoder,
#     h=h,
#     emb_dim=emb_dim,



#     d_ff=d_ff,
#     dropout_rate=dropout_rate_enc,
#     vocab_size=vocab_size+1
#     # pretrained_model_name=pretrained_model_name
# )
# channels = [32,64]
num_layer = 2
vocab_size = 21  # 20种氨基酸 + 1个填充符号
emb_dim = 64    # 词嵌入维度
num_filters = 100  # 每个卷积层的卷积核数量
filter_sizes = [3,4,5]  # 卷积核大小
dropout_rate = 0.2  # Dropout 概率

# # 初始化 CNN 编码器
model = CNNEncoder(vocab_size, emb_dim, num_filters, filter_sizes, dropout_rate)

# 初始化特征融合模块
# fusion_model = FeatureFusion_attention(
#     seq_feature_dim=emb_dim,
#     pc_feature_dim=train_pc.size(1),
#     hidden_dim=64,
#     num_heads=4
# )
fusion_model = FeatureFusion_gate(
    seq_feature_dim=300,
    pc_feature_dim=train_pc.size(1),
    hidden_dim=64
)

# 初始化分类器
classifier = MLPClassifier(
    input_dim=64,  # 融合特征的维度
    hidden_dim=32,
    output_dim=2,  # 分类两类
    dropout_rate = dropout_rate_cl
)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
# model.apply(init_weights)
# fusion_model.apply(init_weights)
# classifier.apply(init_weights)
# 将模型移动到设备
model.to(device)
fusion_model.to(device)
classifier.to(device)

# 训练+测试
lr=0.00001
num_epochs = 10
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# criterion =nn.BCEWithLogitsLoss()
optimizer = optim.Adam(
    list(model.parameters()) + list(fusion_model.parameters()) + list(classifier.parameters()),
    lr=lr,weight_decay=1e-5)


train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

last_fpr, last_tpr, last_roc_auc = None, None, None

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc, train_specificity, train_recall, train_mcc = train(
        model, fusion_model, classifier, train_loader, criterion, optimizer, device
    )

    # 测试
    test_loss, test_acc, test_spec, test_recall, test_mcc, fpr, tpr, roc_auc, y_true, y_pred = test(
        model, fusion_model, classifier, test_loader, criterion, device
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # 存储最后一个 epoch 的 ROC 曲线数据
    if epoch == num_epochs - 1:
        last_fpr, last_tpr, last_roc_auc = fpr, tpr, roc_auc

    # 打印训练和测试结果
    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Train Specificity: {train_specificity:.4f}, Train Recall: {train_recall:.4f}, Train MCC:{train_mcc:.4f}")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
          f"Test Specificity: {test_spec:.4f}, Test Recall: {test_recall:.4f}, Test MCC:{test_mcc:.4f}")
    print("-" * 50)
# 运行测试函数并获取数据
(test_loss, test_acc, test_spec, test_recall, 
 test_mcc, fpr, tpr, roc_auc, y_true, y_pred) = test(model, fusion_model, classifier, test_loader, criterion, device)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# 绘制损失和准确率曲线
plt.figure(figsize=(18, 5))
# 绘制损失曲线
plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

# 绘制准确率曲线
plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(last_fpr, last_tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % last_roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.suptitle('CNN+Gate In ACP2alter')
plt.show()






