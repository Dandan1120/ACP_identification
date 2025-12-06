import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torch
from train_test import *
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
import matplotlib.cm as cm
#主函数在最下面，记得改data的地址!!!

import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader
import torch
import matplotlib.pyplot as plt

# 假设这些函数已经定义
# from train_test import *
# from data import *
# from con_attention import enc_classifier
# from feture_fusion import *
# from mlp import MLPClassifier
# from cnn import *
# from Transformer import TransformerModel
# from sklearn.preprocessing import StandardScaler
# from torch import nn
# import torch.optim as optim


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_validate(model_class, fusion_model_class, classifier_class, dataset, pc, n_splits=5, batch_size=64,
                   device='cuda'):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=50)  # 划分数据集
    vocab_size = 21  # 20种氨基酸 + 1个填充符号
    emb_dim = 64  # 词嵌入维度
    num_filters = 100  # 每个卷积层的卷积核数量
    filter_sizes = [3, 4, 5]  # 卷积核大小
    dropout_rate = 0.2  # Dropout 概率
    dropout_rate_cl = 0.1
    num_epochs = 5
    # 设置随机种子
    seed = 50
    set_seed(seed)

    all_fpr = []
    all_tpr = []
    all_roc_auc = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}/{n_splits}")
        accuracies = []  # 存储每个折叠的验证准确率
        losses = []
        # 划分训练集和验证集
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        # 创建 DataLoader
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class(
            vocab_size, emb_dim, num_filters, filter_sizes, dropout_rate
        )
        fusion_model = fusion_model_class(
            seq_feature_dim=300,
            pc_feature_dim=pc.size(1),
            hidden_dim=64
        )
        classifier = classifier_class(
            input_dim=64,  # 融合特征的维度
            hidden_dim=32,
            output_dim=2,  # 分类类别数
            dropout_rate=dropout_rate_cl
        )
        model.to(device)
        fusion_model.to(device)
        classifier.to(device)

        # 初始化优化器和损失函数
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(fusion_model.parameters()) + list(classifier.parameters()),
            lr=0.00001, weight_decay=1e-5
        )
        criterion = torch.nn.CrossEntropyLoss()

        # 训练模型
        for epoch in range(num_epochs):
            train_loss, train_acc, train_specificity, train_recall, train_mcc = train(
                model, fusion_model, classifier, train_loader, criterion, optimizer, device)
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Train Specificity: {train_specificity:.4f}, Train Recall: {train_recall:.4f}, Train MCC:{train_mcc:.4f}")

        # 验证模型
        test_loss, test_acc, test_spec, test_recall, test_mcc, fpr, tpr, roc_auc, y_true, y_pred = test(
        model, fusion_model, classifier, val_loader, criterion, device
    )

        accuracies.append(test_acc)
        losses.append(test_loss)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%, "
              f"Test Specificity: {test_spec:.4f}, Test Recall: {test_recall:.4f}, Test MCC:{test_mcc:.4f}")
        print("-" * 50)

        all_fpr.append(fpr)
        all_tpr.append(tpr)
        all_roc_auc.append(roc_auc)

    # 输出平均验证损失和准确率
    print(f"Mean Validation Loss: {np.mean(losses):.4f}")
    print(f"Mean Validation Accuracy: {np.mean(accuracies):.2f}%")

    # 莫兰迪色系颜色列表
    morandi_colors = [
     "#D87093",  # 玫瑰茜红，浓郁粉色，类似油画中常用的暖粉色调
    "#2F4F4F",  # 暗海蓝色，深沉的蓝，如同油画里表现海洋、深邃天空的颜色
    "#8A2BE2",  # 蓝紫色，经典的油画紫色，可用于表现神秘氛围
    "#FFA500",  # 橙黄色，油画中常用于表现阳光、温暖场景的色彩
    "#B22222"
]

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 8))
    for i in range(n_splits):
        plt.plot(all_fpr[i], all_tpr[i], lw=2, label=f'Fold {i + 1} (AUC = {all_roc_auc[i]:.2f})',
                 color=morandi_colors[i % len(morandi_colors)])

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Fold')
    plt.legend(loc="lower right")
    plt.show()


    return losses, accuracies
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
data = pd.read_csv('ACP2_alternate_protein_properties.csv')

#填充+截断氨基酸序列+将氨基酸序列转换为数字索引
max_length = 50 #允许容纳的氨基酸序列最大长度
sequences = data['Seq'].apply(sequence_to_ints).apply(lambda x: pad_sequence(x, max_length)).tolist()
labels = data['Label'].values

pc = data[["MolecularWeight", "IsoelectricPoint", "InstabilityIndex", "GRAVY", "ExtinctionCoefficient1", "ExtinctionCoefficient2", "AliphaticIndex"]]#理化特征

scaler = StandardScaler()
scaler.fit(pc)
pc_normalized = scaler.transform(pc)

# 张量
pc = torch.tensor(pc_normalized, dtype=torch.float32)

#转换为tensor
X = torch.tensor(sequences, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

#创建数据集
dataset = ProteinDataset(X,pc,y)

#加载数据集
batch_size=64
model_class = CNNEncoder
fusion_model_class = FeatureFusion_gate
classifier_class = MLPClassifier 

#交叉验证
losses, accuracies = cross_validate(model_class, fusion_model_class, classifier_class, dataset,pc, n_splits=5, batch_size=64, device=device)
