# import torch

# def train(model, fusion_model, classifier, train_loader, criterion, optimizer, device):
#     model.train()
#     fusion_model.train()
#     classifier.train()

#     total_loss = 0
#     correct = 0
#     total = 0

#     for sequences, pc_features, labels in train_loader:
#         sequences, pc_features, labels = sequences.to(device), pc_features.to(device), labels.to(device)

#         # 生成序列位置索引
#         seq_idx = torch.arange(sequences.size(1), device=device).unsqueeze(0).expand(sequences.size(0), -1)
#         optimizer.zero_grad()
#         seq_features = model(sequences, seq_idx)
#         # seq_features = model(sequences)
#         fused_features = fusion_model(seq_features, pc_features)
#         outputs = classifier(fused_features)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
        
#         total_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += labels.size(0)
#         correct += predicted.eq(labels).sum().item()

#     avg_loss = total_loss / len(train_loader)
#     accuracy = 100.0 * correct / total
#     return avg_loss, accuracy

# def test(model, fusion_model, classifier, test_loader, criterion, device):
#     model.eval()
#     fusion_model.eval()
#     classifier.eval()

#     total_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for sequences, pc_features, labels in test_loader:
#             sequences, pc_features, labels = sequences.to(device), pc_features.to(device), labels.to(device)

#             # 生成序列位置索引
#             seq_idx = torch.arange(sequences.size(1), device=device).unsqueeze(0).expand(sequences.size(0), -1)
#             seq_features = model(sequences, seq_idx)
#             # seq_features = model(sequences)
#             fused_features = fusion_model(seq_features, pc_features)
#             outputs = classifier(fused_features)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += labels.size(0)
#             correct += predicted.eq(labels).sum().item()

#     avg_loss = total_loss / len(test_loader)
#     accuracy = 100.0 * correct / total
#     return avg_loss, accuracy
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_curve, auc



def train(model, fusion_model, classifier, train_loader, criterion, optimizer, device):
    model.train()
    fusion_model.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    # 初始化混淆矩阵的四个参数
    TP, FP, TN, FN = 0, 0, 0, 0

    # 用于存储真实标签和预测概率
    all_labels = []
    all_probs = []

    for sequences, pc_features, labels in train_loader:
        sequences, pc_features, labels = sequences.to(device), pc_features.to(device), labels.to(device)

        optimizer.zero_grad()
        seq_features = model(sequences)
        fused_features = fusion_model(seq_features, pc_features)
        outputs = classifier(fused_features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 计算混淆矩阵的四个参数
        y_true = labels.cpu().numpy()
        y_pred = predicted.cpu().numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        TP += tp
        FP += fp
        TN += tn
        FN += fn

        # 存储真实标签和预测概率
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
        all_labels.extend(y_true)
        all_probs.extend(probs)

    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    # 计算其他指标
    specificity= TN / (TN + FP) if (TN + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # 计算MCC分数
    mcc = matthews_corrcoef(all_labels, [1 if prob > 0.5 else 0 for prob in all_probs])


    return avg_loss, accuracy,specificity, recall, mcc
    

def test(model, fusion_model, classifier, test_loader, criterion, device):
    model.eval()
    fusion_model.eval()
    classifier.eval()

    total_loss = 0
    correct = 0
    total = 0

    # 初始化混淆矩阵的四个参数
    TP, FP, TN, FN = 0, 0, 0, 0
    # 用于存储真实标签和预测概率
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for sequences, pc_features, labels in test_loader:
            sequences, pc_features, labels = sequences.to(device), pc_features.to(device), labels.to(device)

            seq_features = model(sequences)
            fused_features = fusion_model(seq_features, pc_features)
            outputs = classifier(fused_features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算混淆矩阵的四个参数
            y_true = labels.cpu().numpy()
            y_pred = predicted.cpu().numpy()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
            TP += tp
            FP += fp
            TN += tn
            FN += fn

            # 存储真实标签和预测概率
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().detach().numpy()
            all_labels.extend(y_true)
            all_probs.extend(probs)
            all_preds.extend(y_pred)  # 保存类别标签

    # 计算平均损失和准确率
    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * correct / total

    # 计算其他指标
    specificity= TN / (TN + FP) if (TN + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
    # 计算MCC分数
    mcc = matthews_corrcoef(all_labels, [1 if prob > 0.5 else 0 for prob in all_probs])

    # 计算ROC曲线相关数据
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    return avg_loss, accuracy, specificity, recall,  mcc, fpr, tpr, roc_auc,all_labels,all_preds


import torch
import random
import numpy as np

def set_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    # 设置 CUDA 的随机种子（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    # 设置 Python 的随机种子
    random.seed(seed)
    # 设置 NumPy 的随机种子
    np.random.seed(seed)
