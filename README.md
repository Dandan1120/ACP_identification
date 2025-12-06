# 基于自然语言处理技术的抗癌肽识别  
*A Deep Learning and NLP-based Framework for Anticancer Peptide (ACP) Prediction*  
> 项目信息来源：哈尔滨工业大学《大学生创新训练计划项目中期检查报告》:contentReference[oaicite:1]{index=1}

---

## 📌 项目简介
抗癌肽（Anticancer Peptides, ACPs）因具有 **高选择性、低毒性、良好渗透性** 等优点，被认为是新型肿瘤治疗方式。本项目旨在构建一个 **基于自然语言处理（NLP）与深度学习的多模态特征融合模型**，通过解析序列特征与理化性质，实现对 ACP 的高精度识别。

本研究提出了一种 **CNN + 门控机制（Gate）** 的创新架构，可动态融合序列与理化特征，并在多项指标上显著优于现有 SOTA 方法。

---

## 🚀 功能与特点

### 🔬 多模态特征提取
- **序列特征**：Embedding、CNN、多头注意力、Transformer 等编码结构  
- **理化特征**（7 类）：分子量、等电点、GRAVY、脂肪族指数、稳定性指数、电荷分布等  
- 使用 Biopython 自动化计算特征  

### 🔗 多模态融合方法
- **Gate 动态权重机制**：自适应调节序列/理化特征贡献度  
- **Multi-Attention 机制**：捕获跨特征交互关系  

### 🧠 分类器
- 三层全连接网络（MLP）  
- 支持 Sigmoid 二分类输出  
- 加入 Dropout 与残差连接提升稳健性  

### 📈 实验表现（ACP2alter 数据集）
| 模型 | 测试准确率 | AUC | MCC | 特点 |
|------|------------|-------|--------|-----------|
| **CNN + Gate** | **95.36%** | **0.98** | **0.90** | 收敛快，泛化能力强，特异度显著提升 |
| CNN + MultiAtt | 93.56% | 0.98 | 较优秀 | 注意力头竞争导致噪声 |
| Transformer + Gate | 93.56% | 0.97 | 一般 | 不适用于短序列任务 |
| Transformer + MultiAtt | 92.65% | 0.97 | 一般 | 容易过拟合 |

---

## 🧬 项目结构（建议）

```plaintext
project/
│── data/                 # ACP 数据集
│── src/                  # 核心模型代码
│   ├── dataset.py
│   ├── feature_physics.py
│   ├── embedding.py
│   ├── cnn_encoder.py
│   ├── transformer_encoder.py
│   ├── fusion_gate.py
│   ├── fusion_attention.py
│   ├── classifier.py
│   └── train.py
│── models/               # 已训练模型
│── results/              # 日志、曲线图、混淆矩阵等
│── README.md
└── requirements.txt
