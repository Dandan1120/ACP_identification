from Bio.SeqUtils.ProtParam import ProteinAnalysis
import csv

#！！！看这里
#记得给输出文件改名，不然会覆盖之前的文件！
#！！！

# 读取 CSV 文件
input_file = "dataset\\pdo_main_set1.csv" # 替换为你的 CSV 文件路径
output_file = "pdo_main_set1_protein_properties.csv"

# 定义标准氨基酸字符
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

def clean_sequence(sequence):
    """清理序列，只保留标准氨基酸字符"""
    return "".join([aa for aa in sequence if aa in VALID_AMINO_ACIDS])

def calculate_aliphatic_index(sequence):
    """计算脂肪族指数"""
    aliphatic_aas = {'A': 1.0, 'V': 2.9, 'I': 3.9, 'L': 3.9}
    total = sum(aliphatic_aas.get(aa, 0) for aa in sequence)
    return (total / len(sequence)) * 100

# 打开输入和输出文件
with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    reader = csv.reader(infile)
    header = next(reader)  # 跳过表头
    
    # 表头
    outfile.write("Seq,Label,MolecularWeight,IsoelectricPoint,InstabilityIndex,GRAVY,ExtinctionCoefficient1,ExtinctionCoefficient2,AliphaticIndex\n")
    
    # 遍历
    for row in reader:
        protein_seq = row[0]  # 第一列是序列
        protein_label = row[1]  # 第二列是标签
        
        # 清理序列
        cleaned_seq = clean_sequence(protein_seq)
        
        # 检查清理后的序列是否为空
        if not cleaned_seq:
            # print(f"Warning: Sequence '{protein_seq}' is empty after cleaning. Skipping.")
            continue
        
        # 计算蛋白质理化性质
        try:
            protein = ProteinAnalysis(cleaned_seq)
            
            # 分子量
            molecular_weight = protein.molecular_weight()
            
            # 等电点
            isoelectric_point = protein.isoelectric_point()
            
            # 不稳定指数
            instability_index = protein.instability_index()
            
            # 平均疏水性（GRAVY）
            gravy = protein.gravy()
            
            # 消光系数（两种方法）
            extinction_coefficient1 = protein.molar_extinction_coefficient()[0]  # 含二硫键
            extinction_coefficient2 = protein.molar_extinction_coefficient()[1]  # 不含二硫键
            
            # 脂肪族指数（手动计算）
            aliphatic_index = calculate_aliphatic_index(cleaned_seq)
            
            # 预测二级结构
            # secondary_structure = predict_secondary_structure(cleaned_seq)
            
            # 写入结果
            outfile.write(f"{protein_seq},{protein_label},{molecular_weight:.2f},{isoelectric_point:.2f},{instability_index:.2f},{gravy:.2f},{extinction_coefficient1:.2f},{extinction_coefficient2:.2f},{aliphatic_index:.2f}\n")
        except Exception as e:
            print(f"Error processing sequence '{protein_seq}': {e}")

print(f"Results saved to {output_file}")