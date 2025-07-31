import os
import json
import re
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def extract_best_ocplx(file_path):
    """
    从文件中提取所有 Best OCplx 值
    """
    best_ocplx_values = []
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
            # 使用正则表达式匹配所有 "Best OCplx:" 行
            pattern = r"Best OCplx:\s*([\d.]+)"
            matches = re.findall(pattern, content)
            
            if matches:
                # 将所有匹配值转换为浮点数
                best_ocplx_values = [float(match) for match in matches]
                
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
    
    return best_ocplx_values

def get_algorithm_ocplx_pairs(root_dir, dataset, version, category):
    """
    获取指定数据集、版本、类别下所有算法的 Best OCplx 值
    并返回两两算法之间的比较数组
    """
    # 存储每个算法的 Best OCplx 值
    algorithm_ocplx = defaultdict(list)
    
    # 构建目标目录路径
    target_dir = os.path.join(root_dir, dataset, version, category)
    
    # 检查目录是否存在
    if not os.path.exists(target_dir):
        print(f"错误: 目录不存在 - {target_dir}")
        return None
    
    # 遍历所有算法目录
    for algorithm in os.listdir(target_dir):
        algorithm_path = os.path.join(target_dir, algorithm)
        
        # 检查是否是目录
        if not os.path.isdir(algorithm_path):
            continue
        
        # 构建目标文件名
        target_file = f"best_sequence_{version}.txt"
        file_path = os.path.join(algorithm_path, target_file)
        
        # 检查文件是否存在并提取 OCplx 值
        if os.path.isfile(file_path):
            ocplx_values = extract_best_ocplx(file_path)
            if ocplx_values:
                algorithm_ocplx[algorithm] = ocplx_values
    
    # 创建两两算法比较的数组
    algorithm_pairs = {}
    algorithms = list(algorithm_ocplx.keys())
    
    # 生成所有可能的算法对
    for i in range(len(algorithms)):
        for j in range(i + 1, len(algorithms)):
            alg1 = algorithms[i]
            alg2 = algorithms[j]
            
            # 获取两个算法的 OCplx 值
            values1 = algorithm_ocplx.get(alg1, [])
            values2 = algorithm_ocplx.get(alg2, [])
            
            # 确保两个列表长度相同（运行次数相同）
            min_len = min(len(values1), len(values2))
            if min_len == 0:
                continue
                
            # 截取相同长度的部分
            values1 = values1[:min_len]
            values2 = values2[:min_len]
            
            # 创建比较数组
            pair_name = f"{alg1}_vs_{alg2}"
            algorithm_pairs[pair_name] = {
                alg1: values1,
                alg2: values2
            }
    
    return algorithm_pairs

def save_algorithm_comparisons(comparisons, output_file):
    """
    保存算法比较结果到 JSON 文件
    """
    with open(output_file, 'w') as f:
        json.dump(comparisons, f, indent=4)

def calculate_p_value(group1, group2):
    """
    计算两组数据的Mann-Whitney U检验p值
    """
    try:
        # 确保数组长度相同
        min_len = min(len(group1), len(group2))
        if min_len < 3:
            return 1.0  # 样本太小，返回无效值
        
        group1 = np.array(group1[:min_len])
        group2 = np.array(group2[:min_len])
        
        # 执行Wilcoxon秩和检验
        statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        return p_value
    except Exception as e:
        print(f"计算p值时出错: {str(e)}")
        return 1.0

def create_p_value_matrix(comparison_data):
    """
    创建p值下三角矩阵
    """
    # 获取所有唯一算法名称
    all_algorithms = set()
    for pair_name in comparison_data:
        algs = pair_name.split('_vs_')
        all_algorithms.update(algs)
    
    # 按字母顺序排序算法
    sorted_algorithms = sorted(all_algorithms)
    num_algorithms = len(sorted_algorithms)
    
    # 初始化矩阵
    p_matrix = np.full((num_algorithms, num_algorithms), np.nan)
    algorithm_index = {alg: idx for idx, alg in enumerate(sorted_algorithms)}
    
    # 填充矩阵的对角线（自身比较设为1.0）
    np.fill_diagonal(p_matrix, 1.0)
    
    # 填充两两比较的p值
    for pair_name, pair_data in comparison_data.items():
        algs = pair_name.split('_vs_')
        if len(algs) != 2:
            continue
            
        alg1, alg2 = algs
        group1 = pair_data.get(alg1, [])
        group2 = pair_data.get(alg2, [])
        
        if not group1 or not group2:
            continue
            
        # 计算p值
        p_val = calculate_p_value(group1, group2)
        
        # 获取矩阵索引（确保只填充下三角）
        i = algorithm_index[alg1]
        j = algorithm_index[alg2]
        
        # 确保i > j（下三角位置）
        if i < j:
            i, j = j, i
        
        # 只填充下三角（包括对角线）
        if i >= j:
            p_matrix[i, j] = p_val
    
    return sorted_algorithms, p_matrix

def save_p_value_table(algorithms, p_matrix, output_file):
    """
    将p值矩阵保存为下三角表格
    """
    num_algorithms = len(algorithms)
    
    with open(output_file, 'a') as f:
        # 写入表头
        f.write("Algorithm\t" + "\t".join(algorithms) + "\n")
        
        # 写入每行数据
        for i in range(num_algorithms):
            line = [algorithms[i]]
            
            # 只写入下三角部分（包括对角线）
            for j in range(num_algorithms):
                if i >= j:  # 下三角位置
                    value = p_matrix[i, j]
                    # 格式化输出
                    if i == j:  # 对角线（自身比较）
                        line.append("-")
                    elif np.isnan(value):  # 缺失值
                        line.append("N/A")
                    else:  # p值
                        # 科学计数法显示小值
                        if value < 0.001:
                            line.append(f"{value:.2e}")
                        else:
                            line.append(f"{value:.6f}")
                else:  # 上三角位置留空
                    line.append("")
            
            f.write("\t".join(line) + "\n")
        
        # 添加显著性说明
        f.write("\nSignificance levels:\n")
        f.write("* p < 0.05\n")
        f.write("** p < 0.01\n")
        f.write("*** p < 0.001\n")

# 主程序
if __name__ == "__main__":
    # 配置参数 (根据实际需求修改这些值)
    root_dir = 'output'           # 根目录路径
    dataset = 'notepad__spl'        # 目标数据集
    version = 'v2'                # 目标版本号
    category = 'EWM1'              # 目标类别
    output_dir = 'stat_results'   # 输出目录
    
    # 获取算法比较数据
    comparisons = get_algorithm_ocplx_pairs(root_dir, dataset, version, category)
    output_file = f"{dataset}_{version}_{category}_algorithm_comparisons.json"
    
    if comparisons:
        # 保存到 JSON 文件
        save_algorithm_comparisons(comparisons, output_file)
        print(f"成功生成算法比较数据，已保存到 {output_file}")
        
        # 打印摘要信息
        print(f"\n数据集: {dataset}, 版本: {version}, 类别: {category}")
        print(f"找到 {len(comparisons)} 对算法比较:")
        
        for pair_name, pair_data in comparisons.items():
            alg1, alg2 = pair_name.split('_vs_')
            runs = len(pair_data[alg1])
            print(f"- {pair_name}: {runs} 次运行")
        
        # 打印示例数据
        sample_pair = next(iter(comparisons.items()))
        print(f"\n示例比较 ({sample_pair[0]}):")
        print(f"  {list(sample_pair[1].keys())[0]}: {sample_pair[1][list(sample_pair[1].keys())[0]][:5]}...")
        print(f"  {list(sample_pair[1].keys())[1]}: {sample_pair[1][list(sample_pair[1].keys())[1]][:5]}...")
    else:
        print("未找到有效的算法比较数据")
    # 算法比较数据文件
    output_txt = "p_value_matrix.txt"
    
    try:
        # 读取比较数据
        with open(output_file, 'r') as f:
            comparison_data = json.load(f)
        
        # 创建p值矩阵
        algorithms, p_matrix = create_p_value_matrix(comparison_data)
        
        # 保存为下三角表格
        save_p_value_table(algorithms, p_matrix, output_txt)
        print(f"p值矩阵已保存到 {output_txt}")
        
        # 打印摘要
        print("\n算法列表:", algorithms)
        print("p值矩阵摘要:")
        print("行\\列", end="\t")
        for alg in algorithms:
            print(alg[:5], end="\t")
        print()
        
        for i, alg_row in enumerate(algorithms):
            print(alg_row[:5], end="\t")
            for j in range(len(algorithms)):
                if i >= j:
                    value = p_matrix[i, j]
                    if i == j:
                        print("-", end="\t")
                    elif np.isnan(value):
                        print("N/A", end="\t")
                    else:
                        print(f"{value:.4f}", end="\t")
                else:
                    print("", end="\t")
            print()
            
    except FileNotFoundError:
        print(f"错误: 输入文件 {output_txt} 不存在")
    except json.JSONDecodeError:
        print(f"错误: 文件 {output_txt} 格式无效")
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")