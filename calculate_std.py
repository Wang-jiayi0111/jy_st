import math
import os
import numpy as np
import re
from collections import defaultdict

# 配置参数 (根据实际需求修改这些值)
root_dir = '/home/ps/jy_exp/output'           # 根目录路径
target_version = 'v6'         # 目标版本号 (v2, v2.1, v6等)
target_category = 'EWM'     # 目标类别 (EWM 或 noEWM)
target_algorithm = 'A3C25_8e-5_3'      # 目标算法 (如A3C)

# 目标文件名模板
target_filename = f"best_sequence_{target_version}.txt"

# 存储找到的文件内容
all_contents = []

# 遍历所有数据集目录
for dataset in os.listdir(root_dir):
    dataset_path = os.path.join(root_dir, dataset)
    # 检查是否是目录
    if not os.path.isdir(dataset_path):
        continue
    
    # 构建目标文件路径
    file_path = os.path.join(
        dataset_path,
        target_version,
        target_category,
        target_algorithm,
        target_filename
    )
    # 检查文件是否存在并读取内容
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                content = f.read().strip()  # 读取内容并去除首尾空白
                if content:  # 只添加非空内容
                    # 添加数据集标识作为分隔符（可选）
                    all_contents.append(f"# === Dataset: {dataset} ===\n{content}\n")
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")

# 将所有内容写入input.txt
if all_contents:
    with open('input.txt', 'w') as output_file:
        # 写入文件头信息（可选）
        output_file.write(f"# Combined results for version: {target_version}, "
                         f"category: {target_category}, algorithm: {target_algorithm}\n\n")
        
        # 写入所有文件内容
        output_file.write("\n".join(all_contents))
    
    print(f"成功合并 {len(all_contents)} 个文件内容到 input1.txt")
else:
    print("未找到符合条件的文件")

def extract_best_ocplx(content):
    """
    从内容中提取所有 Best OCplx 值
    """
    best_ocplx_values = []
    
    # 使用正则表达式匹配所有 "Best OCplx:" 行
    pattern = r"Best OCplx:\s*([\d.]+)"
    
    matches = re.findall(pattern, content)
    if matches:
        try:
            # 将所有匹配值转换为浮点数
            best_ocplx_values = [float(match) for match in matches]
        except ValueError:
            pass
    
    return best_ocplx_values

def calculate_best_ocplx_statistics(input_file):
    """
    分析 input1.txt 文件，计算每个数据集的 Best OCplx 统计信息
    """
    # 存储每个数据集的所有 Best OCplx 值
    dataset_ocplx = defaultdict(list)
    
    # 当前数据集
    current_dataset = None
    
    # 读取文件内容
    with open(input_file, 'r') as f:
        content = f.read()
    
    # 按数据集分割内容
    datasets = re.split(r'# === Dataset: (.+?) ===\n', content)[1:]
    
    # 交替处理数据集名称和内容
    for i in range(0, len(datasets), 2):
        if i + 1 < len(datasets):
            dataset_name = datasets[i].strip()
            dataset_content = datasets[i + 1]
            
            # 提取该数据集的所有 Best OCplx 值
            best_ocplx_values = extract_best_ocplx(dataset_content)
            
            if best_ocplx_values:
                dataset_ocplx[dataset_name] = best_ocplx_values
    
    # 计算每个数据集的统计信息
    results = []
    for dataset, values in dataset_ocplx.items():
        # 确保我们处理的是数值列表
        if values and all(isinstance(v, float) for v in values):
            # 转换为 numpy 数组以便计算统计量
            arr = np.array(values)
            
            # 计算统计信息
            mean = np.mean(arr)
            std_dev = np.std(arr)
            min_val = np.min(arr)
            max_val = np.max(arr)
            count = len(values)
            
            results.append({
                'dataset': dataset,
                'mean': mean,
                'std_dev': std_dev,
                'min': min_val,
                'max': max_val,
                'count': count,
                'values': values
            })
        else:
            print(f"警告: 数据集 '{dataset}' 未找到有效的 Best OCplx 值")
    
    return results

def save_ocplx_statistics(results, output_file):
    """
    将 Best OCplx 统计结果保存到文件
    """
    with open(output_file, 'a') as f:
        # 写入标题
        f.write("Dataset\tMean Best OCplx\tStd Deviation\tMin\tMax\tCount\n")
        
        # 写入每个数据集的结果
        for result in results:
            f.write(f"{result['dataset']}\t{result['mean']:.2f}\t{result['std_dev']:.2f}\t"
                    f"{result['min']:.2f}\t{result['max']:.2f}\t{result['count']}\n")
        
        # 添加分隔线
        f.write("\n\nDetailed Best OCplx Values:\n")
        
        # 写入每个数据集的详细数值
        for result in results:
            f.write(f"\n{result['dataset']} values:\n")
            # 每行显示10个值
            for i in range(0, len(result['values']), 10):
                line_values = result['values'][i:i+10]
                f.write("\t".join(f"{v:.6f}" for v in line_values) + "\n")

input_file = "input.txt"
output_file = "dataset_best_ocplx_statistics.txt"

if not os.path.exists(input_file):
    print(f"错误: 输入文件 {input_file} 不存在")
    print("请先生成 input1.txt 文件")
else:
    results = calculate_best_ocplx_statistics(input_file)
    
    if results:
        save_ocplx_statistics(results, output_file)
        print(f"分析完成！结果已保存到 {output_file}")
        print(f"共处理 {len(results)} 个数据集")
        
        # 打印摘要
        print("\nBest OCplx 统计摘要:")
        print("Dataset\t\tMean\t\tStd Dev\t\tMin\t\tMax\t\tCount")
        for result in results:
            print(f"{result['dataset'][:15]}\t{result['mean']:.6f}\t{result['std_dev']:.6f}\t"
                    f"{result['min']:.6f}\t{result['max']:.6f}\t{result['count']}")
    else:
        print("未找到有效的 Best OCplx 数据")