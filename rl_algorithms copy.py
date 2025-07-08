from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_op import ClassOp
import numpy as np
import re

def calculate_OCplx_sequence(attributes, methods, sequence, w_M=0.5, w_A=0.5):
    """
    计算单一序列的Overall Complexity (OCplx)。
    sequence: 类的完整序列，例如 [1,2,3,4,5,6]
    attributes: 属性耦合矩阵
    methods: 方法耦合矩阵
    w_M: 方法耦合的权重
    w_A: 属性耦合的权重
    """

    OCplx = 0
    method_couplings = []
    attribute_couplings = []
    class_pairs = []
    # 创建一部字典，记录每个类在序列中的位置
    class_order = {cls: index for index, cls in enumerate(sequence)}

    # 收集所有依赖对 (cls, dep)，其中 cls 在序列中，dep 是 cls 的依赖且 dep 在 cls 之后
    all_dependencies = []
    for cls in sequence:
        for dep in range(1, len(sequence) + 1):
            # 假设类编号从1开始
            raw_method_coupling = methods[cls - 1, dep - 1]
            raw_attribute_coupling = attributes[cls - 1, dep - 1]
            if raw_method_coupling != 0 or raw_attribute_coupling != 0:
                # 仅包括 dep 在 cls 之后的依赖
                if dep in class_order and class_order[dep] > class_order[cls]:
                    all_dependencies.append((cls, dep, raw_method_coupling, raw_attribute_coupling))

    if not all_dependencies:
        print("序列中没有任何待计入的依赖关系。")
        return OCplx, method_couplings, attribute_couplings, class_pairs

    # 计算归一化所需的最小值和最大值
    min_method = np.min(methods)
    max_method = np.max(methods)
    max_attribute = np.max(attributes)
    min_attribute = np.min(attributes)
    
    # 遍历所有依赖对，计算归一化耦合值及其贡献
    for cls, dep, raw_method_coupling, raw_attribute_coupling in all_dependencies:
        # 归一化方法耦合
        if max_method > min_method:
            method_coupling = (raw_method_coupling) / (max_method - min_method)
        else:
            method_coupling = 0  # 避免除以零

        # 归一化属性耦合
        if max_attribute > min_attribute:
            attribute_coupling = (raw_attribute_coupling) / (max_attribute - min_attribute)
        else:
            attribute_coupling = 0  # 避免除以零

        # 存储归一化后的耦合值
        method_couplings.append(method_coupling)
        attribute_couplings.append(attribute_coupling)
        class_pairs.append((cls, dep))

        # 计算贡献 scplx
        contribution = np.sqrt(w_M * (method_coupling ** 2) + w_A * (attribute_coupling ** 2))
        OCplx += contribution

    return OCplx, method_couplings, attribute_couplings, class_pairs

def parse_sequences(file_path):
    """Parse sequences from the input file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    runs = []
    pattern = r'Run (\d+):\nBest OCplx: ([\d.]+)\nBest Sequence: (\[[\d, ]+\])'
    matches = re.findall(pattern, content)
    
    for match in matches:
        run_num = int(match[0])
        old_ocplx = float(match[1])
        sequence = eval(match[2])
        runs.append((run_num, old_ocplx, sequence))
    
    return runs

def generate_output_file(runs, output_path):
    """Generate output file with corrected OCplx values"""
    with open(output_path, 'w') as f:
        for run in runs:
            run_num, _, sequence, new_ocplx = run
            f.write(f"Run {run_num}:\n")
            f.write(f"Best OCplx: {new_ocplx}\n")
            f.write(f"Best Sequence: {sequence}\n")

def main():
    pro = "input_SPM"  # Specify the project name
    reward_v = "v2.1"
    data = load_shared_data(pro, reward_v)
    classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
        data["classes"], data["methods"], data["attributes"],
        data["method_counts"], data["attr_counts"], data["NOF"], 
        data["output_dir"], data["GNN_class"]
    )
    
    # Weight parameters
    # wM = 0.6335612556270102
    # wA = 0.3664387443729899         # ANT

    # wM = 0.43916402279659417
    # wA = 0.5608359772034058         # ATM

    # wM = 0.40448490280540916
    # wA = 0.5955150971945908         # BCEL

    # wM = 0.5539165588978445
    # wA = 0.4460834411021554          # DNS

    # wM = 0.5683864179034746
    # wA = 0.4316135820965255           #SPM

    # wM = 0.3546897896362407
    # wA = 0.6453102103637594       # daisy

    # wM = 0.516496350958985
    # wA = 0.48350364904101495    # elevator

    # wM = 0.5032356176442202
    # wA = 0.49676438235577974    # email__spl

    # wM = 0.7967639817891569
    # wA = 0.2032360182108432     # notepad__spl
    
    # Parse input sequences
    input_file = f"/home/ps/jy_exp/output/{pro}/{reward_v}/EWM/A3C/best_sequence_{reward_v}.txt"
    runs = parse_sequences(input_file)
    
    # Calculate corrected OCplx for each sequence
    corrected_runs = []
    for run_num, old_ocplx, sequence in runs:
        new_ocplx, _, _, _ = calculate_OCplx_sequence(
            attributes, methods, sequence, w_M=wM, w_A=wA
        )
        corrected_runs.append((run_num, old_ocplx, sequence, new_ocplx))
    
    # Generate output file
    output_file = f"/home/ps/jy_exp/output/{pro}/{reward_v}/EWM/A3C/best_sequence_{reward_v}_.txt"
    generate_output_file(corrected_runs, output_file)
    print(f"Corrected sequences written to {output_file}")

if __name__ == "__main__":
    main()