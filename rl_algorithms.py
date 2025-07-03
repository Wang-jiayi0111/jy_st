from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_op import ClassOp
import numpy as np

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

        print(
            f"  ({cls}, {dep}) \nRaw Method Coupling: {raw_method_coupling}, Raw Attribute Coupling: {raw_attribute_coupling}\n "
            f"Normalized Method Coupling: {method_coupling:.4f}, Normalized Attribute Coupling: {attribute_coupling:.4f}\n "
            f"SCplx: {contribution:.4f}"
        )

    # print(f"\n总的OCplx: {OCplx:.4f}\n")
    return OCplx, method_couplings, attribute_couplings, class_pairs

data = load_shared_data("elevator", "v6")
classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
    data["classes"], data["methods"], data["attributes"],
    data["method_counts"], data["attr_counts"], data["NOF"], 
    data["output_dir"], data["GNN_class"]
)
current_sequence =[12, 9, 6, 5, 11, 7, 2, 3, 8, 1, 10, 4]
w_M = 0.516
w_A = 0.484
# w_M = 0.5
# w_A = 0.5
OCplx, _, _, _ = calculate_OCplx_sequence(
                        attributes, methods, current_sequence, w_M=w_M, w_A=w_A
                    )
print("OCplx:", OCplx)