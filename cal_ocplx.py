from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_op import ClassOp
import numpy as np

SYSTEM_WEIGHTS = {
    "ANT": (0.6335612556270102, 0.3664387443729899),
    "ATM": (0.43916402279659417, 0.5608359772034058),
    "BCEL": (0.40448490280540916, 0.5955150971945908),
    "DNS": (0.5539165588978445, 0.4460834411021554),
    "SPM": (0.5683864179034746, 0.4316135820965255),
    "daisy": (0.3546897896362407, 0.6453102103637594),
    "elevator": (0.516496350958985, 0.48350364904101495),
    "email__spl": (0.5032356176442202, 0.49676438235577974),
    "notepad__spl": (0.7967639817891569, 0.2032360182108432),
}

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

    print(f"\n总的OCplx: {OCplx:.4f}\n")
    return OCplx, method_couplings, attribute_couplings, class_pairs

sys_name = "daisy"
data = load_shared_data(sys_name, "v2")
classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
    data["classes"], data["methods"], data["attributes"],
    data["method_counts"], data["attr_counts"], data["NOF"], 
    data["output_dir"], data["GNN_class"]
)
current_sequence =  [21, 22, 1, 19, 13, 9, 8, 6, 16, 7, 11, 10, 5, 12, 4, 18, 17, 20, 23, 2, 3, 15, 14]

if_EWM = True

if if_EWM:  
    w_M, w_A = SYSTEM_WEIGHTS[sys_name]
else:
    w_M, w_A = 0.5, 0.5

OCplx, _, _, _ = calculate_OCplx_sequence(
                        attributes, methods, current_sequence, w_M=w_M, w_A=w_A
                    )
print("OCplx:", OCplx)