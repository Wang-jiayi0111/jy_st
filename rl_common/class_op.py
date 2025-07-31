import os
import shutil
import numpy as np
from collections import defaultdict

# 使用熵权法计算权重
def entropy_weight(*indicators, n):

    R = np.vstack(indicators).T
    P = R / (R.sum(axis=0) + 1e-12)
    
    K = 1 / np.log(n)
    # 处理log(0)的情况
    with np.errstate(divide='ignore', invalid='ignore'):
        log_P = np.log(P)
        log_P[~np.isfinite(log_P)] = 0  # 将-inf和NaN替换为0
    e = -K * (P * log_P).sum(axis=0)
    W = (1 - e) / (1 - e).sum()
    return W

def cal_scplx_matrix(methods, attributes, n, w_M=0.5, w_A=0.5):
    # 计算归一化所需的最小值和最大值
    min_method = np.min(methods)
    max_method = np.max(methods)
    max_attribute = np.max(attributes)
    min_attribute = np.min(attributes)
    scplx_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # 归一化方法耦合
            if max_method > min_method:
                coupling_methods = (methods[i][j]-min_method) / (max_method - min_method)
            else:
                coupling_methods = 0
            # 归一化属性耦合
            if max_attribute > min_attribute:
                coupling_attribute = (attributes[i][j]-min_attribute) / (max_attribute - min_attribute)
            else:
                coupling_attribute = 0
            # 计算耦合复杂度
            scplx_matrix[i][j] = np.sqrt(w_M * coupling_methods**2 + w_A * coupling_attribute**2)
    return scplx_matrix

class ClassOp:
    @staticmethod
    def load_classes(file_path):
        """加载类名文件，返回 {类索引: 类名} 的字典"""
        classes = {}
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                index = int(parts[0])
                class_name = parts[1]
                classes[index] = class_name
        return classes

    @staticmethod
    def load_data(file_path, num_classes):
        """加载依赖关系文件（方法/属性），返回 num_classes x num_classes 的依赖矩阵"""
        data = np.zeros((num_classes, num_classes), dtype=int)
        with open(file_path, 'r') as file:
            for line in file:
                parts = list(map(int, line.strip().split()))
                data[parts[0]-1, parts[1]-1] = parts[2]
        return data

    @staticmethod
    def load_GNNclass(sys_name):
        GNN_dir = "/home/ps/jy_exp/output/GNN_res6/Scores"
        file_path = os.path.join(GNN_dir, sys_name + "_scores.txt")
        print(f"加载GNN类评分文件: {file_path}")
        
        try:
            scores_dict = {}
            with open(file_path, 'r') as file:
                for line in file:
                    # 分割每行数据，去除可能的空白字符
                    parts = line.strip().split('--')
                    if len(parts) == 2:
                        class_index = int(parts[0])+1
                        score = float(parts[1])
                        scores_dict[class_index] = score
            return scores_dict
        except FileNotFoundError:
            print(f"文件 {file_path} 不存在")
            return None

    @staticmethod
    def cal_CBO(methods, attributes, n, w_M=0.5, w_A=0.5):
        """计算类的Coupling Between Objects (CBO)"""
        methods = methods
        attributes = attributes
        n = n
        w_M = w_M
        w_A = w_A
        scplx_matrix = cal_scplx_matrix(methods, attributes, n, w_M, w_A)
       
        # 计算CBO
        CBO = np.zeros(n)
        for i in range(n):
            for j in range(n):
                CBO[i] += scplx_matrix[i][j]
        return CBO

    @staticmethod
    def get_NOF(file_path):
        if file_path != '':
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            # 提取指定列的数据
            data = [int(line.split()[1]) for line in lines if line.strip()]
            
            return np.array(data)
        else:
            return []
    
    @staticmethod
    def count_features(method_matrix, attr_matrix):
        """统计每个类被其他类依赖的方法和属性总次数"""
        method_counts = np.sum(method_matrix, axis=0)  # 方法被依赖次数（列和）
        attr_counts = np.sum(attr_matrix, axis=0)      # 属性被依赖次数（列和）
        return method_counts, attr_counts

    @staticmethod
    def get_dependency_matrix(method_matrix, attr_matrix):
        """合并方法依赖和属性依赖，返回总依赖矩阵"""
        return method_matrix + attr_matrix
    
    @staticmethod
    def calculate_class_importance(index_to_class, NOF, CBO, dependency_matrix, max_iter=100, epsilon=1e-6):
        # 获取类列表和总数
        classes = sorted(index_to_class.keys())
        n = max(map(int, index_to_class.keys()))

        # 归一化NOF和CBO
        norm_NOF = (NOF - NOF.min()) / (NOF.max() - NOF.min() + 1e-9)
        norm_CBO = (CBO - CBO.min()) / (CBO.max() - CBO.min() + 1e-9)        
        
        W_nof, W_cbo = entropy_weight(norm_NOF, norm_CBO, n=n)
        print("W_nof:", W_nof)
        print("W_cbo:", W_cbo)
        
        # 初始复杂度C
        C = W_nof * NOF + W_cbo * CBO
        
        # 初始化IC和CC
        IC = C.copy()
        CC = C.copy()

        # region 校正
        # new_IC = np.zeros(n)
        # new_CC = np.zeros(n)
        # for i in range(n):
        #     for j in range(n):
        #         # 校正IC：Σ q_ji * CC_j (j→i)
        #         if dependency_matrix[j][i] > 0:
        #             q_ji = dependency_matrix[j][i] / (NOF[i] + 1e-9)
        #             new_IC[i] += q_ji * CC[j]
        #         # 校正CC：Σ q_ij * IC_j (i→j)
        #         if dependency_matrix[i][j] > 0:
        #             q_ij = dependency_matrix[i][j] / (NOF[j] + 1e-9)
        #             new_CC[i] += q_ij * IC[j]
        
        # # 归一化
        # IC = new_IC / (np.linalg.norm(new_IC) + 1e-9)
        # CC = new_CC / (np.linalg.norm(new_CC) + 1e-9)

        # endregion
        
        epsilon = 0.1 * n
        # HITS算法迭代
        for _ in range(max_iter):
            new_IC = [0.0 for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if dependency_matrix[i][j] > 0:  # i 依赖 j
                        new_IC[j] += CC[i]
            
            
            new_CC = [0.0 for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    if dependency_matrix[i][j] > 0:  # i 依赖 j
                        new_CC[i] += IC[j]

            # 归一化
            sum_IC = sum(new_IC)
            sum_CC = sum(new_CC)
            new_IC = [x / sum_IC for x in new_IC]
            new_CC = [x / sum_CC for x in new_CC]

            # 检查收敛
            error = sum(abs(new_IC[i] - IC[i]) for i in range(n)) + sum(abs(new_CC[i] - CC[i]) for i in range(n))

            if error <= epsilon:
                IC, CC = new_IC, new_CC
                break

            # region 原迭代收敛
            # new_IC = dependency_matrix @ CC    # IC_i = Σ (q_ij * CC_j)
            # new_CC = dependency_matrix.T @ IC  # CC_i = Σ (q_ji * IC_j)
            
            # # 归一化
            # new_CC = new_CC / (np.linalg.norm(new_CC) + 1e-9)
            # new_IC = new_IC / (np.linalg.norm(new_IC) + 1e-9)

            # # 检查收敛
            # if (np.linalg.norm(new_CC - CC) < epsilon and 
            #     np.linalg.norm(new_IC - IC) < epsilon):
            #     break

            # endregion
            
            CC, IC = new_CC, new_IC
        
        # 最终重要性值
        T = (np.array(IC) + np.array(CC)) / 2
        return {class_idx: T[i] for i, class_idx in enumerate(classes)}

    @staticmethod
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

        #     print(
        #         f"  ({cls}, {dep}) \nRaw Method Coupling: {raw_method_coupling}, Raw Attribute Coupling: {raw_attribute_coupling}\n "
        #         f"Normalized Method Coupling: {method_coupling:.4f}, Normalized Attribute Coupling: {attribute_coupling:.4f}\n "
        #         f"SCplx: {contribution:.4f}"
        #     )

        # print(f"\n总的OCplx: {OCplx:.4f}\n")
        return OCplx, method_couplings, attribute_couplings, class_pairs

    @staticmethod
    def clear_folder(folder_path):
        print(f"清空文件夹: {folder_path}")
        # 删除整个文件夹（包括文件夹本身）
        shutil.rmtree(folder_path)
        # 重新创建空文件夹
        os.makedirs(folder_path)