from rl_common.class_op import ClassOp, cal_scplx_matrix
from jy_exp.rl_common.data_loader import load_shared_data

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
            
            CC, IC = new_CC, new_IC
        
        # 最终重要性值
        T = (np.array(IC) + np.array(CC)) / 2
        return {class_idx: T[i] for i, class_idx in enumerate(classes)}

sys_name = ["daisy", "elevator", "email__spl", "notepad__spl", "input_ANT", "input_ATM", "input_BCEL", "input_DNS", "input_DNS"]
data = load_shared_data("daisy", "v2")
classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
    data["classes"], data["methods"], data["attributes"],
    data["method_counts"], data["attr_counts"], data["NOF"], 
    data["output_dir"], data["GNN_class"]
)
num_classes = max(map(int, classes.keys()))
CBO = ClassOp.cal_CBO(methods, attributes, num_classes, w_M=0.5, w_A=0.5)
dependency_matrix = cal_scplx_matrix(methods, attributes, num_classes)

all_importance = ClassOp.calculate_class_importance(classes, NOF_val, CBO, dependency_matrix)
print("Class Importance Scores:")
print(" ".join(str(idx-1) for idx, _ in sorted(all_importance.items(), key=lambda x: x[1], reverse=True)))