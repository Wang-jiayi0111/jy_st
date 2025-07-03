import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill
from jy_exp.rl_DQN.dqn_brain import DQN
import jy_exp.rl_DQN.dqn_train as dqn_train
from jy_exp.rl_common.class_op import ClassOp
from jy_exp.rl_common.class_op import entropy_weight
from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_integration_env import ClassIntegrationEnv


# 加载数据
sys_name = "input_BCEL"   # 系统名称 
rl_name = "DQN"
num_runs = 30
reward_v = "v2.1"             # v2---重要性、v2.1---GNN复杂度、v6---丁艳茹
if_EWM = True                  # 是否使用熵权法
num_episodes = 8000

# Q-Learning训练参数 
LR = 1e-5                    # 学习率8e-4        1e-3(daisy和elevator) 
EPSILON = 0.15               # 探索率
GAMMA = 0.99                 # 折扣因子0.95
TARGET_REPLACE_ITER = 50     # 目标网络更新频率  50（c=100
MEMORY_CAPACITY = 20000      # 记忆池容量
BATCH_SIZE = 128              # 批量大小 64 128

seed = 40

def plot_training_curve(return_list, best_ocplx, params, output_dir, num_episodes):
    plt.figure(figsize=(20, 10))
    
    # 绘制奖励曲线
    plt.plot(range(len(return_list)), return_list, linewidth=1.5)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    plt.title(f'DQN on {sys_name}')
    plt.suptitle(f'Overall Complexity (OCplx): {best_ocplx}', fontsize=10, color='red')

    param_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
    plt.annotate(fill(param_text, width=30),
                 xy=(0.98, 0.65), xycoords='figure fraction',
                 fontsize=10, ha='right', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    # 保存图片
    plt.savefig(os.path.join(output_dir, f'DQN{num_episodes}-at-{current_time}.png'))
    plt.close()

def run_dqn(classes, methods, attributes, method_counts, attr_counts, num_episodes, device, reward_v="v6", NOF=[], GNN_class=None):
    if if_EWM:
        sum_method = np.sum(methods, axis=1)
        sum_attr = np.sum(attributes, axis=1)
        norm_methods = (sum_method - methods.min()) / (methods.max() - methods.min() + 1e-9)
        norm_attributes = (sum_attr - attributes.min()) / (attributes.max() - attributes.min() + 1e-9)
        w_a, w_m = entropy_weight(norm_attributes, norm_methods, n=max(map(int, classes.keys())))
    else:
        w_a = 0.5
        w_m = 0.5

    # 创建环境
    env = ClassIntegrationEnv(classes=classes, methods=methods, attributes=attributes,
                            method_counts=method_counts, attr_counts=attr_counts,
                            wA=w_a, wM=w_m, 
                            version=reward_v, NOF=NOF, GNN_class=GNN_class, seed=seed)
    env.seed(seed)

    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]

    # 创建智能体
    agent = DQN(N_STATES, N_ACTIONS, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, device)

    state = env.reset()[0]  # 环境重置
    # 进行训练
    return_list, best_OCplx, best_sequence = dqn_train.DQN_train(env, agent, num_episodes)
    return return_list, best_OCplx, best_sequence

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = load_shared_data(sys_name, reward_v)
    classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
        data["classes"], data["methods"], data["attributes"],
        data["method_counts"], data["attr_counts"], data["NOF"], 
        data["output_dir"], data["GNN_class"]
    )
    window_size = 10
    current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")

    # 定义参数字典
    params = {
        "Learning Rate": LR,
        "Epsilon": EPSILON,
        "Gamma": GAMMA,
        "Target Update": TARGET_REPLACE_ITER,
        "Memory Capacity": MEMORY_CAPACITY,
        "Batch Size": BATCH_SIZE,
        "if_EWM": if_EWM
    }

    if if_EWM:
        output_dir = os.path.join(output_dir, 'EWM')
    else:
        output_dir = os.path.join(output_dir, 'noEWM')
    run_dir = os.path.join(output_dir, rl_name)
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    ClassOp.clear_folder(run_dir)
    dqn_results = []

    for run in range(num_runs):
        print(f"\n=== {sys_name}开始第 {run+1} 次训练 ===")   
        print(f"\nDQN Run {run+1}/{num_runs}")
        return_list, best_ocplx, best_seq = run_dqn(classes, methods, attributes, 
                                            method_counts, attr_counts, num_episodes, 
                                            device, reward_v, NOF=NOF_val, GNN_class=GNN_class)
        dqn_results.append(best_ocplx)

        moving_avg = []
        for i in range(0, len(return_list), window_size):
            window = return_list[i:i+window_size]
            if len(window) > 0:
                moving_avg.append(np.mean(window))

        # 创建x轴坐标（中点位置）
        x_positions = np.arange(window_size//2, len(return_list), window_size)

        # 确保长度匹配（处理边界情况）
        x_positions = x_positions[:len(moving_avg)]

        # 保存奖励曲线
        plt.figure(figsize=(20, 8))
        # plt.plot(range(len(return_list)), return_list)
        
        plt.plot(return_list, alpha=0.3, label='原始奖励')  # 原始数据半透明显示
        plt.plot(x_positions, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title(f'DQN on {sys_name} - Run {run+1}')
        plt.suptitle(f'Overall Complexity (OCplx): {best_ocplx}', fontsize=10, color='red')
        plt.savefig(os.path.join(run_dir, f'run_{run+1}-at-{current_time}.png'))
        # 保存最佳序列
        with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
            f.write(f"Run {run+1}:\n")
            f.write(f"Best OCplx: {best_ocplx}\n")
            f.write(f"Best Sequence: {best_seq}\n")
        plt.close()

    with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")  # 每行一个键值对
        f.write(f"\n")