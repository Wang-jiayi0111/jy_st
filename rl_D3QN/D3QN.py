import os
import torch
import random
import datetime
import numpy as np
from textwrap import fill
import matplotlib.pyplot as plt
from jy_exp.rl_D3QN.d3qn_brain import D3QN
from jy_exp.rl_common.class_op import ClassOp
import jy_exp.rl_D3QN.d3qn_train as d3qn_train
from jy_exp.rl_common.class_op import entropy_weight
from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_integration_env import ClassIntegrationEnv


# 加载数据
sys_name = "input_DNS"   # 系统名称 email_spl 、SPM
rl_name = "D3QN"
num_runs = 30
reward_v = "v2.1"             # v2---重要性、v2.1---GNN复杂度、v6---丁艳茹
if_EWM = False
                  # 是否使用熵权法
num_episodes = 3000
seed = 41

# Q-Learning训练参数 
LR = 8e-4                    # 学习率
EPSILON = 0.99               # 探索率
GAMMA = 0.93                 # 折扣因子0.95
BATCH_SIZE = 64              # 批量大小 64 128
MEMORY_CAPACITY = 20000      # 记忆池容量
TARGET_REPLACE_ITER = 100     # 目标网络更新频率  没用到


def run_d3qn(classes, methods, attributes, method_counts, attr_counts, num_episodes, device, seed, reward_v="v6", NOF=[], GNN_class=None):
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
    # agent = DQN(N_STATES, N_ACTIONS, LR, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, BATCH_SIZE, device)
    agent = D3QN(
        N_STATES,
        N_ACTIONS,
        LR,
        EPSILON,
        GAMMA,
        TARGET_REPLACE_ITER,
        MEMORY_CAPACITY,
        BATCH_SIZE,
        num_episodes,
        device
    )

    state = env.reset()[0]  # 环境重置
    # 进行训练
    return_list, best_OCplx, best_sequence = d3qn_train.D3QN_train(env, agent, num_episodes)
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
        # "Target Update": TARGET_REPLACE_ITER,
        "Memory Capacity": MEMORY_CAPACITY,
        "Batch Size": BATCH_SIZE,
        "if_EWM": if_EWM,
        "seed": seed,
    }

    if if_EWM:
        output_dir = os.path.join(output_dir, 'EWM')
    else:
        output_dir = os.path.join(output_dir, 'noEWM')
    run_dir = os.path.join(output_dir, rl_name)
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    png_dir = os.path.join(run_dir, 'png')
    os.makedirs(png_dir, exist_ok=True)
    # ClassOp.clear_folder(run_dir)
    # ClassOp.clear_folder(png_dir)
    dqn_results = []
    
    with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
        for key, value in params.items():
            f.write(f"\n{key}: {value}")  # 每行一个键值对
        f.write(f"\n")

    for run in range(num_runs):
        gobal_seed = seed + run
        # random.seed(gobal_seed)
        # np.random.seed(gobal_seed)
        # torch.manual_seed(gobal_seed)
        print(f"\n=== {sys_name}开始第 {run+1} 次训练 ===")   
        print(f"\nD3QN Run {run+1}/{num_runs}")
        return_list, best_ocplx, best_seq = run_d3qn(classes, methods, attributes, 
                                            method_counts, attr_counts, num_episodes, device,
                                            gobal_seed, reward_v, NOF=NOF_val, GNN_class=GNN_class)
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
        plt.title(f'D3QN on {sys_name}')
        plt.suptitle(f'Overall Complexity (OCplx): {best_ocplx}', fontsize=10, color='red')
        plt.savefig(os.path.join(png_dir, f'run_{run+1}-at-{current_time}.png'))
        # 保存最佳序列
        with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
            f.write(f"Run {run+1}:\n")
            f.write(f"Best OCplx: {best_ocplx}\n")
            f.write(f"Best Sequence: {best_seq}\n")
        plt.close()
