import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from textwrap import fill
from jy_exp.rl_common.class_op import ClassOp
from jy_exp.rl_common.class_integration_env import ClassIntegrationEnv
from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_op import entropy_weight
from jy_exp.rl_sarsa.sarsa_brain import SARSA
import jy_exp.rl_sarsa.sarsa_train as sarsa_train

# 加载数据
sys_name = "notepad__spl"   # 系统名称
rl_name = "SARSA"
reward_v = "v2"             # v2---重要性、v2.1---GNN复杂度、v6---丁艳茹
if_EWM = False              # 是否使用熵权法
num_episodes = 3000
num_runs = 30

# SARSA训练参数 
LR = 1e-3                   # 学习率  1e-5
EPSILON = 0.1               # 探索率
GAMMA = 0.99                # 折扣因子    0.95
seed = 40

def plot_training_curve(return_list, best_ocplx, params, output_dir, num_episodes):
    plt.figure(figsize=(20, 10))
    
    # 绘制奖励曲线
    plt.plot(range(len(return_list)), return_list, linewidth=1.5)
    plt.xlabel('Episodes', fontsize=12)
    plt.ylabel('Rewards', fontsize=12)
    plt.title(f'SARSA on {sys_name}')
    plt.suptitle(f'Overall Complexity (OCplx): {best_ocplx}', fontsize=10, color='red')

    param_text = "\n".join([f"{k}: {v}\n" for k, v in params.items()])
    plt.annotate(fill(param_text, width=30),
                 xy=(0.98, 0.65), xycoords='figure fraction',
                 fontsize=10, ha='right', va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    plt.savefig(os.path.join(output_dir, f'SARSA{num_episodes}-at-{current_time}.png'))
    plt.close()

def run_sarsa(classes, methods, attributes, method_counts, attr_counts, num_episodes, device, reward_v="v6", NOF=[], GNN_class=None):
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
                            version=reward_v, NOF=NOF, GNN_class=GNN_class,seed=seed)
    env.seed(seed)

    N_ACTIONS = env.action_space.n
    N_STATES = env.observation_space.shape[0]

    # 创建智能体
    agent = SARSA(N_STATES, N_ACTIONS, LR, EPSILON, GAMMA, device)

    # 进行训练
    best_OCplx, best_sequence, return_list = sarsa_train.SARSA_train(env, agent, num_episodes)
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

    params = {
        "Learning Rate": LR,
        "Epsilon": EPSILON,
        "Gamma": GAMMA,
        "Reward Version": reward_v
    }

    # if if_EWM:
    #     output_dir = os.path.join(output_dir, 'EWM')
    # else:
    #     output_dir = os.path.join(output_dir, 'noEWM')
    run_dir = os.path.join(output_dir, rl_name)
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    ClassOp.clear_folder(run_dir)
    sarsa_results = []

    for run in range(num_runs):
        print(f"\n=== 开始{sys_name}第 {run+1} 次训练 ===")   
        print(f"\nSARSA Run {run+1}/{num_runs}")
        return_list, best_ocplx, best_seq = run_sarsa(classes, methods, attributes, 
                                                    method_counts, attr_counts, num_episodes, 
                                                    device, reward_v, NOF=NOF_val, GNN_class=GNN_class)
        sarsa_results.append(best_ocplx)

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
        plt.title(f'SARSA on {sys_name} - Run {run+1}')
        plt.suptitle(f'Overall Complexity (OCplx): {best_ocplx}', fontsize=10, color='red')
        plt.savefig(os.path.join(run_dir, f'rewards_run_{run+1}-at-{current_time}.png'))
        # 保存最佳序列
        with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
            f.write(f"Run {run+1}:\n")
            f.write(f"Best OCplx: {best_ocplx}\n")
            f.write(f"Best Sequence: {best_seq}\n")
        plt.close()

    with open(os.path.join(run_dir, f'best_sequence_{reward_v}.txt'), 'a') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")  # 每行一个键值对