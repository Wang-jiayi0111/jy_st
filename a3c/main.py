import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import queue
import random
import torch.multiprocessing as mp
from jy_exp.a3c.network import Net
from jy_exp.a3c.worker import Worker
from jy_exp.a3c.shared_adam import SharedAdam
from jy_exp.rl_common.class_op import entropy_weight
from jy_exp.rl_common.data_loader import load_shared_data
from jy_exp.rl_common.class_integration_env import ClassIntegrationEnv
from jy_exp.rl_common.class_op import ClassOp


sys_name = "notepad__spl"   # 系统名称
rl_name = "A3C"
reward_v = "v2.1"             # v2---重要性、v2.1---GNN复杂度、v6---丁艳茹
if_EWM = False              # 是否使用熵权法
num_episodes = 3000
num_runs = 30

# 训练参数
UPDATE_GLOBAL_ITER = 200    # 10
GAMMA = 0.99
MAX_EP = 3000
thread_count = 3    
RL=1e-3              # 1e-6


def set_global_seeds(seed):
    """设置全局随机种子（主进程）"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)  # 添加Python哈希种子

def run_a3c(classes, methods, attributes, 
            method_counts, attr_counts, num_episodes, 
            reward_v='v6', device='cpu', NOF=[], GNN_class=None, base_seed=42):
    if if_EWM:
        sum_method = np.sum(methods, axis=1)
        sum_attr = np.sum(attributes, axis=1)
        norm_methods = (sum_method - methods.min()) / (methods.max() - methods.min() + 1e-9)
        norm_attributes = (sum_attr - attributes.min()) / (attributes.max() - attributes.min() + 1e-9)
        w_a, w_m = entropy_weight(norm_attributes, norm_methods, n=max(map(int, classes.keys())))
    else:
        w_a = 0.5
        w_m = 0.5
    
    # 必须将多进程代码封装在函数内
    def train_process():
        
        # 初始化环境
        env = ClassIntegrationEnv(
            classes=classes,
            methods=methods,
            attributes=attributes,
            method_counts=method_counts,
            attr_counts=attr_counts,
            wA=w_a, wM=w_m, 
            version=reward_v, 
            NOF=NOF,
            GNN_class=GNN_class, 
            seed=base_seed
        )
        N_S = env.observation_space.shape[0]
        N_A = env.action_space.n
        
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        device = torch.device("cpu")  # 强制使用CPU，避免CUDA错误
        # 全局网络和优化器
        gnet = Net(N_S, N_A).to(device)
        gnet.share_memory().to(device)
        opt = SharedAdam(gnet.parameters(), lr=RL)

        # 共享变量
        global_ep = mp.Value('i', 0)
        global_ep_r = mp.Value('d', 0.)
        res_queue = mp.Queue()

        print("全局网络和共享变量已创建")
        # 创建工作进程
        workers = [
            Worker(
                env=env,
                gnet=gnet,
                opt=opt,
                global_ep=global_ep,
                global_ep_r=global_ep_r,
                res_queue=res_queue,
                name=i,
                N_S=N_S,
                N_A=N_A,
                MAX_EP=num_episodes,
                UPDATE_GLOBAL_ITER=UPDATE_GLOBAL_ITER,
                GAMMA=GAMMA,
                device=device,
                base_seed=base_seed
            ) 
            for i in range(thread_count)
        ]

        # 启动训练
        [w.start() for w in workers]
        print("工作进程已启动")

        # 收集结果
        return_list = []
        best_reward = -np.inf
        best_sequence = []

        # region 原始代码
        # # 计数器，记录已收到的结束信号数量
        # finished_workers = 0
        # start_time = time.time()
        # timeout = 300  # 设置超时时间为5分钟

        # while finished_workers < thread_count:  # 等待所有worker发送结束信号
        #     try:
        #         r = res_queue.get(timeout=1.0)
        #         if r is not None:
        #             if isinstance(r, tuple):  # 检查是否为最佳奖励和序列
        #                 current_reward, current_sequence, _ = r
        #                 print("r:", r)
        #                 if current_reward > best_reward and len(current_sequence) == len(set(current_sequence)):
        #                     best_reward = current_reward
        #                     best_sequence = current_sequence
        #             else:
        #                 return_list.append(r)  # 存储奖励
        #         else:
        #             finished_workers += 1  # 收到结束信号，增加计数
                    
        #         # 如果训练已经达到最大轮次且已经等待了足够长时间，强制结束等待
        #         current_time = time.time()
        #         if global_ep.value >= MAX_EP and (current_time - start_time) > timeout:
        #             print(f"已达到最大训练轮次 {MAX_EP}，且等待超时，强制结束等待。")
        #             break
                    
        #     except queue.Empty:
        #         # 如果队列获取超时，检查是否已经达到最大训练轮次
        #         if global_ep.value >= MAX_EP:
        #             # 检查是否等待时间过长
        #             current_time = time.time()
        #             if (current_time - start_time) > timeout:
        #                 print(f"等待工作进程结果超时，可能有进程卡住。已完成 {finished_workers}/{thread_count} 个工作进程。")
        #                 break

        # # 强制终止所有工作进程
        # for w in workers:
        #     if w.is_alive():
        #         w.terminate()
        #         print(f"强制终止工作进程 {w.name}")
        
        # # 等待所有进程结束，但设置较短的超时时间
        # for w in workers:
        #     w.join(timeout=5.0)
        # endregion

        finished_workers = 0
        while finished_workers < thread_count:
            try:
                r = res_queue.get(timeout=10) # 设置超时，避免死锁
                if r is None:
                    break
                if isinstance(r, tuple):
                    current_reward, current_sequence, _ = r
                    print("r:", r)
                    if current_reward > best_reward:
                        best_reward = current_reward
                        best_sequence = current_sequence
                else:
                    return_list.append(r)
            except queue.Empty:
                print("等待结果超时，可能工作进程出现问题")
                break

        for w in workers:
            if w.is_alive():
                w.terminate()
                print(f"强制终止工作进程 {w.name}")

        [w.join(timeout=5.0) for w in workers]

        # region 修改
        # while True:
        #     try:
        #         r = res_queue.get(timeout=10) # 设置超时，避免死锁
        #         if r is None:
        #             break
        #         if isinstance(r, tuple):
        #             current_reward, current_sequence, _ = r
        #             if current_reward > best_reward:
        #                 best_reward = current_reward
        #                 best_sequence = current_sequence
        #         else:
        #             return_list.append(r)
        #     except queue.Empty:
        #         print("等待结果超时，可能工作进程出现问题")
        #         break
            
        # print("best_sequence:", best_sequence)
        # print("best_reward:", best_reward)

        # [w.join() for w in workers]
        # endregion
        return return_list, best_sequence, best_reward

    # 执行训练
    return_list, best_sequence, best_reward = train_process()
    
    # 计算OCplx
    best_sequence = [num + 1 for num in best_sequence]
    OCplx = ClassOp.calculate_OCplx_sequence(attributes, methods, best_sequence)[0]

    return return_list, OCplx, best_sequence


# 独立运行时保护
if __name__ == "__main__":                             
    # 测试代码
    base_seed = 42
    set_global_seeds(base_seed)
    mp.set_start_method('spawn') 
    data = load_shared_data(sys_name, reward_v)
    classes, methods, attributes, method_counts, attr_counts, NOF_val, output_dir, GNN_class = (
        data["classes"], data["methods"], data["attributes"],
        data["method_counts"], data["attr_counts"], data["NOF"], 
        data["output_dir"], data["GNN_class"]
    )
    window_size = 10
    current_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
    

    if if_EWM:
        output_dir = os.path.join(output_dir, 'EWM')
    else:
        output_dir = os.path.join(output_dir, 'noEWM')

    run_dir = os.path.join(output_dir, rl_name)
    print(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    ClassOp.clear_folder(run_dir)
    a3c_results = []

    params = {
        UPDATE_GLOBAL_ITER: 'UPDATE_GLOBAL_ITER',
        'GAMMA': GAMMA,
        'MAX_EP': MAX_EP,
        'thread_count': thread_count,
        'RL': RL,
        'if_EWM': if_EWM,
    }

    for run in range(num_runs):
        run_seed = base_seed + run * 10
        print(f"\n=== 开始{sys_name}第 {run+1} 次训练 ===")   
        print(f"\nA3C Run {run+1}/{num_runs}")
        return_list, best_ocplx, best_seq = run_a3c(classes, methods, attributes, method_counts, attr_counts, num_episodes, 
                                                    reward_v, device='cpu', NOF=NOF_val, GNN_class=GNN_class,base_seed=run_seed)
        a3c_results.append(best_ocplx)

        moving_avg = []
        for i in range(0, len(return_list), window_size):
            window = return_list[i:i+window_size]
            if len(window) > 0:
                moving_avg.append(np.mean(window))

        # 创建x轴坐标（中点位置）
        x_positions = np.arange(window_size//2, len(return_list), window_size)

        # # 确保长度匹配（处理边界情况）
        # x_positions = x_positions[:len(moving_avg)]

        min_length = min(len(x_positions), len(moving_avg))
        x_positions = x_positions[:min_length]
        moving_avg = moving_avg[:min_length]

        # 保存奖励曲线
        plt.figure(figsize=(20, 8))
        # plt.plot(range(len(return_list)), return_list)
        plt.plot(return_list, alpha=0.3, label='原始奖励')  # 原始数据半透明显示
        plt.plot(x_positions, moving_avg, 'r-', linewidth=2, label=f'{window_size}轮移动平均')

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title(f'A3C on {sys_name} - Run {run+1}')
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
        f.write(f"\n")