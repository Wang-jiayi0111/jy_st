# worker.py (新文件)
import torch
import random
import numpy as np
import torch.multiprocessing as mp
from .network import Net  # 从本地模块导入
from .utils import v_wrap, record, push_and_pull 
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
writer = SummaryWriter(log_dir='./logs/a3c')

class Worker(mp.Process):
    def __init__(self, env, gnet, opt, global_ep, global_ep_r, res_queue, name, 
                 N_S, N_A, MAX_EP, UPDATE_GLOBAL_ITER, GAMMA, device, base_seed, **kwargs):
        super(Worker, self).__init__()
        self.name = f'w{name:02d}'  # 更现代的字符串格式化
        self.g_ep = global_ep       # 共享全局训练轮数
        self.g_ep_r = global_ep_r
        self.res_queue = res_queue  # 结果队列（用于进程间通信）
        self.gnet = gnet            # 全局网络
        self.opt = opt              # 全局优化器
        self.lnet = Net(N_S, N_A).to(device)   # 本地网络
        self.env = env              # 使用传入的环境
        self.local_best_reward = -np.inf
        self.local_best_sequence = []
        self.best_reward_episode = None
        self.N_S = N_S
        self.N_A = N_A
        self.episode_num = MAX_EP
        self.global_iter = UPDATE_GLOBAL_ITER
        self.gamma = GAMMA
        self.device = device
        # self.schedule = scheduler = ReduceLROnPlateau(opt, mode='min', patience=10, factor=0.5)
        self.base_seed = base_seed
        self.worker_seed = base_seed + name * 1000  # *1

    def run(self):
        try:
            random.seed(self.worker_seed)
            np.random.seed(self.worker_seed)
            torch.manual_seed(self.worker_seed)
            
            # 确保环境也使用正确的种子
            if hasattr(self.env, 'seed'):
                self.env.seed(self.worker_seed)

            total_step = 1
            while self.g_ep.value < self.episode_num:
                s = self.env.reset()
                buffer_s, buffer_a, buffer_r = [], [], []
                ep_r = 0.
                while True:
                    if self.name == 'w00':
                        self.env.render()
                    
                    # 选择动作
                    a = self.lnet.choose_action(
                        v_wrap(s[None, :]), 
                        self.env.available_actions
                    )
                    # 执行动作
                    s_, r, done, _ = self.env.step(a)
                    if done:
                        r = -1 
                    
                    # 存储经验
                    ep_r += r
                    buffer_a.append(np.array(a))
                    buffer_s.append(s)
                    buffer_r.append(r)
                    # print("buffer_r:", buffer_r)

                    # 定期更新全局网络
                    if total_step % self.global_iter == 0 or done:
                        c_loss, a_loss = push_and_pull(
                            self.opt, self.lnet, self.gnet, 
                            done, s_, buffer_s, buffer_a, buffer_r, 
                            self.gamma, self.device
                        )
                        if c_loss is not None and a_loss is not None:
                            global_episode = self.g_ep.value
                            writer.add_scalar('Loss/Critic', c_loss, total_step)
                            writer.add_scalar('Loss/Actor', a_loss, total_step)
                            writer.add_scalar('Loss/Total', c_loss+a_loss, total_step)
                    
                        buffer_s, buffer_a, buffer_r = [], [], []

                        if done:
                            record(
                                self.g_ep, self.g_ep_r, ep_r, 
                                self.res_queue, self.name, self.local_best_reward
                            )
                            if ep_r > self.local_best_reward:
                                self.local_best_reward = ep_r
                                self.local_best_sequence = self.env.current_sequence.copy()
                                self.best_reward_episode = self.g_ep.value
                            break
                    s = s_
                    total_step += 1

                    if self.g_ep.value >= self.episode_num:
                        break

                if self.g_ep.value >= self.episode_num:
                    break
            
            # 进程结束时传递结果
            self.res_queue.put((
                self.local_best_reward, 
                self.local_best_sequence, 
                self.best_reward_episode
            ))
            self.res_queue.put(None)

        except Exception as e:
            print(f"Worker {self.name} 发生异常: {str(e)}")