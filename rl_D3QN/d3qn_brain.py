import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./logs/d3qn')

class DuelingNet(nn.Module):
    """Dueling Network架构"""
    def __init__(self, n_states, n_actions):
        super(DuelingNet, self).__init__()
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # Advantage流
        self.advantage = nn.Linear(128, n_actions)
        # Value流
        self.value = nn.Linear(128, 1)

            # 价值流V的初始化范围较大（因为需要估计绝对状态价值）
        nn.init.uniform_(self.value.weight, -0.1, 0.1)
        nn.init.constant_(self.value.bias, 0.0)
        
        # 优势流A的初始化范围较小（因为只需要估计相对优势）
        nn.init.uniform_(self.advantage.weight, -0.01, 0.01)
        nn.init.constant_(self.advantage.bias, 0.0)

    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        # 合并：Q = V + A - mean(A)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

class D3QN(nn.Module):
    def __init__(self, n_states, n_actions, lr, eps, gamma, target_replace_iter, 
                 memory_capacity, batch_size, num_episodes, device):
        super(D3QN, self).__init__()
        # 评估网络和目标网络（使用DuelingNet）
        self.eval_net = DuelingNet(n_states, n_actions).to(device)
        self.target_net = DuelingNet(n_states, n_actions).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())  # 初始同步
        
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()  # Huber Loss
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.9)
        
        # 算法参数
        self.n_actions = n_actions
        self.n_states = n_states
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))  # [s, a, r, s_]
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.device = device
        
        # 探索率参数
        self.total_steps = num_episodes
        self.eps_start = eps
        self.eps_end = 0.01
        self.eps_decay = 0.9995  # 衰减率
        self.eps = self.eps_start

    # def choose_action(self, s, available_actions):
    #     s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         eval_q = self.eval_net(s)
    #         mask = torch.full_like(eval_q, -np.inf)
    #         mask[:, available_actions] = eval_q[:, available_actions]
    #         return mask.argmax(dim=1).item()

    def choose_action(self, s, available_actions):
        if np.random.random() < self.eps:
            return np.random.choice(available_actions)
        else:
            s = torch.FloatTensor(s).unsqueeze(0).to(self.device)
            with torch.no_grad():
                eval_q = self.eval_net(s)
                mask = torch.full_like(eval_q, -np.inf)
                mask[:, available_actions] = eval_q[:, available_actions]
                return mask.argmax(dim=1).item()

    def store_transition(self, s, a, r, s_):
        """存储经验到回放缓冲区"""
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_target_network(self):
        """软更新目标网络"""
        tau = 0.005  # 混合系数
        for target_param, eval_param in zip(self.target_net.parameters(), 
                                         self.eval_net.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1 - tau) * target_param.data)

    def learn(self):
        """训练网络（Double DQN + Dueling）"""
        if self.memory_counter < self.batch_size:
            return
        
        # 从回放缓冲区采样
        sample_index = np.random.choice(min(self.memory_counter, self.memory_capacity), 
                                      self.batch_size, replace=False)
        b_memory = self.memory[sample_index, :]
        
        # 提取数据并转为Tensor
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(self.device)

        # Double DQN逻辑
        with torch.no_grad():
            # 用eval_net选择动作
            max_actions = self.eval_net(b_s_).max(1)[1].unsqueeze(1)
            # 用target_net计算目标Q值
            q_next = self.target_net(b_s_).gather(1, max_actions)
            q_target = b_r + self.gamma * q_next

        # 计算当前Q值
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # 计算损失并更新
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_net.parameters(), max_norm=1.0)  # 梯度裁剪
        self.optimizer.step()
        self.scheduler.step()
  
        if self.learn_step_counter % 5 == 0:  # 每隔5步触发更新
            self.eps =  max(self.eps_end, self.eps * self.eps_decay)

        # 记录日志
        writer.add_scalar('Qvalue/mean', q_eval.mean().item(), self.learn_step_counter)
        writer.add_scalar('Loss', loss.item(), self.learn_step_counter)
        writer.add_scalar('Exploration/epsilon', self.eps, self.learn_step_counter)
        self.learn_step_counter += 1

        self.update_target_network()