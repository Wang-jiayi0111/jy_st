import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR  
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/dqn')

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 256)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_actions)
        # self.out.weight.data.normal_(0, 0.1)

        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class DQN(nn.Module):
    def __init__(self, n_states, n_actions, lr, eps, gamma, targer_repalce_iter, memory_capacity, batch_size, device):
        super(DQN, self).__init__()
        # 评估网络和目标网络
        self.eval_net = Net(n_states, n_actions).to(device)
        self.target_net = Net(n_states, n_actions).to(device)
        # 优化器和损失函数
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.SmoothL1Loss()
        # 添加学习率调度器（示例：每1000步学习率乘以0.9）
        self.scheduler = StepLR(self.optimizer, step_size=800, gamma=0.80)
        # 算法参数
        self.n_action = n_actions
        self.n_state = n_states
        # self.eps = eps
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2))
        self.target_replace_iter = targer_repalce_iter
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.device = device
        self.distribution = torch.distributions.Categorical
        self.eps_start = 0.5       #0.9
        self.eps_end = 0.05             #0.05
        self.eps_decay = 1000           # 0.995
        self.eps = self.eps_start


    def choose_action(self, s, available_actions):
        self.eval_net.eval()  # 设置模型为评估模式
        s = torch.tensor([s], dtype=torch.float).to(self.device)
        # s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        with torch.no_grad():  # 不计算梯度
            logits = self.eval_net(s)  # 直接调用 eval_net 的 forward
        prob = F.softmax(logits, dim=1).data  # 计算动作概率
        available_actions = [i for i in range(prob.shape[1]) if i in available_actions]
        prob = prob[:, available_actions]  # 只保留可用动作的概率

        # 添加数值稳定性处理
        prob = prob + 1e-8  # 避免除零错误
        prob = prob / prob.sum()  # 重新归一化概率

        m = self.distribution(prob)  # 创建动作分布
        # sampled_action = m.sample().cpu().numpy()[0]  # 采样动作
        sampled_action = m.sample().item()  # 采样动作
        res = available_actions[sampled_action]  # 采样动作并映射回原动作空间
        return res  # 返回采样的动作


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1    

    def update_target_network(self):
        """渐进式更新目标网络"""
        tau = 0.01  # 混合系数
        for target_param, eval_param in zip(self.target_net.parameters(), 
                                        self.eval_net.parameters()):
            target_param.data.copy_(tau * eval_param.data + (1-tau) * target_param.data)  

    def learn(self):
        self.eval_net.train()  # 设置模型为训练模式

        # # 定期更新目标网络
        # if self.learn_step_counter % self.target_replace_iter == 0:
        #     self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.update_target_network()  # 改为调用软更新
        self.learn_step_counter += 1

        # 从经验回放中采样
        sample_size = min(self.memory_counter, self.memory_capacity)
        sample_index = np.random.choice(sample_size, self.batch_size)
        b_memory = self.memory[sample_index, :]

        # 提取批数据
        b_s = torch.FloatTensor(b_memory[:, :self.n_state]).to(self.device)
        b_a = torch.LongTensor(b_memory[:, self.n_state:self.n_state+1].astype(int)).to(self.device)
        b_r = torch.FloatTensor(b_memory[:, self.n_state+1:self.n_state+2]).to(self.device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_state:]).to(self.device)

        # 计算当前 Q 值
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # 计算目标 Q 值
        with torch.no_grad():
            q_next = self.target_net(b_s_).max(1)[0].view(self.batch_size, 1)
            q_target = b_r + self.gamma * q_next

        # 计算损失并更新网络
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()  # 更新学习率

        # tensorboard记录
        current_lr = self.optimizer.param_groups[0]['lr']
        writer.add_scalar('LearningRate', current_lr, self.learn_step_counter)
        writer.add_scalar('Loss', loss.item(), global_step=self.learn_step_counter)
        writer.add_scalar('Exploration/epsilon', self.eps, self.learn_step_counter)
        writer.add_scalar('Qvalue/mean', q_eval.mean().item(), self.learn_step_counter)