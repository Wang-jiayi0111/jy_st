import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/sarsa')

class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        # self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, n_actions)
        # self.fc2.weight.data.normal_(0, 0.1)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

class SARSA(nn.Module):
    def __init__(self, n_states, n_actions, lr, eps, gamma, device):
        super(SARSA, self).__init__()
        # 只需要一个评估网络
        self.eval_net = Net(n_states, n_actions).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss_func = nn.MSELoss()
        
        # 算法参数
        self.n_action = n_actions
        self.n_state = n_states
        self.eps = eps
        self.gamma = gamma
        self.device = device
        self.distribution = torch.distributions.Categorical
        self.step_count = 0

    def choose_action(self, s, available_actions):
        self.eval_net.eval()  # 设置模型为评估模式
        s = torch.tensor([s], dtype=torch.float).to(self.device)
        with torch.no_grad():
            logits = self.eval_net(s)
        prob = F.softmax(logits, dim=1).data
        available_actions = [i for i in range(prob.shape[1]) if i in available_actions]
        prob = prob[:, available_actions]
        
        # 数值稳定性处理
        prob = prob + 1e-8
        prob = prob / prob.sum()
        
        m = self.distribution(prob)
        sampled_action = m.sample().item()
        res = available_actions[sampled_action]
        return res

    def learn(self, s, a, r, s_, a_, done):
        self.eval_net.train()  # 设置模型为训练模式
        
        # 转换为tensor
        s = torch.FloatTensor([s]).to(self.device)
        a = torch.LongTensor([[a]]).to(self.device)
        r = torch.FloatTensor([r]).to(self.device)
        s_ = torch.FloatTensor([s_]).to(self.device)
        a_ = torch.LongTensor([[a_]]).to(self.device)
        
        # 计算当前Q值
        q_eval = self.eval_net(s).gather(1, a)
        
        # 计算目标Q值 (SARSA使用下一个动作a_)
        with torch.no_grad():
            if done:
                q_target = r
            else:
                q_next = self.eval_net(s_).gather(1, a_)
                q_target = r + self.gamma * q_next
        
        # 计算损失并更新网络
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        
        writer.add_scalar('Loss', loss.item(), global_step=self.step_count)
        self.step_count += 1
        self.optimizer.step()