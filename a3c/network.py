# network.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import set_init

class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 策略网络
        self.actor_fc1 = nn.Linear(state_dim, 256)
        self.actor_fc2 = nn.Linear(256, action_dim)
        
        # 价值网络
        self.critic_fc1 = nn.Linear(state_dim, 256)
        self.critic_fc2 = nn.Linear(256, 1)

        # 修改Critic输出层（c=100）
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.LayerNorm(256),
        #     nn.SiLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.SiLU(),
        #     nn.Linear(128, 1)  # 直接输出，不添加约束
        # )

        # # region 改
        # self.shared_fc = nn.Sequential(
        #     nn.Linear(state_dim, 256),
        #     nn.LayerNorm(256),
        #     nn.SiLU(),
        #     nn.Dropout(0.1)
        # )
        
        # # 策略网络
        # self.actor = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.SiLU(),
        #     nn.Linear(128, action_dim)
        # )
        
        # # 价值网络
        # self.critic = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.LayerNorm(128),
        #     nn.SiLU(),
        #     nn.Linear(128, 1)
        # )
        
        # # 初始化权重
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.orthogonal_(m.weight, gain=0.01)
        #         nn.init.constant_(m.bias, 0)
        # # endregion
        
        # 初始化
        set_init([self.actor_fc1, self.actor_fc2, self.critic_fc1, self.critic_fc2])
        self.distribution = torch.distributions.Categorical

    def forward(self, x):
        # 策略分支
        actor = torch.relu(self.actor_fc1(x))
        logits = self.actor_fc2(actor)
        
        # 价值分支
        critic = torch.relu(self.critic_fc1(x))
        values = self.critic_fc2(critic)
        # c=100
        # values = self.critic(x)

        # region
        # shared_features = self.shared_fc(x)
        # logits = self.actor(shared_features)
        # values = self.critic(shared_features)
        # endregion
        
        return logits, values

    def choose_action(self, s, available_actions):
        self.eval()
        logits, _ = self.forward(s)
        prob = F.softmax(logits, dim=1).data
        
        # 过滤无效动作
        available_actions = [a for a in available_actions if a < self.action_dim]
        prob = prob[:, available_actions]
        
        # 数值稳定性处理
        prob = prob + 1e-8
        prob = prob / prob.sum()
        
        m = self.distribution(prob)
        return available_actions[m.sample().item()]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        #=====================
        td = v_t - values
        c_loss = td.pow(2)
        
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        # a_loss = -exp_v - 0.1 * entropy
        a_loss = -exp_v
        # ====================

        # # 1. Critic使用Huber+正则化
        # td = v_t - values
        # c_loss = F.huber_loss(values, v_t, delta=1.0)  # 改用Huber损失
        
        # # 2. Actor损失
        # probs = F.softmax(logits, dim=1)
        # m = self.distribution(probs)  # 关键修复：定义分布实例
        
        # entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
        # exp_v = m.log_prob(a) * td.detach().squeeze()
        # a_loss = -exp_v.mean() - 0.01 * entropy

        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
 

        loss_ratio = c_loss.mean().abs() / (a_loss.mean().abs() + 1e-8)
        # print("ratio:", loss_ratio.item())
        # return (c_loss + a_loss).mean()
        return c_loss, a_loss