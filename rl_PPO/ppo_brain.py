import torch
import torch.nn.functional as F
import torch.nn as nn
import jy_exp.rl_PPO.ppo_train as ppo_train
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='./logs/ppo')

# region 原来的
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
    #     self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
    #     self.fc2 = torch.nn.Linear(hidden_dim, action_dim)


    # def forward(self, x):

    #     x = torch.relu(self.fc1(x))
    #     return F.softmax(self.fc2(x), dim=1)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)  # 新增BN层
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))  # 应用BN
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # return self.fc2(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)
# endregion

# 修改PolicyNet，增加更多层和Dropout
# class PolicyNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(PolicyNet, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.dropout1 = nn.Dropout(0.2)  # 新增Dropout层
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)  # 新增中间层
#         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
#         self.dropout2 = nn.Dropout(0.1)
#         self.fc3 = nn.Linear(hidden_dim//2, action_dim)
        
#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = self.dropout2(x)
#         return F.softmax(self.fc3(x), dim=-1)

# class ValueNet(torch.nn.Module):
#     def __init__(self, state_dim, hidden_dim):
#         super(ValueNet, self).__init__()
#         self.fc1 = nn.Linear(state_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim)
#         self.dropout1 = nn.Dropout(0.1)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
#         self.bn2 = nn.BatchNorm1d(hidden_dim//2)
#         self.fc3 = nn.Linear(hidden_dim//2, 1)
        
#     def forward(self, x):
#         x = F.relu(self.bn1(self.fc1(x)))
#         x = self.dropout1(x)
#         x = F.relu(self.bn2(self.fc2(x)))
#         return self.fc3(x)


class PPO(nn.Module):
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        super(PPO, self).__init__()
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,  # 原3e-4 → 1e-4
            eps=1e-7   # 提高数值稳定性
        )

    # 添加学习率调度器
        self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.actor_optimizer, 'min', patience=10, factor=0.7, min_lr=1e-6, verbose=True)
        self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.critic_optimizer, 'min', patience=10, factor=0.7, min_lr=1e-5, verbose=True)

        self.action_dim = action_dim
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs    # 一条序列的数据用来训练轮数
        self.eps = eps          # PPO中截断范围的参数
        self.device = device
        self.distribution = torch.distributions.Categorical  # 动作分布
        self.step_count = 0

    def take_action(self, s, available_actions):
        self.eval()  # 设置模型为评估模式
        s = torch.tensor([s], dtype=torch.float).to(self.device)
        prob = self.actor(s)  # 计算动作概率
        available_actions = [i for i in range(prob.shape[1]) if i in available_actions]
        prob = prob[:, available_actions]  # 只保留可用动作的概率

        # 添加数值稳定性处理
        prob = prob + 1e-8  # 避免除零错误
        prob = prob / prob.sum()  # 重新归一化概率
        # print(f"Filtered action probabilities: {prob}")

        m = self.distribution(prob)  # 创建动作分布
        res = available_actions[m.sample().cpu().numpy()[0]]  # 采样动作并映射回原动作空间
        return res  # 返回采样的动作

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        with torch.no_grad():
            #计算TD目标（目标价值）
            rewards = rewards * 1/1000
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            # print(f"TD Target - Max: {td_target.max().item():.2f}, Min: {td_target.min().itxem():.2f}")
            # values = self.critic(states)
            #计算TD误差
            td_delta = td_target - self.critic(states)
            #计算优势函数
            advantage = ppo_train.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            # print(f"Advantage - Max: {advantage.max().item():.2f}, Min: {advantage.min().item():.2f}")
            #计算旧策略的对数概率
            old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        
        self.train()
        #更新策略网络和价值网络
        for i in range(self.epochs):
            # print("第", i, "轮训练")
            #计算新策略对数概率
            log_probs = torch.log(self.actor(states).gather(1, actions)+1e-8)
            #计算概率比
            ratio = torch.exp(log_probs - old_log_probs)
            #计算PPO的目标概率
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantage  # 截断

            #计算价值损失
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            # critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            critic_loss = F.mse_loss(self.critic(states), td_target.detach())

            # 计算策略熵
            probs = self.actor(states)
            log_probs = torch.log(probs)

            entropy = -torch.sum(probs * log_probs, dim=1).mean()
            target_entropy = -0.5 * torch.log(torch.tensor(self.action_dim, dtype=torch.float))
            current_entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1).mean()
            # 动态调整系数，初期大后期小
            entropy_coef = torch.clamp(0.1 * (current_entropy - target_entropy) + 0.01, 
                                    min=0.001, max=0.05)

            total_loss = 1.0 * actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            #清空梯度
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            #反向传播
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
    
            # 更保守的参数更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()
    
            # # 只在epoch结束时更新学习率
            # if i == self.epochs - 1:
            #     self.actor_scheduler.step(actor_loss.item())
            #     self.critic_scheduler.step(critic_loss.item())

            # tensorboard记录
            writer.add_scalar('Loss/actor', actor_loss.item(), global_step=self.step_count)
            writer.add_scalar('Loss/critic', critic_loss.item(), global_step=self.step_count)
            writer.add_scalar('Train/Entropy', entropy.item(), self.step_count)
            writer.add_scalar('Train/Entropy_Coeff', entropy_coef, self.step_count)
            writer.add_scalar('Stats/Advantage_Mean', advantage.mean().item(), self.step_count)

            self.step_count += 1
            

