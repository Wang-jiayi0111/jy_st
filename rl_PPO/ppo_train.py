from tqdm import tqdm
import numpy as np
import torch
import collections
import random
from jy_exp.rl_common.class_op import ClassOp

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    best_OCplx_sequence = []        #记录全局最佳(最低）的OCplx序列
    best_OCplx_reward = -np.inf     #记录全局最佳的OCplx的reward
    best_OCplx = float('inf')       #记录全局最佳的OCplx
    best_reward_OCplx = float('inf')  #记录全局最佳奖励对应的OCplx
    best_reward_sequence = []              #记录全局最佳序列
    best_reward = -np.inf
    total_reward = 0

    for i in range(10):
        env.available_actions = list(range(env.num_classes)) 
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:

            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                #与环境交互获取经验数据
                step_count = 0      #步数计数器

                while not done:
                    # 根据当前状态选择一个动作
                    action = agent.take_action(state, env.available_actions)
                    #执行动作
                    current_state = state.copy()
                    next_state, reward, done, _ = env.step(action)
                    # 保存经验数据
                    transition_dict['states'].append(current_state.copy())
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state.copy())
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)

                    state = next_state
                    episode_return += reward
                    step_count += 1

                if len(return_list) == 0:
                    total_reward = episode_return
                else:
                    total_reward = 0.98 * total_reward + 0.02 * episode_return
                #更新智能体的策略网络和价值网络
                agent.update(transition_dict)

                # 记录当前episode的return
                return_list.append(total_reward)
                # print("episode:", i_episode, "return:", total_reward)
                #计算当前序列的OCplx
                current_sequence = env.current_sequence
                current_sequence = [num + 1 for num in current_sequence]  # 将类编号从0开始改为从1开始
                OCplx, _, _, _ = ClassOp.calculate_OCplx_sequence(env.attributes, env.methods, current_sequence, w_M=env.wM, w_A=env.wA)

                # 更新全局最佳的OCplx和序列
                if OCplx < best_OCplx and OCplx != 0:
                    best_OCplx = OCplx
                    best_OCplx_sequence = current_sequence.copy()
                    best_OCplx_reward = episode_return
                
                # 更新全局最佳奖励和序列
                if episode_return > best_reward:
                    best_reward = episode_return
                    best_reward_sequence = current_sequence.copy()
                    best_reward_OCplx = OCplx

                #更新进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-1000:]),
                        })
                pbar.update(1)
    if best_reward == best_OCplx_reward:
        print("全局最佳奖励和最低OCplx的奖励相同")

    # 打印全局最佳的 OCplx 和序列
    print("全局最佳（最低）的 OCplx：", best_OCplx)
    print("全局最佳（最低）的 OCplx序列：", best_OCplx_sequence)
    print("全局最佳（最低）的 OCplx对应的reward：", best_OCplx_reward)
    return best_OCplx, best_OCplx_sequence, return_list
    
    # print("全局最佳奖励对应的OCplx：", best_reward_OCplx)
    # print("全局最佳的 reward：", best_reward)
    # print("全局最佳的 reward序列：", best_reward_sequence)
    # return best_reward_OCplx, best_reward_sequence, return_list


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.flip(dims=[0])  # 替代numpy的[::-1]
    advantage = 0
    advantage_list = []
    for delta in td_delta:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list = torch.stack(advantage_list).flip(dims=[0])  # 恢复顺序
    return advantage_list
