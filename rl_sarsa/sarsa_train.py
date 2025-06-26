import numpy as np
from tqdm import tqdm
from jy_exp.rl_common.class_op import ClassOp

def SARSA_train(env, agent, num_episodes):
    return_list = []
    best_OCplx_sequence = []        #记录全局最佳(最低）的OCplx序列
    best_OCplx_reward = -np.inf     #记录全局最佳的OCplx的reward
    best_OCplx = float('inf')       #记录全局最佳的OCplx
    best_reward_OCplx = float('inf')  #记录全局最佳奖励对应的OCplx
    best_reward_sequence = []              #记录全局最佳序列
    best_reward = -np.inf
    total_reward = 0
    warmup_episodes = 200
    
    for i in range(10):
        env.available_actions = list(range(env.num_classes)) 
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(num_episodes // 10):
                state = env.reset()
                action = agent.choose_action(state, env.available_actions)
                episode_return = 0
                done = False
                
                while not done:
                    # 执行动作
                    next_state, reward, done, _ = env.step(action)
                    if done:
                        break
                    # 选择下一个动作
                    next_action = agent.choose_action(next_state, env.available_actions)
                    # 学习
                    agent.learn(state, action, reward, next_state, next_action, done)
                    
                    episode_return += reward
                    state = next_state
                    action = next_action

                current_global_episode = num_episodes / 10 * i + i_episode + 1
                if current_global_episode > warmup_episodes:
                    # 更新滑动平均奖励
                    if len(return_list) == 0:
                        total_reward = episode_return
                    else:
                        total_reward = 0.98 * total_reward + 0.02 * episode_return
                    return_list.append(total_reward)

                    # 更新最佳OCplx
                    current_sequence = env.current_sequence
                    current_sequence = [num + 1 for num in env.current_sequence]
                    OCplx, _, _, _ = ClassOp.calculate_OCplx_sequence(
                        env.attributes, env.methods, current_sequence, w_M=env.wM, w_A=env.wA
                    )
                    if OCplx < best_OCplx and OCplx != 0:
                        best_OCplx = OCplx
                        best_OCplx_sequence = current_sequence.copy()
                        best_OCplx_reward = episode_return

                    if episode_return > best_reward:
                        best_reward = episode_return
                        best_reward_sequence = current_sequence.copy()
                        best_reward_OCplx = OCplx

                # if len(return_list) == 0:
                #     total_reward = episode_return
                # else:
                #     total_reward = 0.98 * total_reward + 0.02 * episode_return
                
                # return_list.append(total_reward)
                # current_sequence = env.current_sequence
                # current_sequence = [num + 1 for num in current_sequence]  # 将类编号从0开始改为从1开始
                # OCplx, _, _, _ = ClassOp.calculate_OCplx_sequence(env.attributes, env.methods, current_sequence, w_M=env.wM, w_A=env.wA)

                # if OCplx < best_OCplx and OCplx != 0:
                #     best_OCplx = OCplx
                #     best_OCplx_sequence = current_sequence.copy()
                #     best_OCplx_reward = episode_return

                #                 # 更新全局最佳奖励和序列
                # if episode_return > best_reward:
                #     best_reward = episode_return
                #     best_reward_sequence = current_sequence.copy()
                #     best_reward_OCplx = OCplx

                # 更新进度条
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
    print("全局最佳的 reward：", best_reward)
    return best_OCplx, best_OCplx_sequence, return_list
    
    # print("全局最佳奖励对应的OCplx：", best_reward_OCplx)
    # print("全局最佳的 reward：", best_reward)
    # print("全局最佳的 reward序列：", best_reward_sequence)
    # return best_reward_OCplx, best_reward_sequence, return_list