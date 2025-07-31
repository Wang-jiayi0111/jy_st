import numpy as np
from tqdm import tqdm
from jy_exp.rl_common.class_op import ClassOp

def DQN_train(env, agent, num_episodes):
    return_list = []
    best_OCplx_sequence = []        #记录全局最佳(最低）的OCplx序列
    best_OCplx_reward = -np.inf     #记录全局最佳的OCplx的reward
    best_OCplx = float('inf')       #记录全局最佳的OCplx
    total_reward = 0
    warmup_episodes = 200
    for i in range(10):
        env.available_actions = list(range(env.num_classes)) 
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:

            for i_episode in range(num_episodes // 10):
                state = env.reset()
                episode_return = 0
                done = False
                while not done:
                    # 选择动作
                    action = agent.choose_action(state, env.available_actions)
                    # 执行动作
                    next_state, reward, done, _ = env.step(action)
                    # 保存经验数据
                    agent.store_transition(state, action, reward, next_state)
                    episode_return += reward
                    # 下一个状态
                    state = next_state
                    agent.learn()
                
                # 仅在预热期结束后记录数据
                current_global_episode = num_episodes / 10 * i + i_episode + 1
                if current_global_episode > warmup_episodes:
                    # 更新滑动平均奖励
                    if len(return_list) == 0:
                        total_reward = episode_return
                    else:
                        total_reward = 0.98 * total_reward + 0.02 * episode_return
                    return_list.append(total_reward)

                    # 更新最佳OCplx
                    current_sequence = [num + 1 for num in env.current_sequence]
                    OCplx, _, _, _ = ClassOp.calculate_OCplx_sequence(
                        env.attributes, env.methods, current_sequence, w_M=env.wM, w_A=env.wA
                    )
                    if OCplx < best_OCplx and OCplx != 0:
                        best_OCplx = OCplx
                        best_OCplx_sequence = current_sequence.copy()
                        best_OCplx_reward = episode_return

                
                # 更新进度条
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-1000:]),
                        })
                pbar.update(1)

    print("全局最佳（最低）的 OCplx：", best_OCplx)
    print("全局最佳（最低）的 OCplx序列：", best_OCplx_sequence)
    print("全局最佳（最低）的 OCplx对应的reward：", best_OCplx_reward)

    return return_list, best_OCplx, best_OCplx_sequence