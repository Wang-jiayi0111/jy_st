import torch
import torch.nn as nn
import numpy as np

# v_wrap函数：将numpy数组转换为PyTorch张量
def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)  # 返回转换后的张量


# set_init函数：初始化神经网络的权重和偏置
def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)  # 使用正态分布初始化权重
        nn.init.constant_(layer.bias, 0.)  # 将偏置初始化为0

# record函数：记录和打印训练进度
def record(global_ep, global_ep_r, ep_r, res_queue, name, local_best_reward):
    with global_ep.get_lock():
        global_ep.value += 1  # 增加全局训练轮数
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r  # 初始化全局累积奖励
        else:
            global_ep_r.value = global_ep_r.value * 0.98 + ep_r * 0.02  # 更新全局累积奖励
            # global_ep_r.value =  ep_r  # 更新全局累积奖励

    res_queue.put(global_ep_r.value)  # 将累积奖励放入队列
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.3f" % global_ep_r.value,  # 打印当前进程的训练轮数和累积奖励
    )

# push_and_pull函数：同步本地网络和全局网络的参数
def push_and_pull(opt, lnet, gnet, done, s_, buffer_s, buffer_a, buffer_r, gamma, device):
    # 计算目标价值v_s_
    if done:
        v_s_ = 0.  
    else:
        v_s_ = lnet.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0, 0]

    # 计算目标价值序列buffer_v_target
    buffer_v_target = []
    for r in buffer_r[::-1]:  # reverse buffer r
        # r = r * 1/1000
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # reward_scale = 1.0 / (np.std(buffer_r) + 1e-8)  # 基于奖励标准差
    # buffer_v_target = []
    # for r in buffer_r[::-1]:
    #     print("r:", r)
    #     r = r * min(reward_scale, 10.0)  # 限制最大缩放倍数
    #     v_s_ = r + gamma * v_s_
    #     buffer_v_target.append(v_s_)
    # buffer_v_target.reverse()

    # 计算损失
    c_loss, a_loss  = lnet.loss_func(
        # 状态
        v_wrap(np.vstack(buffer_s)),   
        # 动作     
        v_wrap(np.array(buffer_a), dtype=np.int64) if buffer_a[0].dtype == np.int64 else v_wrap(np.vstack(buffer_a)),
        # 目标价值
        v_wrap(np.array(buffer_v_target)[:, None])
    )
    

    # 合并损失为一个标量
    total_loss = c_loss.mean() + a_loss.mean()  # 确保是标量
    
    # 反向传播
    opt.zero_grad()
    total_loss.backward()

    # 梯度裁剪 (防止爆炸)
    torch.nn.utils.clip_grad_norm_(lnet.parameters(), 0.5)


    # 将本地网络的梯度赋值给全局网络的梯度
    for lp, gp in zip(lnet.parameters(), gnet.parameters()):
        gp._grad = lp.grad

    # 更新全局网络参数
    opt.step()

    # 拉取全局网络参数到本地网络
    lnet.load_state_dict(gnet.state_dict())
    return c_loss.mean().item(), a_loss.mean().item()
