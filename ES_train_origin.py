#!/usr/bin/env python3
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from FJSP_Env import FJSP
from Params import configs
from mb_agg import *
from policy import Policy
from uniform_instance import FJSPDataset
from validation_optimization_3_obj import validate2

# 设置设备（GPU或CPU）
device = torch.device(configs.device)
print(f"Using device: {device}")

# 训练参数
MAX_BATCH_EPISODES = 100
MAX_BATCH_STEPS = 1200
NOISE_STD = 0.01
LEARNING_RATE = 0.001

# 文件路径设置
filepath = '../saved_network_MOFJSP'
TIMESTAMP = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))


def evaluate(env, data, agent, g_pool_step, pref, device):
    """
    评估策略在给定环境上的表现
    参数:
        env: FJSP环境实例
        data: 输入数据
        agent: 策略代理
        g_pool_step: 图池化对象
        pref: 偏好向量
        device: 计算设备
    返回:
        reward: 计算得到的奖励值
    """
    # 重置环境
    adj, fea, candidate, mask, mask_mch, dur, mch_time, job_time,machines_batch = env.reset(data)

    # 将数据移动到指定设备
    env_mask_mch = torch.from_numpy(mask_mch).to(device)
    env_dur = torch.from_numpy(dur).float().to(device)
    pool = None

    while True:
        # 处理邻接矩阵 - 修复CUDA错误的关键部分
        adj_temp = torch.from_numpy(adj)

        # 安全处理：先在CPU上转换为稀疏张量，再移动到目标设备
        try:
            # 1. 先移动到CPU
            adj_temp_cpu = adj_temp.cpu()

            # 2. 转换为稀疏格式（在CPU上）
            adj_temp_sparse = adj_temp_cpu.to_sparse()

            # 3. 移动到目标设备
            adj_temp_sparse_gpu = adj_temp_sparse.to(device)

            # 4. 聚合观察
            env_adj = aggr_obs(adj_temp_sparse_gpu, configs.n_j * configs.n_m)
        except Exception as e:
            print(f"Error processing adjacency matrix: {e}")
            # 回退方案：使用密集张量
            env_adj = adj_temp.to(device)

        # 处理其他特征
        env_fea = torch.from_numpy(fea).float().to(device)
        env_fea = env_fea.reshape(-1, env_fea.size(-1))
        env_candidate = torch.from_numpy(candidate).long().to(device)
        env_mask = torch.from_numpy(mask).to(device)
        env_mch_time = torch.from_numpy(mch_time).float().to(device)

        # 策略决策（无梯度计算）
        with torch.no_grad():
            action, a_idx, log_a, action_node, _, mask_mch_action, hx = agent.policy_job(
                x=env_fea,
                graph_pool=g_pool_step,
                padded_nei=None,
                adj=env_adj,
                candidate=env_candidate,
                mask=env_mask,
                mask_mch=env_mask_mch,
                dur=env_dur,
                a_index=0,
                old_action=0,
                mch_pool=pool,
                old_policy=True,
                T=1,
                greedy=True
            )

            pi_mch, pool = agent.policy_mch(action_node, hx, mask_mch_action, env_mch_time,env_mch_time)

        # 选择机器动作
        _, mch_a = pi_mch.squeeze(-1).max(1)

        # 环境执行动作
        adj, fea, reward, done, candidate, mask, job, _, mch_time, job_time,machines_batch = env.step(
            action.cpu().numpy(), mch_a
        )

        # 检查是否完成
        if env.done_batch.all():
            # 确保pref在CPU上计算（因为环境数据在CPU）
            pref_cpu = pref.cpu()

            # 计算最终奖励（双目标）
            # 目标1: 最大完工时间
            # 目标2: 机器总负荷
            reward = (
                    pref_cpu[0] * env.schedules_batch[:, :, 3].max(-1)[0] +
                    pref_cpu[1] * env.machines_batch[0].sum(-1)
            )
            reward = -reward.item()  # 取负值因为是最小化问题
            break

    return reward


def sample_noise(agent, device):
    """
    为策略参数生成正负噪声
    参数:
        agent: 策略代理
        device: 计算设备
    返回:
        正负噪声列表
    """
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch
    actor_job_pos = []
    actor_job_neg = []

    # 为作业策略生成噪声
    for p in actor_job.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise).to(device)
        actor_job_pos.append(noise_t)
        actor_job_neg.append(-noise_t)

    actor_mch_pos = []
    actor_mch_neg = []

    # 为机器策略生成噪声
    for p in actor_mch.parameters():
        noise = np.random.normal(size=p.data.size())
        noise_t = torch.FloatTensor(noise).to(device)
        actor_mch_pos.append(noise_t)
        actor_mch_neg.append(-noise_t)

    return actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg


def eval_with_noise(env, data, agent, actor_job_noise, actor_mch_noise, g_pool_step, pref, device):
    """
    使用带噪声的策略进行评估
    参数:
        env: FJSP环境实例
        data: 输入数据
        agent: 策略代理
        actor_job_noise: 作业策略噪声
        actor_mch_noise: 机器策略噪声
        g_pool_step: 图池化对象
        pref: 偏好向量
        device: 计算设备
    返回:
        r: 评估奖励
    """
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch

    # 保存原始参数
    old_params_actor_job = {k: v.clone() for k, v in actor_job.state_dict().items()}
    old_params_actor_mch = {k: v.clone() for k, v in actor_mch.state_dict().items()}

    # 添加噪声到策略参数
    for p, p_n in zip(actor_job.parameters(), actor_job_noise):
        p.data += NOISE_STD * p_n
    for p, p_n in zip(actor_mch.parameters(), actor_mch_noise):
        p.data += NOISE_STD * p_n

    # 评估带噪声的策略
    r = evaluate(env, data, agent, g_pool_step, pref, device)

    # 恢复原始参数
    actor_job.load_state_dict(old_params_actor_job)
    actor_mch.load_state_dict(old_params_actor_mch)

    return r


def train_step(agent, actor_job_noise, actor_mch_noise, batch_reward):
    """
    使用评估结果更新策略参数
    参数:
        agent: 策略代理
        actor_job_noise: 作业策略噪声列表
        actor_mch_noise: 机器策略噪声列表
        batch_reward: 批次奖励列表
    """
    actor_job = agent.policy_job
    actor_mch = agent.policy_mch

    # 标准化奖励
    norm_reward = np.array(batch_reward)
    norm_reward -= np.mean(norm_reward)
    s = np.std(norm_reward)
    if abs(s) > 1e-6:
        norm_reward /= s
    norm_reward = torch.from_numpy(norm_reward).float()

    # 更新作业策略
    weighted_noise = None
    for noise, reward in zip(actor_job_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    for p, p_update in zip(actor_job.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update

    # 更新机器策略
    weighted_noise = None
    for noise, reward in zip(actor_mch_noise, norm_reward):
        if weighted_noise is None:
            weighted_noise = [reward * p_n for p_n in noise]
        else:
            for w_n, p_n in zip(weighted_noise, noise):
                w_n += reward * p_n

    for p, p_update in zip(actor_mch.parameters(), weighted_noise):
        update = p_update / (len(batch_reward) * NOISE_STD)
        p.data += LEARNING_RATE * update


def setup_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    # 设置目标权重（双目标）
    weight = torch.tensor([1,0,0])
    weight = weight / weight.sum()
    print("objective weight: ", weight)

    record = 0  # 记录最佳超体积分数

    # 创建数据集
    train_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high,
                                MAX_BATCH_EPISODES * MAX_BATCH_STEPS, 400)
    validat_dataset = FJSPDataset(configs.n_j, configs.n_m, configs.low, configs.high, 128, 400)
    valid_loader = DataLoader(validat_dataset, batch_size=configs.batch_size)

    data_loader = iter(train_dataset)

    # 创建图池
    g_pool_step = g_pool_cal(
        graph_pool_type=configs.graph_pool_type,
        batch_size=torch.Size([1, configs.n_j * configs.n_m, configs.n_j * configs.n_m]),
        n_nodes=configs.n_j * configs.n_m,
        device=device
    )

    # 设置随机种子
    setup_seed(200)

    # 创建策略代理
    agent = Policy(
        configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
        num_layers=configs.num_layers,
        neighbor_pooling_type=configs.neighbor_pooling_type,
        input_dim=configs.input_dim,
        hidden_dim=configs.hidden_dim,
        num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
        num_mlp_layers_actor=configs.num_mlp_layers_actor,
        hidden_dim_actor=configs.hidden_dim_actor,
        num_mlp_layers_critic=configs.num_mlp_layers_critic,
        hidden_dim_critic=configs.hidden_dim_critic,
        pref_dim=3  # 双目标
        , device=device
        ,n_ope=configs.n_j * configs.n_m

    )

    # 设置为评估模式
    agent.policy_job.eval()
    agent.policy_mch.eval()

    # 初始化记录列表
    vali_list = []
    hv_list = []
    score_list = []
    step_list = []

    # 创建环境实例（修复：在循环外部创建）
    ope_nums_of_jobs = np.array([configs.n_m for _ in range(configs.n_j)])
    env = FJSP(configs.n_j, configs.n_m,ope_nums_of_jobs=ope_nums_of_jobs)

    # 主训练循环
    for step_idx in range(1, MAX_BATCH_STEPS):
        actor_job_noise = []
        actor_mch_noise = []
        batch_reward = []

        # 生成随机偏好向量（双目标）
        pref = torch.rand([3], device=device)  # 直接在目标设备上创建
        pref = pref / torch.sum(pref)

        # 将偏好分配给策略
        agent.policy_job.assign(pref)
        agent.policy_mch.assign(pref)

        # 批次处理
        for _ in range(MAX_BATCH_EPISODES):
            # 为策略生成噪声
            actor_job_pos, actor_job_neg, actor_mch_pos, actor_mch_neg = sample_noise(agent, device)
            actor_job_noise.append(actor_job_pos)
            actor_job_noise.append(actor_job_neg)
            actor_mch_noise.append(actor_mch_pos)
            actor_mch_noise.append(actor_mch_neg)

            # 获取数据
            batch = next(data_loader)
            data = np.expand_dims(batch, axis=0)

            # 使用正噪声评估
            reward = eval_with_noise(
                env, data, agent,
                actor_job_pos, actor_mch_pos,
                g_pool_step, pref, device
            )
            batch_reward.append(reward)

            # 使用负噪声评估
            reward = eval_with_noise(
                env, data, agent,
                actor_job_neg, actor_mch_neg,
                g_pool_step, pref, device
            )
            batch_reward.append(reward)

        # 更新策略参数
        train_step(agent, actor_job_noise, actor_mch_noise, batch_reward)

        # 定期验证
        if step_idx == 0 or step_idx % 1 == 0:
            # 验证性能
            hv_score, score1, score2,score3, sum_score = validate2(valid_loader, agent, n_sols=11,device=device,pref=pref)

            print(f"Step id {step_idx}, hv is {hv_score:.2f}, sum_score is {sum_score:.2f}, "
                  f"score1 is {score1:.2f}, score2 is {score2:.2f}")

            # 记录结果
            step_list.append(step_idx)
            score_list.append(sum_score.item())
            hv_list.append(hv_score.item())

            # 保存最佳模型
            if record < hv_score:
                epoch_dir = os.path.join(filepath, 'makespan_and_total_time')
                epoch_dir = os.path.join(epoch_dir, TIMESTAMP)
                if not os.path.exists(epoch_dir):
                    os.makedirs(epoch_dir)

                print(f"#######Save Step id {step_idx}, hv is {hv_score} #########")
                job_savePath = os.path.join(epoch_dir, 'policy_job.pth')
                machine_savePate = os.path.join(epoch_dir, 'policy_mch.pth')

                torch.save(agent.policy_job.state_dict(), job_savePath)
                torch.save(agent.policy_mch.state_dict(), machine_savePate)
                record = hv_score

    # 绘制结果
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(step_list, score_list)
    plt.title("Sum Score over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Score")

    plt.subplot(1, 2, 2)
    plt.plot(step_list, hv_list)
    plt.title("Hypervolume over Steps")
    plt.xlabel("Steps")
    plt.ylabel("Hypervolume")

    plt.tight_layout()
    plt.savefig(os.path.join(filepath, f"training_results_{TIMESTAMP}.png"))
    plt.show()

    print(f"The best validation hypervolume is {max(hv_list)}")