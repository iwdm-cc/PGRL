# ==========================================
# 文件名：ppo_fjsp_toy_experiment.py
# 说明：
# 本文件用于在简化的柔性作业车间调度（FJSP）环境中，
# 使用基于 PyTorch 实现的 PPO 算法进行训练。
# 环境为 Toy 版本，包含3个作业、3台机器，每个作业包含2道工序。
# 模型为简单的全连接神经网络，输出为动作策略和值函数。
# 实验输出训练曲线图和CSV结果文件，便于后续对比和可视化。
# ==========================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决某些平台 PyTorch 启动冲突

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# 设置中文字体和负号正常显示（适配Windows中文显示）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 一、调度环境（简化 FJSP 环境）
# -------------------------------
class SimpleFJSPEnv:
    def __init__(self):
        self.jobs = [
            [(0, 3), (1, 2)],
            [(1, 2), (2, 4)],
            [(0, 1), (2, 3)]
        ]
        self.reset()

    def reset(self):
        self.job_progress = [0, 0, 0]
        self.machine_status = [0, 0, 0]
        self.total_time = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array(self.machine_status + self.job_progress, dtype=np.float32)

    def step(self, action):
        job_id, machine_id = action
        step = self.job_progress[job_id]
        if step >= len(self.jobs[job_id]):
            return self._get_state(), -10, True, {}

        _, duration = self.jobs[job_id][step]
        self.machine_status[machine_id] = duration
        self.job_progress[job_id] += 1

        self.total_time += 1
        self.machine_status = [max(0, t - 1) for t in self.machine_status]

        done = all(p >= 2 for p in self.job_progress)
        reward = -1
        return self._get_state(), reward, done, {}

# -------------------------------
# 二、策略网络与特征提取器
# -------------------------------
class SimpleFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super().__init__()
        self.feature = SimpleFeatureExtractor(input_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.feature(x)
        return self.actor(x), self.critic(x)

# -------------------------------
# 三、PPO 智能体
# -------------------------------
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.model = PolicyNetwork(state_dim, 64, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, _ = self.model(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, log_probs_old, rewards):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions)
        log_probs_old = torch.stack(log_probs_old)
        returns = self.compute_returns(rewards)

        logits, values = self.model(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        advantages = returns - values.squeeze()
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values.squeeze(), returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# -------------------------------
# 四、训练与评估流程
# -------------------------------
def run_experiment(episodes=100):
    env = SimpleFJSPEnv()
    agent = PPOAgent(state_dim=6, action_dim=9)  # 3作业×3机器 = 9个动作组合

    all_rewards = []
    all_makespans = []

    for ep in range(episodes):
        state = env.reset()
        states, actions, rewards, log_probs = [], [], [], []
        done = False

        while not done:
            action_id, log_prob = agent.select_action(state)
            job_id = action_id // 3
            machine_id = action_id % 3
            next_state, reward, done, _ = env.step((job_id, machine_id))

            states.append(state)
            actions.append(action_id)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        agent.update(states, actions, log_probs, rewards)
        all_rewards.append(sum(rewards))
        all_makespans.append(env.total_time)

    # 输出CSV文件
    result_df = pd.DataFrame({
        'Episode': np.arange(episodes),
        'TotalReward': all_rewards,
        'Makespan': all_makespans
    })
    result_df.to_csv("ppo_fjsp_results.csv", index=False)

    # 绘制训练曲线图
    plt.figure()
    plt.plot(all_makespans, label="完工时间 Makespan")
    plt.xlabel("训练轮次 Episode")
    plt.ylabel("完工时间 Makespan")
    plt.title("PPO 训练调度策略过程中的完工时间变化曲线")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("ppo_makespan_curve.png")
    plt.show()

    return result_df

# -------------------------------
# 主函数入口
# -------------------------------
if __name__ == "__main__":
    run_experiment(episodes=100)
