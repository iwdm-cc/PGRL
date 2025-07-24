# ================================================
# 文件名：3.fjsp_ablation_experiment.py
# 实验目标：比较不同特征提取结构（GNN / TCN / 全部）在调度问题上的性能表现
# 输出：CSV文件 + 对比图 makespan_compare.png
# ================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 简化调度环境 ==========
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

# ========== 2. 模块：GCN Layer ==========
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, adj):
        A_hat = adj + torch.eye(adj.size(0)).to(X.device)
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        norm_adj = torch.matmul(torch.matmul(D_hat, A_hat), D_hat)
        X = self.linear(torch.matmul(norm_adj, X))
        return F.relu(X)

# ========== 3. 模块：TCN ==========
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]
        return self.relu(out)

# ========== 4. 模块：特征提取器（消融控制） ==========
class FeatureExtractor(nn.Module):
    def __init__(self, use_gnn=True, use_tcn=True):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_tcn = use_tcn

        self.hidden_dim = 64
        self.node_feat_dim = 4
        self.time_seq_len = 5

        if use_gnn:
            self.op_gcn = GCNLayer(self.node_feat_dim, self.hidden_dim)
            self.mac_gcn = GCNLayer(self.node_feat_dim, self.hidden_dim)

        if use_tcn:
            self.tcn1 = TCNBlock(1, 32, 3, 1)
            self.tcn2 = TCNBlock(32, 64, 3, 2)

        in_dim = 0
        if use_gnn:
            in_dim += self.hidden_dim * 2
        if use_tcn:
            in_dim += 64
        if in_dim == 0:
            in_dim = 6  # fallback：使用原始状态向量

        self.fusion = nn.Linear(in_dim, 128)

    def forward(self, state):
        B = state.shape[0]
        features = []

        if self.use_gnn:
            # 构造图结构特征，只做一次，重复B次
            op_feat = torch.randn(3, self.node_feat_dim)
            op_adj = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
            mac_feat = torch.randn(3, self.node_feat_dim)
            mac_adj = torch.ones(3, 3) - torch.eye(3)

            op_embed = self.op_gcn(op_feat, op_adj).mean(dim=0).unsqueeze(0).repeat(B, 1)
            mac_embed = self.mac_gcn(mac_feat, mac_adj).mean(dim=0).unsqueeze(0).repeat(B, 1)
            features.append(op_embed)
            features.append(mac_embed)

        if self.use_tcn:
            tcn_seq = torch.randn(B, self.time_seq_len).unsqueeze(1)  # [B, 1, T]
            tcn_out = self.tcn1(tcn_seq)
            tcn_out = self.tcn2(tcn_out)
            tcn_embed = tcn_out.mean(dim=2)  # [B, hidden]
            features.append(tcn_embed)

        if not features:
            return self.fusion(state)
        else:
            cat = torch.cat(features, dim=-1)  # [B, total_dim]
            return self.fusion(cat)


# ========== 5. PPO 智能体 ==========
class PPOAgent:
    def __init__(self, state_dim, action_dim, use_gnn=True, use_tcn=True, lr=1e-3):
        self.feature = FeatureExtractor(use_gnn, use_tcn)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.gamma = 0.99

    def parameters(self):
        return list(self.feature.parameters()) + list(self.actor.parameters()) + list(self.critic.parameters())

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        emb = self.feature(state)
        logits = self.actor(emb)
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
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions)
        log_probs_old = torch.stack(log_probs_old)
        returns = self.compute_returns(rewards)

        emb = self.feature(states)
        logits = self.actor(emb)
        values = self.critic(emb).squeeze()

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        advantages = returns - values.detach()

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
        loss = -torch.min(surr1, surr2).mean() + F.mse_loss(values, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ========== 6. 主训练函数 ==========
def run_experiment(model_name, use_gnn, use_tcn, episodes=100):
    env = SimpleFJSPEnv()
    agent = PPOAgent(state_dim=6, action_dim=9, use_gnn=use_gnn, use_tcn=use_tcn)
    all_makespans = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        states, actions, rewards, log_probs = [], [], [], []

        while not done:
            action_id, log_prob = agent.select_action(state)
            job_id, machine_id = action_id // 3, action_id % 3
            next_state, reward, done, _ = env.step((job_id, machine_id))

            states.append(state)
            actions.append(action_id)
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state

        agent.update(states, actions, log_probs, rewards)
        all_makespans.append(env.total_time)

    # 保存结果
    df = pd.DataFrame({'Episode': np.arange(episodes), 'Makespan': all_makespans})
    csv_path = f"results_{model_name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[{model_name}] 完成，结果保存至：{csv_path}")
    return df

# ========== 7. 批量运行所有配置 ==========
def run_all_models():
    configs = [
        ("MLP_Only",     False, False),
        ("GNN_Only",     True,  False),
        ("TCN_Only",     False, True),
        ("GNN_TCN_Full", True,  True),
    ]

    results = {}
    for name, use_gnn, use_tcn in configs:
        df = run_experiment(name, use_gnn, use_tcn)
        results[name] = df

    return results

# ========== 8. 绘图与结果对比 ==========
def plot_results(results):
    plt.figure(figsize=(8, 5))
    for name, df in results.items():
        plt.plot(df['Episode'], df['Makespan'], label=name)
    plt.xlabel("Episode")
    plt.ylabel("完工时间 Makespan")
    plt.title("不同特征提取结构下的调度表现对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("makespan_compare.png")
    plt.show()

# ========== 主程序入口 ==========
if __name__ == "__main__":
    all_results = run_all_models()
    plot_results(all_results)
