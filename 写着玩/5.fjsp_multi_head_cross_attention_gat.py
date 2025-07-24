# ================================================
# 文件名：fjsp_cross_attention_gat.py
# 实验目标：在FJSP中评估不同特征提取结构（含GAT和交叉注意力）的调度性能
# 输出：CSV文件 + 对比图 makespan_compare.png


# 多头注意力
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

# ========== 环境 ========== #
class SimpleFJSPEnv:
    def __init__(self):
        self.jobs = [[(0, 3), (1, 2)], [(1, 2), (2, 4)], [(0, 1), (2, 3)]]
        self.reset()

    def reset(self):
        self.job_progress = [0, 0, 0]
        self.machine_status = [0, 0, 0]
        self.total_time = 0
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
        return self._get_state(), -1, done, {}

# ========== 模型模块 ========== #
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, X, adj):
        A_hat = adj + torch.eye(adj.size(0)).to(X.device)
        D_hat = torch.diag(torch.pow(A_hat.sum(1), -0.5))
        norm_adj = D_hat @ A_hat @ D_hat
        return F.relu(self.linear(norm_adj @ X))

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, X, adj):
        H = self.W(X)
        N = H.size(0)
        a_input = torch.cat([H.repeat(1, N).view(N * N, -1), H.repeat(N, 1)], dim=1)
        e = F.leaky_relu(a_input @ self.a).view(N, N)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        return F.relu(attention @ H)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return F.relu(out[:, :, :-self.conv.padding[0]])

class CrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, hidden_dim=128):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key_value):
        Q = self.query_proj(query).unsqueeze(1)      # [B, 1, D]
        K = self.key_proj(key_value)                 # [B, N, D]
        V = self.value_proj(key_value)               # [B, N, D]
        attn_weights = torch.softmax(Q @ K.transpose(1, 2) / np.sqrt(K.size(-1)), dim=-1)
        context = attn_weights @ V  # [B, 1, D]
        return self.out_proj(context.squeeze(1))
# ========== 多头 Cross-Attention 模块 ==========
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, query_dim, kv_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(kv_dim, hidden_dim)
        self.value_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key_value, return_attn=False):
        B = query.shape[0]
        Q = self.query_proj(query).view(B, self.num_heads, self.head_dim).transpose(0, 1)  # [H, B, D]
        K = self.key_proj(key_value).view(B, -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)  # [H, B, N, D]
        V = self.value_proj(key_value).view(B, -1, self.num_heads, self.head_dim).permute(2, 0, 1, 3)  # [H, B, N, D]

        attn_scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [H, B, 1, N]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [H, B, 1, N]
        context = torch.matmul(attn_weights, V).squeeze(2)  # [H, B, D]
        context = context.transpose(0, 1).contiguous().view(B, -1)  # [B, H*D]
        output = self.out_proj(context)  # [B, hidden_dim]

        if return_attn:
            return output, attn_weights
        return output

# ========== 特征提取器 ========== #
class FeatureExtractor(nn.Module):
    def __init__(self, use_gnn=True, use_tcn=True, fusion="concat", gnn_type="gcn"):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_tcn = use_tcn
        self.fusion = fusion
        self.hidden_dim = 64
        self.node_feat_dim = 4
        self.time_seq_len = 5

        if use_gnn:
            if gnn_type == "gcn":
                self.op_gnn = GCNLayer(self.node_feat_dim, self.hidden_dim)
                self.mac_gnn = GCNLayer(self.node_feat_dim, self.hidden_dim)
            elif gnn_type == "gat":
                self.op_gnn = GATLayer(self.node_feat_dim, self.hidden_dim)
                self.mac_gnn = GATLayer(self.node_feat_dim, self.hidden_dim)

        if use_tcn:
            self.tcn1 = TCNBlock(1, 32)
            self.tcn2 = TCNBlock(32, 64)

        # 预设一个占位，稍后根据实际输出维度来设置 output 层
        self.output = None

        # 占位 CrossAttention
        if use_gnn and use_tcn and fusion == "cross_attention":
            # self.cross_attn = CrossAttention(query_dim=64, kv_dim=self.hidden_dim)

            # 多头注意力
            self.cross_attn = MultiHeadCrossAttention(query_dim=64, kv_dim=self.hidden_dim, hidden_dim=128, num_heads=4)

        # 延后初始化 output 层（使用 dummy 输入试算）
        self._init_output_layer()

    def _init_output_layer(self):
        # 用假的输入测试一次前向，推理出输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 6)
            out = self.forward(dummy_input, dry_run=True)
            self.output = nn.Linear(out.shape[1], 128)

    def forward(self, state, dry_run=False):
        B = state.shape[0]

        if self.use_gnn:
            op_feat = torch.randn(3, self.node_feat_dim)
            op_adj = torch.tensor([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.float32)
            mac_feat = torch.randn(3, self.node_feat_dim)
            mac_adj = torch.ones(3, 3) - torch.eye(3)

            op_embed = self.op_gnn(op_feat, op_adj).mean(dim=0).unsqueeze(0).repeat(B, 1)
            mac_embed = self.mac_gnn(mac_feat, mac_adj).mean(dim=0).unsqueeze(0).repeat(B, 1)

        if self.use_tcn:
            tcn_seq = torch.randn(B, self.time_seq_len).unsqueeze(1)
            tcn_out = self.tcn2(self.tcn1(tcn_seq)).mean(dim=2)

        if self.use_gnn and self.use_tcn and self.fusion == "cross_attention":
            kv = torch.stack([op_embed, mac_embed], dim=1)
            out = self.cross_attn(tcn_out, kv)  # 注意：这里输出可能不是 64
        elif self.use_gnn and self.use_tcn:
            out = torch.cat([op_embed, mac_embed, tcn_out], dim=1)
        elif self.use_gnn:
            out = torch.cat([op_embed, mac_embed], dim=1)
        elif self.use_tcn:
            out = tcn_out
        else:
            out = state

        if dry_run:
            return out  # 不经过 Linear，只用于 shape 计算

        return self.output(out)



# ========== PPO智能体 ========== #
class PPOAgent:
    def __init__(self, state_dim, action_dim, **feat_cfg):
        self.feature = FeatureExtractor(**feat_cfg)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
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
        R, returns = 0, []
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

        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        log_probs = dist.log_prob(actions)
        advantages = returns - values.detach()

        ratio = torch.exp(log_probs - log_probs_old)
        loss = -torch.min(ratio * advantages, torch.clamp(ratio, 0.8, 1.2) * advantages).mean() + \
               F.mse_loss(values, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ========== 实验运行与绘图 ========== #
def run_experiment(name, use_gnn, use_tcn, fusion="concat", gnn_type="gcn", episodes=100):
    env = SimpleFJSPEnv()
    agent = PPOAgent(state_dim=6, action_dim=9, use_gnn=use_gnn, use_tcn=use_tcn, fusion=fusion, gnn_type=gnn_type)
    makespans = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        traj = []
        while not done:
            action_id, log_prob = agent.select_action(state)
            job_id, machine_id = action_id // 3, action_id % 3
            next_state, reward, done, _ = env.step((job_id, machine_id))
            traj.append((state, action_id, log_prob, reward))
            state = next_state
        states, actions, log_probs, rewards = zip(*traj)
        agent.update(states, actions, log_probs, rewards)
        makespans.append(env.total_time)

    df = pd.DataFrame({'Episode': range(episodes), 'Makespan': makespans})
    df.to_csv(f"results_{name}.csv", index=False)
    print(f"[{name}] 完成 ✅")
    return df

def run_all_models():
    configs = [
        ("MLP_Only",     False, False, "concat", "gcn"),
        ("GNN_Only",     True,  False, "concat", "gcn"),
        ("TCN_Only",     False, True,  "concat", "gcn"),
        ("GNN_TCN_Full", True,  True,  "concat", "gcn"),
        ("GNN_TCN_CA_GAT", True, True, "cross_attention", "gat"),
        ("GNN_TCN_MultiHead_CA_GAT", True, True, "cross_attention", "gat")  # ✅ 新增配置
    ]
    results = {}
    for name, gnn, tcn, fusion, gnn_type in configs:
        results[name] = run_experiment(name, gnn, tcn, fusion, gnn_type)
    return results

def plot_results(results):
    plt.figure(figsize=(9, 5))
    for name, df in results.items():
        plt.plot(df['Episode'], df['Makespan'], label=name)
    plt.xlabel("Episode")
    plt.ylabel("完工时间 Makespan")
    plt.title("不同特征结构下的调度性能对比")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("makespan_compare.png")
    plt.show()

# ========== 主程序 ========== #
if __name__ == "__main__":
    all_results = run_all_models()
    plot_results(all_results)
