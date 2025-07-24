# ================================================
# Brandimarte FJSP + GNN/TCN/MHA + PPO + 消融实验（完整脚本）
# 作者：你
# 说明：
#   - 支持 Brandimarte 数据集（机器编号从 1 开始、且可能不连续）
#   - 合法动作掩码，防止非法 job-machine 组合
#   - 多头 Cross-Attention 融合 GNN/TCN
#   - 多指标评估与消融实验

#Model,Makespan(mean),Flowtime(mean),AvgCompletion(mean),Utilization(mean)
#MLP_Only,7.4,2.6,0.26,0.44862745098039214
#GNN_Only,29.8,106.0,10.599999999999998,0.5425174667568176
#TCN_Only,14.4,41.6,4.16,0.47658730158730156
#GNN_TCN,21.8,55.4,5.54,0.517872814820086
#GNN_TCN_MHA,13.4,16.2,1.6200000000000003,0.5075396825396825


# ================================================

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
torch.set_num_threads(min(4, os.cpu_count() or 1))  # 防止CPU炸裂

# -----------------------------
# 1) Brandimarte 数据读取
# -----------------------------
def load_brandimarte(filepath):
    """
    兼容 Brandimarte .fjs 文件格式（行内：op_cnt nm (m t){nm} ...）
    返回：
        jobs: List[List[List[(m, t)]]]
    """
    jobs = []
    with open(filepath) as f:
        header = list(map(int, f.readline().strip().split()))
        if len(header) < 2:
            raise ValueError("首行至少应包含: <num_jobs> <num_machines>")
        J, M = header[:2]

        for _ in range(J):
            line = ''
            while line.strip() == '':
                line = f.readline()
            parts = list(map(int, line.strip().split()))
            op_cnt = parts[0]
            idx = 1
            ops = []
            for _ in range(op_cnt):
                nmach = parts[idx]; idx += 1
                machines = []
                for _ in range(nmach):
                    m = parts[idx]; t = parts[idx + 1]
                    machines.append((m, t))
                    idx += 2
                ops.append(machines)
            jobs.append(ops)
    return jobs  # 注意：不再返回 M（因为机器编号可能> M）


# -----------------------------
# 2) BrandEnv（机器编号重映射 + 合法动作）
# -----------------------------
class BrandEnv:
    def __init__(self, jobs):
        self.jobs = jobs
        self.J = len(jobs)
        # 真实机器编号集合（不从0开始、且不保证连续）
        all_machines = sorted({m for job in jobs for step in job for m, _ in step})
        self.machine_id2idx = {m: i for i, m in enumerate(all_machines)}
        self.idx2machine_id = {i: m for m, i in enumerate(all_machines)}
        self.M_real = len(all_machines)

        self.reset()

    def reset(self):
        self.job_progress = [0] * self.J
        self.machine_busy = [0] * self.M_real
        self.time = 0
        self.completion_time = [None] * self.J

        # 统计指标
        self.machine_busy_acc = np.zeros(self.M_real, dtype=np.int64)  # 机器忙碌累计时间

        return self._get_state()

    def _get_state(self):
        # 状态：机器剩余加工时间 + 作业当前进度
        return np.array(self.machine_busy + self.job_progress, dtype=np.float32)

    def get_valid_actions(self):
        """返回 当前时刻所有合法的 (job, m_external) 组合"""
        valid = []
        for j in range(self.J):
            step = self.job_progress[j]
            if step >= len(self.jobs[j]):  # 作业已完成
                continue
            for m_external, _ in self.jobs[j][step]:
                internal_m = self.machine_id2idx[m_external]
                if self.machine_busy[internal_m] == 0:
                    valid.append((j, m_external))
        return valid

    def step(self, action):
        """action = (j, m_external)"""
        j, m_ext = action
        step = self.job_progress[j]
        if step >= len(self.jobs[j]):
            # 非法：作业已完成
            return self._get_state(), -10.0, True, {}

        options = dict(self.jobs[j][step])
        if m_ext not in options:
            # 非法：该作业当前步不能选这个机器
            return self._get_state(), -5.0, False, {}

        duration = options[m_ext]
        m_int = self.machine_id2idx[m_ext]

        # 安排加工
        self.machine_busy[m_int] = duration
        self.job_progress[j] += 1

        # 时间推进 1 步（离散时间简化版）
        self.time += 1
        for k in range(self.M_real):
            if self.machine_busy[k] > 0:
                self.machine_busy[k] = max(0, self.machine_busy[k] - 1)
                self.machine_busy_acc[k] += 1

        # 记录完成时间
        for j2 in range(self.J):
            if self.job_progress[j2] == len(self.jobs[j2]) and self.completion_time[j2] is None:
                self.completion_time[j2] = self.time

        done = all(ct is not None for ct in self.completion_time)
        reward = -1.0  # 简单负奖励（鼓励更快完工）

        return self._get_state(), reward, done, {}

# -----------------------------
# 3) 模型模块
# -----------------------------
class GATLayer(nn.Module):
    """最简 GAT（单头）实现"""
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

    def forward(self, X, adj):
        """
        X: [N, F]
        adj: [N, N] 0/1
        """
        H = self.W(X)  # [N, out]
        N = H.size(0)
        a_input = torch.cat(
            [H.repeat(1, N).view(N * N, -1), H.repeat(N, 1)],
            dim=1
        )  # [N*N, 2*out]
        e = self.leaky_relu(a_input @ self.a).view(N, N)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        return F.relu(attention @ H)

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        out = out[:, :, :-self.conv.padding[0]]  # causal裁剪
        return F.relu(out)

class MultiHeadCrossAttention(nn.Module):
    """
    Query = TCN 表示（B, Dq）
    Key/Value = GNN 表示（B, N, Dk）
    """
    def __init__(self, q_dim, kv_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, kv, return_attn=False):
        """
        q:  [B, Dq]
        kv: [B, N, Dk]
        """
        B, N, _ = kv.shape

        Q = self.q_proj(q).view(B, self.num_heads, self.head_dim)                  # [B, H, D]
        K = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        V = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # [B, H, 1, D] @ [B, H, D, N] -> [B, H, 1, N]
        attn_scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, 1, N]

        context = torch.matmul(attn_weights, V).squeeze(2)  # [B, H, D]
        context = context.reshape(B, -1)                     # [B, H*D]

        out = self.out_proj(context)                         # [B, hidden_dim]
        if return_attn:
            return out, attn_weights
        return out

# -----------------------------
# 4) 特征提取器（可配：GNN/TCN/MHA）
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self,
                 input_dim,             # 原始state维度（当无GNN/TCN时使用）
                 use_gnn=True,
                 use_tcn=True,
                 use_mha=False,         # 是否使用多头cross-attention融合
                 gnn_node_feat_dim=4,
                 gnn_hidden=64,
                 tcn_hidden=64,
                 tcn_seq_len=5,
                 heads=4):
        super().__init__()
        self.use_gnn = use_gnn
        self.use_tcn = use_tcn
        self.use_mha = use_mha
        self.gnn_hidden = gnn_hidden
        self.tcn_hidden = tcn_hidden
        self.tcn_seq_len = tcn_seq_len

        # GNN（两张图：工序图/机器图；此处我们用“假特征+固定结构”占位，只为跑通消融）
        if self.use_gnn:
            self.op_gnn = GATLayer(gnn_node_feat_dim, gnn_hidden)
            self.mac_gnn = GATLayer(gnn_node_feat_dim, gnn_hidden)

        # TCN
        if self.use_tcn:
            self.tcn1 = TCNBlock(1, 32, 3, 1)
            self.tcn2 = TCNBlock(32, tcn_hidden, 3, 2)

        # Cross-Attention
        if self.use_gnn and self.use_tcn and self.use_mha:
            self.mha = MultiHeadCrossAttention(q_dim=tcn_hidden, kv_dim=gnn_hidden, hidden_dim=128, num_heads=heads)
            final_in = 128
        else:
            final_in = 0
            if self.use_gnn:
                final_in += gnn_hidden * 2
            if self.use_tcn:
                final_in += tcn_hidden
            if final_in == 0:
                final_in = input_dim   # 只用原始状态

        self.out = nn.Linear(final_in, 128)

    def forward(self, state):
        """
        state: [B, D_state]
        为了快速跑通，这里的 GNN 输入我们用“随机图 + 固定结构”的占位实现；
        若要严谨，请在这里把真实的 3 类图（工序图、机器图、析取图）按 batch 构造传进来。
        """
        B = state.size(0)
        feats = []

        if self.use_gnn:
            # “占位”图，真实实验请把以下改为你构造的图数据（可随 batch 动态生成）
            N_op, N_mac = 6, 6   # 这里只是随便定义个小图
            op_X  = torch.randn(N_op,  4, device=state.device)
            mac_X = torch.randn(N_mac, 4, device=state.device)
            op_adj  = torch.triu(torch.ones(N_op,  N_op,  device=state.device), diagonal=1)
            mac_adj = (torch.ones(N_mac, N_mac, device=state.device) - torch.eye(N_mac, device=state.device))

            op_emb  = self.op_gnn(op_X,  op_adj).mean(0, keepdim=True).repeat(B, 1)   # [B, gnn_hidden]
            mac_emb = self.mac_gnn(mac_X, mac_adj).mean(0, keepdim=True).repeat(B, 1) # [B, gnn_hidden]

        if self.use_tcn:
            # 这里同样用随机历史序列占位（真实工程应喂给近期机器状态历史）
            tcn_seq = torch.randn(B, 1, self.tcn_seq_len, device=state.device)  # [B, 1, T]
            tcn_out = self.tcn2(self.tcn1(tcn_seq)).mean(dim=2)  # [B, tcn_hidden]

        if self.use_gnn and self.use_tcn and self.use_mha:
            # 将 op/ mac 节点堆叠成 kv: [B, N, D]
            kv = torch.stack([op_emb, mac_emb], dim=1)  # [B, 2, gnn_hidden]
            out = self.mha(tcn_out, kv)                 # [B, 128]
        else:
            if self.use_gnn:
                feats.append(op_emb)
                feats.append(mac_emb)
            if self.use_tcn:
                feats.append(tcn_out)
            if len(feats) == 0:
                out = state
            else:
                out = torch.cat(feats, dim=1)

        return self.out(out)  # [B, 128]
# -----------------------------
# 5) PPO（带合法动作掩码）
# -----------------------------
class PPOAgent(nn.Module):
    def __init__(self,
                 state_dim,
                 act_space,          # List[(j, m_ext)]，动作空间（只包含真实可能出现的组合）
                 feature_cfg         # dict: use_gnn / use_tcn / use_mha ...
                 ):
        super().__init__()
        self.act_space = act_space
        self.act2idx = {a: i for i, a in enumerate(act_space)}
        self.idx2act = {i: a for a, i in self.act2idx.items()}
        self.act_dim = len(act_space)

        self.feature = FeatureExtractor(input_dim=state_dim, **feature_cfg)
        self.actor   = nn.Linear(128, self.act_dim)
        self.critic  = nn.Linear(128, 1)

        self.optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.gamma = 0.99
        self.clip_eps = 0.2

    def forward(self, state):
        emb = self.feature(state)
        logits = self.actor(emb)
        value  = self.critic(emb)
        return logits, value

    @torch.no_grad()
    def select_action(self, state, valid_actions):
        """
        state: np.array
        valid_actions: List[(j, m_ext)]
        """
        device = next(self.parameters()).device
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]
        logits, _ = self.forward(state_t)
        logits = logits.squeeze(0)  # [A]

        mask = torch.full_like(logits, float('-inf'))
        valid_idx = [self.act2idx[a] for a in valid_actions if a in self.act2idx]
        mask[valid_idx] = 0.0

        probs = F.softmax(logits + mask, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a_idx = dist.sample()
        logp  = dist.log_prob(a_idx)

        return a_idx.item(), logp.item()

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, batch):
        device = next(self.parameters()).device
        states = torch.tensor(np.array(batch['states']), dtype=torch.float32, device=device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        old_logp = torch.tensor(batch['logps'], dtype=torch.float32, device=device)
        rewards  = batch['rewards']

        returns = self.compute_returns(rewards).to(device)

        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(F.softmax(logits, dim=-1))
        logp = dist.log_prob(actions)
        advantages = returns - values.squeeze().detach()

        ratio = torch.exp(logp - old_logp)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages

        loss_policy = -torch.min(surr1, surr2).mean()
        loss_value  = F.mse_loss(values.squeeze(), returns)
        loss = loss_policy + 0.5 * loss_value

        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optim.step()
# -----------------------------
# 6) 训练 & 评估（多指标 & 消融）
# -----------------------------
def build_act_space(jobs):
    """把所有 (job, machine_external_id) 组合去重，作为固定动作空间"""
    act_space = []
    seen = set()
    for j, job in enumerate(jobs):
        for step in job:
            for m, _ in step:
                key = (j, m)
                if key not in seen:
                    seen.add(key)
                    act_space.append(key)
    return act_space

def run_one_experiment(
        name,
        jobs,
        episodes=100,
        feature_cfg=None,
        seed=0
    ):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = BrandEnv(jobs)
    state_dim = env.M_real + env.J
    act_space = build_act_space(jobs)

    agent = PPOAgent(state_dim, act_space, feature_cfg)
    device = torch.device('cpu')
    agent.to(device)

    # 记录
    ep_records = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        batch = defaultdict(list)

        # 一个 ep 中的单轮交互
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                # 无动作可选，强制结束
                break

            a_idx, logp = agent.select_action(s, valid_actions)
            j, m_ext = agent.idx2act[a_idx]
            s2, r, done, _ = env.step((j, m_ext))

            batch['states'].append(s)
            batch['actions'].append(a_idx)
            batch['logps'].append(logp)
            batch['rewards'].append(r)

            s = s2

        # 更新 PPO
        if len(batch['states']) > 0:
            agent.update(batch)

        # 多指标
        makespan = env.time
        flowtime = sum(t for t in env.completion_time if t is not None)
        avg_completion = flowtime / env.J
        utilization = env.machine_busy_acc.sum() / (env.M_real * makespan) if makespan > 0 else 0.0

        ep_records.append({
            'Episode': ep,
            'Makespan': makespan,
            'Flowtime': flowtime,
            'AvgCompletion': avg_completion,
            'Utilization': utilization
        })

        if (ep + 1) % max(1, (episodes // 5)) == 0:
            print(f"[{name}] Ep {ep+1}/{episodes} | Mk={makespan:.1f} | Flow={flowtime:.1f} | Util={utilization:.3f}")

    df = pd.DataFrame(ep_records)
    df.to_csv(f"{name}.csv", index=False)
    return df

def run_all(jobs, episodes=100, seed=0):
    """
    五组消融：
    1) MLP_Only
    2) GNN_Only
    3) TCN_Only
    4) GNN_TCN
    5) GNN_TCN_MHA
    """
    cfgs = [
        ("MLP_Only",        dict(use_gnn=False, use_tcn=False, use_mha=False)),
        ("GNN_Only",        dict(use_gnn=True,  use_tcn=False, use_mha=False)),
        ("TCN_Only",        dict(use_gnn=False, use_tcn=True,  use_mha=False)),
        ("GNN_TCN",         dict(use_gnn=True,  use_tcn=True,  use_mha=False)),
        ("GNN_TCN_MHA",     dict(use_gnn=True,  use_tcn=True,  use_mha=True)),
    ]

    results = {}
    for name, feature_cfg in cfgs:
        print(f"\n========== 运行 {name} ==========")
        df = run_one_experiment(name, jobs, episodes=episodes, feature_cfg=feature_cfg, seed=seed)
        results[name] = df

    # 画图（以 Makespan 为例）
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['Episode'], df['Makespan'], label=name)
    plt.xlabel("Episode")
    plt.ylabel("Makespan")
    plt.title("不同特征组合的调度性能对比（Makespan）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_makespan.png")
    plt.show()

    # 画 AvgCompletion
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['Episode'], df['AvgCompletion'], label=name)
    plt.xlabel("Episode")
    plt.ylabel("AvgCompletion")
    plt.title("不同特征组合的调度性能对比（AvgCompletion）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_avg_completion.png")
    plt.show()

    # 画 Utilization
    plt.figure(figsize=(10, 6))
    for name, df in results.items():
        plt.plot(df['Episode'], df['Utilization'], label=name)
    plt.xlabel("Episode")
    plt.ylabel("Utilization")
    plt.title("不同特征组合的调度性能对比（Utilization）")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_utilization.png")
    plt.show()

    return results

# -----------------------------
# 7) 入口
# -----------------------------
if __name__ == "__main__":
    # 1) 修改为你的 Brandimarte 文件名（例如 MK01）
    data_file = "Brandimarte_MK01.fjs"

    # 2) 读取数据
    jobs = load_brandimarte(data_file)

    # 3) 运行消融实验
    results = run_all(jobs, episodes=10, seed=0)

    # 4) 输出每组实验最后5个 episode 的均值，以便论文表格展示
    summary = []
    for name, df in results.items():
        tail = df.tail(5)
        summary.append({
            "Model": name,
            "Makespan(mean)": tail["Makespan"].mean(),
            "Flowtime(mean)": tail["Flowtime"].mean(),
            "AvgCompletion(mean)": tail["AvgCompletion"].mean(),
            "Utilization(mean)": tail["Utilization"].mean()
        })
    summary_df = pd.DataFrame(summary)
    print("\n==== 末5个Episode均值（可直接用于论文表格） ====")
    print(summary_df.round(3))
    summary_df.to_csv("ablation_summary_last5.csv", index=False)
