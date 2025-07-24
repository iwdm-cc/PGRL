
# ================================================
# 文件名：fjsp_brandimarte_machine_mapped.py
# 功能：合法动作掩码 + 机器编号映射 + Brandimarte 数据支持
# ================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ---------- 数据读取 ----------
def load_brandimarte(filepath):
    jobs = []
    with open(filepath) as f:
        header = list(map(int, f.readline().strip().split()))
        J, M = header[:2]
        for j in range(J):
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
                    m = parts[idx]; t = parts[idx+1]
                    machines.append((m, t))
                    idx += 2
                ops.append(machines)
            jobs.append(ops)
    return jobs

# ---------- 环境 ----------
class BrandEnv:
    def __init__(self, jobs):
        self.jobs = jobs
        self.J = len(jobs)
        # 获取所有真实机器编号并映射
        all_machines = sorted({m for job in jobs for step in job for m, _ in step})
        self.machine_id2idx = {m: i for i, m in enumerate(all_machines)}
        self.idx2machine_id = {i: m for m, i in enumerate(all_machines)}
        self.M_real = len(all_machines)
        self.reset()

    def reset(self):
        self.job_progress = [0] * self.J
        self.machine_busy = [0] * self.M_real
        self.completion_time = [None] * self.J
        self.time = 0
        return self._get_state()

    def _get_state(self):
        return np.array(self.machine_busy + self.job_progress, dtype=np.float32)

    def step(self, action):
        j, m = action
        step = self.job_progress[j]
        if step >= len(self.jobs[j]):
            return self._get_state(), -10, True, {}

        options = dict(self.jobs[j][step])
        if m not in options:
            return self._get_state(), -5, False, {}

        duration = options[m]
        internal_m = self.machine_id2idx[m]
        self.machine_busy[internal_m] = duration
        self.job_progress[j] += 1
        self.time += 1
        self.machine_busy = [max(0, t-1) for t in self.machine_busy]

        for j in range(self.J):
            if self.job_progress[j] == len(self.jobs[j]) and self.completion_time[j] is None:
                self.completion_time[j] = self.time

        done = all(ct is not None for ct in self.completion_time)
        return self._get_state(), -1, done, {}

    def get_valid_actions(self):
        valid = []
        for j in range(self.J):
            step = self.job_progress[j]
            if step >= len(self.jobs[j]):
                continue
            options = self.jobs[j][step]
            for m, _ in options:
                valid.append((j, m))
        return valid

# ---------- Agent ----------
class MLPAgent(nn.Module):
    def __init__(self, input_dim, act_count):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, act_count)
        )

    def forward(self, x):
        return self.fc(x)

    def select_valid(self, state, valid_indices):
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(x).squeeze()
        mask = torch.full_like(logits, float('-inf'))
        mask[valid_indices] = 0.0
        probs = F.softmax(logits + mask, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        return a.item(), dist.log_prob(a)

# ---------- Main ----------
def run(filepath, episodes=100):
    jobs = load_brandimarte(filepath)
    env = BrandEnv(jobs)
    input_dim = env.M_real + len(jobs)

    # 动态构建动作空间与映射
    act_space = []
    act2idx = {}
    for j, job in enumerate(jobs):
        for step in job:
            for m, _ in step:
                if (j, m) not in act2idx:
                    act2idx[(j, m)] = len(act_space)
                    act_space.append((j, m))

    agent = MLPAgent(input_dim=input_dim, act_count=len(act_space))
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    all_makespans = []

    for ep in range(episodes):
        s = env.reset()
        done = False
        traj = []

        while not done:
            valid_acts = env.get_valid_actions()
            valid_indices = [act2idx[(j, m)] for j, m in valid_acts if (j, m) in act2idx]
            if not valid_indices:
                break
            a_idx, logp = agent.select_valid(s, valid_indices)
            j, m = act_space[a_idx]
            s2, r, done, _ = env.step((j, m))
            traj.append((s, a_idx, logp, r))
            s = s2

        # 更新策略
        R = 0
        returns = []
        for _, _, _, r in reversed(traj):
            R = r + 0.99 * R
            returns.insert(0, R)

        if len(traj) == 0:
            continue

        states = torch.tensor([x[0] for x in traj], dtype=torch.float32)
        actions = torch.tensor([x[1] for x in traj])
        log_probs_old = torch.stack([x[2] for x in traj])
        returns = torch.tensor(returns)

        logits = agent(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        loss = - (log_probs * (returns - returns.mean())).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_makespans.append(env.time)

    df = pd.DataFrame({'Episode': np.arange(len(all_makespans)), 'Makespan': all_makespans})
    df.to_csv("result_machine_mapped.csv", index=False)
    plt.plot(df['Episode'], df['Makespan'])
    plt.title("掩码训练 - 机器编号映射修复版")
    plt.xlabel("Episode")
    plt.ylabel("Makespan")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("makespan_machine_mapped.png")
    plt.show()
    print("✅ 训练完成，结果保存为 result_machine_mapped.csv")
    return df

if __name__ == "__main__":
    run("Brandimarte_MK01.fjs", episodes=100)

