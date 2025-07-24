# fjsp_dqn_solver.py

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== 问题定义：3个作业，每个作业3道工序 ======
jobs = [
    [[(0, 3), (1, 2)], [(1, 2), (2, 4)], [(0, 5), (2, 3)]],
    [[(1, 3), (2, 2)], [(0, 4), (2, 5)], [(1, 2), (2, 3)]],
    [[(0, 2), (2, 1)], [(0, 3), (1, 2)], [(1, 4), (2, 2)]]
]
num_jobs = len(jobs)
num_machines = 3

# ====== 简单环境 ======
class SimpleFJSPEnv:
    def __init__(self, jobs):
        self.jobs = jobs
        self.reset()

    def reset(self):
        self.job_step = [0] * len(self.jobs)
        self.machine_time = [0] * num_machines
        self.done = False
        return self._get_state()

    def _get_state(self):
        return np.array(self.job_step + self.machine_time, dtype=np.float32)

    def sample_action(self):
        # 返回合法动作：(job_id, machine_id)
        actions = []
        for job_id, step in enumerate(self.job_step):
            if step < len(self.jobs[job_id]):
                for m_id, _ in self.jobs[job_id][step]:
                    actions.append((job_id, m_id))
        return random.choice(actions) if actions else None

    def step(self, action):
        job_id, machine_id = action
        step = self.job_step[job_id]
        candidates = self.jobs[job_id][step]
        time = [t for m, t in candidates if m == machine_id][0]
        start_time = self.machine_time[machine_id]
        end_time = start_time + time
        self.machine_time[machine_id] = end_time
        self.job_step[job_id] += 1
        self.done = all(s == len(self.jobs[i]) for i, s in enumerate(self.job_step))
        reward = -max(self.machine_time) if self.done else 0
        return self._get_state(), reward, self.done

# ====== DQN网络 ======
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ====== 动作映射 ======
def get_all_actions(jobs):
    actions = []
    for job_id, job in enumerate(jobs):
        for step in job:
            for machine_id, _ in step:
                if (job_id, machine_id) not in actions:
                    actions.append((job_id, machine_id))
    return actions

# ====== 主训练 ======
env = SimpleFJSPEnv(jobs)
actions = get_all_actions(jobs)
state_dim = len(env._get_state())
action_dim = len(actions)
action_to_idx = {a: i for i, a in enumerate(actions)}
idx_to_action = {i: a for a, i in action_to_idx.items()}

model = DQN(state_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
replay = []
max_steps = 200
epsilon = 0.2
batch_size = 32

for episode in range(100):
    state = env.reset()
    total_reward = 0
    for _ in range(max_steps):
        if random.random() < epsilon:
            action = env.sample_action()
        else:
            with torch.no_grad():
                q = model(torch.tensor(state).to(device))
            mask = np.zeros(action_dim)
            legal_actions = []
            for a in actions:
                j, m = a
                if env.job_step[j] < len(jobs[j]):
                    steps = jobs[j][env.job_step[j]]
                    if any(m == m_ for m_, _ in steps):
                        legal_actions.append(a)
            if not legal_actions:
                break
            best_a = max(legal_actions, key=lambda a: q[action_to_idx[a]].item())
            action = best_a

        next_state, reward, done = env.step(action)
        replay.append((state, action_to_idx[action], reward, next_state, done))
        if len(replay) > 500:
            replay.pop(0)
        state = next_state
        total_reward += reward
        if done:
            break

        # 训练
        if len(replay) >= batch_size:
            batch = random.sample(replay, batch_size)
            s, a, r, ns, d = zip(*batch)
            s = torch.tensor(s).to(device)
            a = torch.tensor(a).to(device)
            r = torch.tensor(r).to(device)
            ns = torch.tensor(ns).to(device)
            d = torch.tensor(d).to(device)

            q_values = model(s)
            next_q = model(ns).detach().max(1)[0]
            targets = r + (1 - d.float()) * 0.99 * next_q
            predictions = q_values.gather(1, a.unsqueeze(1)).squeeze()
            loss = loss_fn(predictions, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Episode {episode}: Reward = {total_reward}")

# ====== 输出调度结果 ======
print("\n=== 最终调度方案 ===")
state = env.reset()
while not env.done:
    with torch.no_grad():
        q = model(torch.tensor(state).float().to(device))
    legal_actions = []
    for a in actions:
        j, m = a
        if env.job_step[j] < len(jobs[j]):
            steps = jobs[j][env.job_step[j]]
            if any(m == m_ for m_, _ in steps):
                legal_actions.append(a)
    best_a = max(legal_actions, key=lambda a: q[action_to_idx[a]].item())
    print(f"调度: 作业{best_a[0]} → 机器{best_a[1]}")
    state, _, _ = env.step(best_a)
