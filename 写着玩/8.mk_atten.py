# ================================================
# Brandimarte FJSP + GNN/TCN/MHA + PPO + 消融实验（完整脚本）
# 作者：你
# 说明：
#   - 支持 Brandimarte 数据集（机器编号从 1 开始、且可能不连续）
#   - 合法动作掩码，防止非法 job-machine 组合
#   - 多头 Cross-Attention 融合 GNN/TCN
#   - 多指标评估与消融实验
# 修改：
#   - 添加详细中文注释
#   - 改进特征提取器，使用真实环境信息替代随机图
#   - 修复时间推进逻辑，支持连续时间
#   - 优化状态表示，添加更多调度信息
# ================================================

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import heapq
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
torch.set_num_threads(min(4, os.cpu_count() or 1))  # 设置线程数，防止CPU过载


# -----------------------------
# 1) Brandimarte 数据读取
# -----------------------------
def load_brandimarte(filepath):
    """
    加载Brandimarte格式的FJSP数据集
    参数:
        filepath: 文件路径
    返回:
        jobs: 作业列表，结构为 List[List[List[(m, t)]]]
        其中:
          - 外层List: 所有作业
          - 中层List: 作业的工序序列
          - 内层List: 工序可选机器列表，每个元素为(机器ID, 加工时间)
    """
    jobs = []
    with open(filepath) as f:
        # 读取文件头（作业数和机器数）
        header = list(map(int, f.readline().strip().split()))
        if len(header) < 2:
            raise ValueError("首行至少应包含: <num_jobs> <num_machines>")
        J, M = header[:2]  # J: 作业数量, M: 机器数量

        # 读取每个作业的信息
        for _ in range(J):
            line = ''
            # 跳过空行
            while line.strip() == '':
                line = f.readline()
            parts = list(map(int, line.strip().split()))
            op_cnt = parts[0]  # 该作业的工序数量
            idx = 1
            ops = []  # 存储该作业的所有工序

            # 读取每个工序的信息
            for _ in range(op_cnt):
                nmach = parts[idx]  # 该工序可选机器数量
                idx += 1
                machines = []  # 存储该工序的可选机器列表

                # 读取每台可选机器的信息
                for _ in range(nmach):
                    m = parts[idx]  # 机器ID
                    t = parts[idx + 1]  # 加工时间
                    machines.append((m, t))
                    idx += 2
                ops.append(machines)
            jobs.append(ops)
    return jobs


# -----------------------------
# 2) BrandEnv（机器编号重映射 + 合法动作）
# -----------------------------
class BrandEnv:
    def __init__(self, jobs):
        """
        柔性作业车间调度环境
        参数:
            jobs: 作业列表，格式同load_brandimarte的返回值
        """
        self.jobs = jobs
        self.J = len(jobs)  # 作业数量

        # 获取所有真实机器ID（可能不连续）
        all_machines = sorted({m for job in jobs for step in job for m, _ in step})
        self.machine_id2idx = {m: i for i, m in enumerate(all_machines)}  # 机器ID到索引的映射
        self.idx2machine_id = {i: m for m, i in self.machine_id2idx.items()}  # 索引到机器ID的映射
        self.M_real = len(all_machines)  # 实际机器数量

        # 动作空间预计算
        self.act_space = self._build_act_space()

        self.reset()

    def _build_act_space(self):
        """构建动作空间：所有可能的(job, machine_external_id)组合"""
        act_space = []
        for j in range(self.J):
            for step_idx, step in enumerate(self.jobs[j]):
                for m, _ in step:
                    act_space.append((j, step_idx, m))  # (作业ID, 工序ID, 机器ID)
        return act_space

    def reset(self):
        """重置环境状态"""
        # 作业状态
        self.job_progress = [0] * self.J  # 每个作业的当前工序索引
        self.job_step_status = [{} for _ in range(self.J)]  # 每个工序的状态: None(未开始), 'processing', 'finished'
        self.completion_time = [None] * self.J  # 每个作业的完成时间

        # 机器状态
        self.machine_busy_until = [0] * self.M_real  # 每台机器的空闲时间点
        self.machine_schedule = [[] for _ in range(self.M_real)]  # 每台机器的调度计划

        # 全局状态
        self.current_time = 0  # 当前模拟时间
        self.event_queue = []  # 事件队列（完成事件）

        # 统计指标
        self.machine_busy_time = [0] * self.M_real  # 每台机器的总加工时间
        self.total_processing_time = 0  # 总加工时间

        # 工序状态
        self.op_start_times = [[None] * len(job) for job in self.jobs]  # 工序开始时间
        self.op_end_times = [[None] * len(job) for job in self.jobs]  # 工序结束时间

        # 初始化作业状态
        for j in range(self.J):
            for step_idx in range(len(self.jobs[j])):
                self.job_step_status[j][step_idx] = None

        return self._get_state()

    def _get_state(self):
        """
        获取环境状态
        状态向量包括:
          - 每台机器的空闲时间点
          - 每个作业的当前工序索引
          - 每个作业的当前工序状态
        """
        # 机器空闲时间（归一化）
        machine_state = [self.machine_busy_until[i] for i in range(self.M_real)]
        max_time = max(max(machine_state), 1)  # 确保 max_time 至少为 1
        machine_state = [t / max_time for t in machine_state]

        # 作业状态
        job_progress = [self.job_progress[j] / len(self.jobs[j]) for j in range(self.J)]  # 进度百分比
        job_current_step = [self.job_progress[j] for j in range(self.J)]  # 当前工序索引

        # 合并状态
        state = np.array(machine_state + job_progress + job_current_step, dtype=np.float32)
        return state

    def get_valid_actions(self):
        """返回当前时刻所有合法的 (job, step_idx, machine_external_id) 组合"""
        valid = []
        for j in range(self.J):
            step_idx = self.job_progress[j]
            # 检查作业是否已完成
            if step_idx >= len(self.jobs[j]):
                continue

            # 检查工序是否已分配
            if self.job_step_status[j][step_idx] is not None:
                continue

            # 获取可选机器
            for m_ext, t in self.jobs[j][step_idx]:
                m_idx = self.machine_id2idx[m_ext]
                # 检查机器是否可用（当前空闲）
                if self.current_time >= self.machine_busy_until[m_idx]:
                    valid.append((j, step_idx, m_ext))
        return valid

    def step(self, action):
        """
        执行一个动作
        参数:
            action: (job_id, step_idx, machine_external_id)
        返回:
            next_state, reward, done, info
        """
        j, step_idx, m_ext = action

        # 验证动作合法性
        if step_idx != self.job_progress[j]:
            return self._get_state(), -10.0, False, {"error": "Invalid step index"}

        if self.job_step_status[j][step_idx] is not None:
            return self._get_state(), -10.0, False, {"error": "Step already processed"}

        # 获取加工时间
        options = dict(self.jobs[j][step_idx])
        if m_ext not in options:
            return self._get_state(), -5.0, False, {"error": "Machine not available for this step"}

        duration = options[m_ext]
        m_idx = self.machine_id2idx[m_ext]

        # 计算开始时间（机器空闲后才能开始）
        start_time = max(self.current_time, self.machine_busy_until[m_idx])
        end_time = start_time + duration

        # 更新机器状态
        self.machine_busy_until[m_idx] = end_time
        self.machine_busy_time[m_idx] += duration
        self.machine_schedule[m_idx].append((j, step_idx, start_time, end_time))

        # 更新工序状态
        self.op_start_times[j][step_idx] = start_time
        self.op_end_times[j][step_idx] = end_time
        self.job_step_status[j][step_idx] = 'processing'

        # 添加完成事件到队列
        heapq.heappush(self.event_queue, (end_time, j, step_idx))

        # 更新总加工时间
        self.total_processing_time += duration

        # 推进时间到下一个事件点
        if self.event_queue:
            next_event_time = self.event_queue[0][0]
            self.current_time = next_event_time
        else:
            # 没有事件时推进到最近机器的空闲时间
            self.current_time = min(self.machine_busy_until) if self.machine_busy_until else self.current_time + 1

        # 处理已完成的事件
        while self.event_queue and self.event_queue[0][0] <= self.current_time:
            event_time, j_done, step_idx_done = heapq.heappop(self.event_queue)
            # 标记工序完成
            self.job_step_status[j_done][step_idx_done] = 'finished'
            # 如果这是作业的最后一道工序，标记作业完成
            if step_idx_done == len(self.jobs[j_done]) - 1:
                self.completion_time[j_done] = event_time
            # 推进作业进度
            if j_done == j and step_idx_done == step_idx:
                self.job_progress[j] += 1

        # 检查是否所有作业都已完成
        done = all(ct is not None for ct in self.completion_time)

        # 计算奖励（鼓励缩短完工时间）
        reward = -0.1 * duration  # 负奖励与加工时间成比例

        return self._get_state(), reward, done, {}

    def get_schedule(self):
        """获取当前调度方案"""
        schedule = []
        for m_idx in range(self.M_real):
            machine_id = self.idx2machine_id[m_idx]
            for j, step_idx, start, end in self.machine_schedule[m_idx]:
                schedule.append({
                    'machine': machine_id,
                    'job': j,
                    'step': step_idx,
                    'start': start,
                    'end': end,
                    'duration': end - start
                })
        return pd.DataFrame(schedule)

    def get_performance_metrics(self):
        """计算性能指标"""
        if not any(ct is not None for ct in self.completion_time):
            return {}

        makespan = max(ct for ct in self.completion_time if ct is not None)
        flowtime = sum(ct for ct in self.completion_time if ct is not None)
        avg_completion = flowtime / self.J
        utilization = sum(self.machine_busy_time) / (makespan * self.M_real) if makespan > 0 else 0.0

        return {
            'makespan': makespan,
            'flowtime': flowtime,
            'avg_completion': avg_completion,
            'utilization': utilization
        }


# -----------------------------
# 3) 模型模块
# -----------------------------
class GATLayer(nn.Module):
    """图注意力网络层（单头）"""

    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)  # 特征变换矩阵
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))  # 注意力系数向量
        nn.init.xavier_uniform_(self.a.data, gain=1.414)  # 初始化参数
        self.leaky_relu = nn.LeakyReLU(negative_slope)  # 激活函数

    def forward(self, X, adj):
        """
        前向传播
        参数:
            X: 节点特征矩阵 [N, F]
            adj: 邻接矩阵 [N, N] (0/1表示连接关系)
        返回:
            更新后的节点表示 [N, out_dim]
        """
        H = self.W(X)  # [N, out_dim]
        N = H.size(0)

        # 准备注意力机制输入
        a_input = torch.cat(
            [H.repeat(1, N).view(N * N, -1), H.repeat(N, 1)],
            dim=1
        )  # [N*N, 2*out_dim]

        # 计算注意力分数
        e = self.leaky_relu(a_input @ self.a).view(N, N)

        # 应用邻接矩阵掩码
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)  # 归一化

        # 聚合邻居信息
        return F.relu(attention @ H)


class TCNBlock(nn.Module):
    """时间卷积网络块（因果卷积）"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # 保持时序长度不变
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)

    def forward(self, x):
        # x: [B, C, T]
        out = self.conv(x)
        # 因果裁剪：移除尾部填充部分
        out = out[:, :, :-self.conv.padding[0]]
        return F.relu(out)


class MultiHeadCrossAttention(nn.Module):
    """多头交叉注意力机制"""

    def __init__(self, q_dim, kv_dim, hidden_dim=128, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # 线性变换层
        self.q_proj = nn.Linear(q_dim, hidden_dim)
        self.k_proj = nn.Linear(kv_dim, hidden_dim)
        self.v_proj = nn.Linear(kv_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)  # 输出层

    def forward(self, q, kv, return_attn=False):
        """
        前向传播
        参数:
            q: 查询向量 [B, Dq]
            kv: 键值对 [B, N, Dk]
        返回:
            注意力加权后的表示 [B, hidden_dim]
        """
        B, N, _ = kv.shape

        # 线性变换并重塑为多头
        Q = self.q_proj(q).view(B, self.num_heads, self.head_dim)  # [B, H, D]
        K = self.k_proj(kv).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]
        V = self.v_proj(kv).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]

        # 计算注意力分数 [B, H, 1, N]
        attn_scores = torch.matmul(Q.unsqueeze(2), K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # 归一化

        # 加权聚合值向量
        context = torch.matmul(attn_weights, V).squeeze(2)  # [B, H, D]
        context = context.reshape(B, -1)  # [B, H*D]

        # 输出变换
        out = self.out_proj(context)  # [B, hidden_dim]

        if return_attn:
            return out, attn_weights
        return out


# -----------------------------
# 4) 特征提取器（使用真实环境信息）
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self,
                 state_dim,  # 原始状态维度
                 use_gnn=True,  # 是否使用GNN
                 use_tcn=True,  # 是否使用TCN
                 use_mha=False,  # 是否使用多头注意力融合
                 gnn_node_feat_dim=4,  # GNN节点特征维度
                 gnn_hidden=64,  # GNN隐藏层维度
                 tcn_hidden=64,  # TCN隐藏层维度
                 tcn_seq_len=5,  # TCN序列长度
                 heads=4):  # 注意力头数
        super().__init__()
        self.use_gnn = use_gnn
        self.use_tcn = use_tcn
        self.use_mha = use_mha
        self.gnn_hidden = gnn_hidden
        self.tcn_hidden = tcn_hidden
        self.tcn_seq_len = tcn_seq_len

        # 图神经网络（工序图和机器图）
        if self.use_gnn:
            self.op_gnn = GATLayer(gnn_node_feat_dim, gnn_hidden)
            self.mac_gnn = GATLayer(gnn_node_feat_dim, gnn_hidden)

        # 时间卷积网络
        if self.use_tcn:
            self.tcn1 = TCNBlock(1, 32, 3, 1)  # 第一层TCN
            self.tcn2 = TCNBlock(32, tcn_hidden, 3, 2)  # 第二层TCN

        # 交叉注意力融合
        if self.use_gnn and self.use_tcn and self.use_mha:
            self.mha = MultiHeadCrossAttention(q_dim=tcn_hidden, kv_dim=gnn_hidden,
                                               hidden_dim=128, num_heads=heads)
            final_in = 128
        else:
            # 确定最终输入维度
            final_in = 0
            if self.use_gnn:
                final_in += gnn_hidden * 2  # 工序图和机器图
            if self.use_tcn:
                final_in += tcn_hidden
            if final_in == 0:
                final_in = state_dim  # 仅使用原始状态

        # 输出层
        self.out = nn.Linear(final_in, 128)

    def build_graphs(self, env):
        """
        根据当前环境状态构建工序图和机器图
        参数:
            env: 当前环境实例
        返回:
            op_adj: 工序图邻接矩阵 [N_op, N_op]
            mac_adj: 机器图邻接矩阵 [N_mac, N_mac]
            op_X: 工序节点特征 [N_op, F]
            mac_X: 机器节点特征 [N_mac, F]
        """
        # 构建机器节点特征 [机器数量, 特征维度]
        mac_nodes = env.M_real
        mac_X = np.zeros((mac_nodes, 4))
        for m_idx in range(mac_nodes):
            # 特征1: 机器当前空闲时间（归一化）
            max_time = max(env.machine_busy_until) if env.machine_busy_until else 1
            if max_time == 0:
                max_time = 1
            mac_X[m_idx, 0] = env.machine_busy_until[m_idx] / max_time
            # 特征2: 机器利用率
            mac_X[m_idx, 1] = env.machine_busy_time[m_idx] / max(1, env.current_time)
            # 特征3: 机器等待队列长度
            queue_len = sum(1 for j in range(env.J) if env.job_progress[j] < len(env.jobs[j]))
            mac_X[m_idx, 2] = queue_len / env.J
            # 特征4: 机器类型（占位，实际应用中可添加更多特征）
            mac_X[m_idx, 3] = 0

        # 构建机器图（全连接）
        mac_adj = np.ones((mac_nodes, mac_nodes)) - np.eye(mac_nodes)

        # 构建工序节点特征 [工序总数, 特征维度]
        op_nodes = sum(len(job) for job in env.jobs)
        op_X = np.zeros((op_nodes, 4))
        # 构建工序图（链式结构）
        op_adj = np.zeros((op_nodes, op_nodes))

        node_idx = 0
        for j in range(env.J):
            job_len = len(env.jobs[j])
            for step_idx in range(job_len):
                # 特征1: 工序状态（0:未开始, 0.5:加工中, 1:已完成）
                status = env.job_step_status[j].get(step_idx, None)
                if status is None:
                    op_X[node_idx, 0] = 0.0
                elif status == 'processing':
                    op_X[node_idx, 0] = 0.5
                else:  # 'finished'
                    op_X[node_idx, 0] = 1.0

                # 特征2: 工序加工时间（归一化）
                max_duration = max(t for step in env.jobs for m, t in step)
                min_duration = min(t for step in env.jobs for m, t in step)
                duration_range = max_duration - min_duration if max_duration != min_duration else 1
                op_duration = min(t for m, t in env.jobs[j][step_idx])
                op_X[node_idx, 1] = (op_duration - min_duration) / duration_range

                # 特征3: 工序在作业中的位置
                op_X[node_idx, 2] = step_idx / job_len

                # 特征4: 作业紧急程度
                op_X[node_idx, 3] = (env.job_progress[j] - step_idx) / job_len

                # 添加工序图边（链式结构）
                if step_idx > 0:
                    prev_idx = node_idx - 1
                    op_adj[prev_idx, node_idx] = 1  # 前驱关系
                    op_adj[node_idx, prev_idx] = 1

                node_idx += 1

        return op_adj, mac_adj, op_X, mac_X

    def forward(self, state, env=None):
        """
        前向传播
        参数:
            state: 原始状态向量 [B, D_state]
            env: 环境实例（用于构建图结构）
        返回:
            特征向量 [B, 128]
        """
        B = state.size(0)
        device = state.device
        feats = []  # 存储不同模块的特征

        # GNN特征提取
        if self.use_gnn:
            if env is None:
                # 默认使用随机图（兼容旧版）
                N_op, N_mac = 6, 6
                op_X = torch.randn(N_op, 4, device=device)
                mac_X = torch.randn(N_mac, 4, device=device)
                op_adj = torch.triu(torch.ones(N_op, N_op, device=device), diagonal=1)
                mac_adj = torch.ones(N_mac, N_mac, device=device) - torch.eye(N_mac, device=device)
            else:
                # 使用真实环境信息构建图
                op_adj, mac_adj, op_X, mac_X = self.build_graphs(env)
                op_X = torch.tensor(op_X, dtype=torch.float32, device=device)
                mac_X = torch.tensor(mac_X, dtype=torch.float32, device=device)
                op_adj = torch.tensor(op_adj, dtype=torch.float32, device=device)
                mac_adj = torch.tensor(mac_adj, dtype=torch.float32, device=device)

            # GNN处理
            op_emb = self.op_gnn(op_X, op_adj).mean(0, keepdim=True).repeat(B, 1)  # [B, gnn_hidden]
            mac_emb = self.mac_gnn(mac_X, mac_adj).mean(0, keepdim=True).repeat(B, 1)  # [B, gnn_hidden]
            feats.extend([op_emb, mac_emb])

        # TCN特征提取
        if self.use_tcn:
            # 构建机器历史状态序列 [B, 1, seq_len]
            if env is None:
                # 默认随机序列
                tcn_seq = torch.randn(B, 1, self.tcn_seq_len, device=device)
            else:
                # 使用真实机器历史状态（这里简化实现）
                # 实际应用中应记录历史状态
                machine_states = np.array([env.machine_busy_until])
                tcn_seq = torch.tensor(machine_states, dtype=torch.float32, device=device).unsqueeze(1)
                # 如果序列长度不足，进行填充
                if tcn_seq.size(2) < self.tcn_seq_len:
                    padding = torch.zeros(B, 1, self.tcn_seq_len - tcn_seq.size(2), device=device)
                    tcn_seq = torch.cat([padding, tcn_seq], dim=2)
                else:
                    tcn_seq = tcn_seq[:, :, -self.tcn_seq_len:]

            # TCN处理
            tcn_out = self.tcn2(self.tcn1(tcn_seq)).mean(dim=2)  # [B, tcn_hidden]
            feats.append(tcn_out)

        # 特征融合
        if self.use_gnn and self.use_tcn and self.use_mha:
            # 将op和mac嵌入堆叠为键值对 [B, 2, gnn_hidden]
            kv = torch.stack([feats[0], feats[1]], dim=1)
            # 多头注意力融合
            out = self.mha(feats[2], kv)  # [B, 128]
        elif feats:
            # 简单拼接特征
            out = torch.cat(feats, dim=1)
        else:
            # 无特征提取模块，直接使用原始状态
            out = state

        # 输出层
        return self.out(out)  # [B, 128]


# -----------------------------
# 5) PPO代理（带合法动作掩码）
# -----------------------------
class PPOAgent(nn.Module):
    def __init__(self,
                 state_dim,
                 act_space,  # 动作空间列表 [(j, step_idx, m_ext)]
                 feature_cfg  # 特征提取器配置
                 ):
        super().__init__()
        self.act_space = act_space
        self.act2idx = {a: i for i, a in enumerate(act_space)}  # 动作到索引的映射
        self.idx2act = {i: a for a, i in self.act2idx.items()}  # 索引到动作的映射
        self.act_dim = len(act_space)  # 动作空间大小

        # 网络结构
        self.feature = FeatureExtractor(state_dim=state_dim, **feature_cfg)
        self.actor = nn.Linear(128, self.act_dim)  # 策略网络
        self.critic = nn.Linear(128, 1)  # 价值网络

        # 优化器
        self.optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.gamma = 0.99  # 折扣因子
        self.clip_eps = 0.2  # PPO裁剪参数
        self.lmbda = 0.95  # GAE参数

    def forward(self, state, env=None):
        """
        前向传播
        参数:
            state: 状态向量 [B, D]
            env: 环境实例（用于特征提取）
        返回:
            logits: 动作logits [B, A]
            value: 状态价值 [B, 1]
        """
        emb = self.feature(state, env)  # 提取特征
        logits = self.actor(emb)  # 策略输出
        value = self.critic(emb)  # 价值输出
        return logits, value

    @torch.no_grad()
    def select_action(self, state, valid_actions, env=None):
        """
        选择动作
        参数:
            state: 当前状态 (np.array)
            valid_actions: 合法动作列表
            env: 环境实例（用于特征提取）
        返回:
            action_idx: 动作索引
            logp: 动作对数概率
            value: 状态价值估计
        """
        device = next(self.parameters()).device
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # [1, D]
        logits, value = self.forward(state_t, env)
        logits = logits.squeeze(0)  # [A]
        value = value.squeeze().item()

        # 创建动作掩码（非法动作设为负无穷）
        mask = torch.full_like(logits, float('-inf'))
        valid_idx = [self.act2idx[a] for a in valid_actions if a in self.act2idx]
        mask[valid_idx] = 0.0

        # 计算动作概率
        probs = F.softmax(logits + mask, dim=-1)
        dist = torch.distributions.Categorical(probs)
        a_idx = dist.sample()
        logp = dist.log_prob(a_idx)

        return a_idx.item(), logp.item(), value

    def compute_gae(self, rewards, values, dones):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        next_value = 0

        # 反向计算GAE
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_values = values[t + 1]

            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.lmbda * next_non_terminal * gae
            advantages.insert(0, gae)

        return torch.tensor(advantages, dtype=torch.float32)

    def update(self, batch):
        """
        更新策略
        参数:
            batch: 包含一批经验数据的字典
        """
        device = next(self.parameters()).device

        # 转换数据为Tensor
        states = torch.tensor(np.array(batch['states']), dtype=torch.float32, device=device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        old_logps = torch.tensor(batch['logps'], dtype=torch.float32, device=device)
        values = torch.tensor(batch['values'], dtype=torch.float32, device=device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=device)
        dones = torch.tensor(batch['dones'], dtype=torch.float32, device=device)

        # 计算优势估计
        advantages = self.compute_gae(rewards, values, dones).to(device)
        returns = advantages + values

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多次更新（PPO的epoch）
        for _ in range(3):
            # 计算新策略
            logits, new_values = self.forward(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_logps = dist.log_prob(actions)

            # 计算概率比
            ratio = torch.exp(new_logps - old_logps)

            # 计算裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 价值损失
            value_loss = F.mse_loss(new_values.squeeze(), returns)

            # 总损失
            loss = policy_loss + 0.5 * value_loss

            # 反向传播
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optim.step()


# -----------------------------
# 6) 训练 & 评估（多指标 & 消融）
# -----------------------------
def run_one_experiment(
        name,
        jobs,
        episodes=100,
        feature_cfg=None,
        seed=0
):
    """
    运行单个实验
    参数:
        name: 实验名称（用于标识）
        jobs: 作业数据
        episodes: 训练轮数
        feature_cfg: 特征提取器配置
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 初始化环境
    env = BrandEnv(jobs)
    state_dim = env.M_real + env.J * 2  # 状态维度
    act_space = env.act_space  # 动作空间

    # 初始化代理
    agent = PPOAgent(state_dim, act_space, feature_cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.to(device)

    # 记录器
    ep_records = []
    total_time = 0

    # 训练循环
    for ep in range(episodes):
        start_time = time.time()
        s = env.reset()
        done = False
        batch = defaultdict(list)
        ep_reward = 0

        # 回合循环
        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                # 无合法动作，等待事件推进
                env.step((0, 0, 0))  # 空动作
                continue

            # 选择动作
            a_idx, logp, value = agent.select_action(s, valid_actions, env)
            j, step_idx, m_ext = agent.idx2act[a_idx]

            # 执行动作
            s2, r, done, _ = env.step((j, step_idx, m_ext))

            # 存储经验
            batch['states'].append(s)
            batch['actions'].append(a_idx)
            batch['logps'].append(logp)
            batch['values'].append(value)
            batch['rewards'].append(r)
            batch['dones'].append(done)

            s = s2
            ep_reward += r

        # 更新策略
        if len(batch['states']) > 0:
            agent.update(batch)

        # 计算性能指标
        metrics = env.get_performance_metrics()
        ep_time = time.time() - start_time
        total_time += ep_time

        # 记录结果
        record = {
            'Episode': ep + 1,
            'Reward': ep_reward,
            'Time': ep_time,
        }
        record.update(metrics)
        ep_records.append(record)

        # 定期输出
        if (ep + 1) % max(1, (episodes // 10)) == 0 or ep == 0:
            print(f"[{name}] Ep {ep + 1}/{episodes} | "
                  f"Makespan: {metrics.get('makespan', 0):.1f} | "
                  f"Flowtime: {metrics.get('flowtime', 0):.1f} | "
                  f"Util: {metrics.get('utilization', 0):.3f} | "
                  f"Time: {ep_time:.1f}s")

    # 保存结果
    df = pd.DataFrame(ep_records)
    df.to_csv(f"{name}.csv", index=False)

    # 保存调度方案
    schedule_df = env.get_schedule()
    schedule_df.to_csv(f"{name}_schedule.csv", index=False)

    print(f"[{name}] 平均训练时间: {total_time / episodes:.2f}s/ep")
    return df


def run_all(jobs, episodes=100, seed=0):
    """
    运行所有消融实验
    参数:
        jobs: 作业数据
        episodes: 每个实验的回合数
        seed: 随机种子
    """
    # 实验配置（5种模型）
    cfgs = [
        ("MLP_Only", dict(use_gnn=False, use_tcn=False, use_mha=False)),
        ("GNN_Only", dict(use_gnn=True, use_tcn=False, use_mha=False)),
        ("TCN_Only", dict(use_gnn=False, use_tcn=True, use_mha=False)),
        ("GNN_TCN", dict(use_gnn=True, use_tcn=True, use_mha=False)),
        ("GNN_TCN_MHA", dict(use_gnn=True, use_tcn=True, use_mha=True)),
    ]

    results = {}
    for name, feature_cfg in cfgs:
        print(f"\n{'=' * 20} 运行 {name} {'=' * 20}")
        df = run_one_experiment(name, jobs, episodes=episodes, feature_cfg=feature_cfg, seed=seed)
        results[name] = df

    # 可视化结果
    plot_results(results)
    return results


def plot_results(results):
    """可视化不同模型的性能对比"""
    # Makespan对比
    plt.figure(figsize=(12, 8))
    for name, df in results.items():
        plt.plot(df['Episode'], df['makespan'], label=name, linewidth=2)
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("最大完工时间", fontsize=12)
    plt.title("不同特征组合的调度性能对比（最大完工时间）", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("compare_makespan.png", dpi=300)

    # 平均完工时间对比
    plt.figure(figsize=(12, 8))
    for name, df in results.items():
        plt.plot(df['Episode'], df['avg_completion'], label=name, linewidth=2)
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("平均完工时间", fontsize=12)
    plt.title("不同特征组合的调度性能对比（平均完工时间）", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("compare_avg_completion.png", dpi=300)

    # 机器利用率对比
    plt.figure(figsize=(12, 8))
    for name, df in results.items():
        plt.plot(df['Episode'], df['utilization'], label=name, linewidth=2)
    plt.xlabel("训练轮数", fontsize=12)
    plt.ylabel("机器利用率", fontsize=12)
    plt.title("不同特征组合的调度性能对比（机器利用率）", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("compare_utilization.png", dpi=300)

    plt.show()


# -----------------------------
# 7) 入口
# -----------------------------
if __name__ == "__main__":
    # 1) 修改为你的 Brandimarte 文件名
    data_file = "Brandimarte_MK01.fjs"

    # 2) 读取数据
    print(f"加载数据: {data_file}")
    jobs = load_brandimarte(data_file)
    print(f"加载完成: {len(jobs)}个作业")

    # 3) 运行消融实验
    print("\n开始消融实验...")
    start_time = time.time()
    results = run_all(jobs, episodes=10, seed=42)

    # 4) 输出最终性能
    summary = []
    for name, df in results.items():
        last_5 = df.tail(5)
        summary.append({
            "Model": name,
            "Makespan(mean)": last_5["makespan"].mean(),
            "Flowtime(mean)": last_5["flowtime"].mean(),
            "AvgCompletion(mean)": last_5["avg_completion"].mean(),
            "Utilization(mean)": last_5["utilization"].mean(),
            "Time(mean)": last_5["Time"].mean()
        })

    summary_df = pd.DataFrame(summary)
    print("\n最终性能对比:")
    print(summary_df.round(3))
    summary_df.to_csv("ablation_summary.csv", index=False)

    print(f"\n总耗时: {time.time() - start_time:.1f}秒")