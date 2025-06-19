import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.nn import GATConv
from sklearn.manifold import TSNE
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# -----------------------------
# 全局设置
# -----------------------------
save_path = "scheduling_graphs"  # 结果保存路径
os.makedirs(save_path, exist_ok=True)  # 确保目录存在

# -----------------------------
# MK01柔性车间调度实例数据
# -----------------------------
# 参考标准柔性车间调度问题MK01实例
# 10个工件，6台机器，每个工件6道工序（除了工件3有5道工序）

# 机器数量
num_machines = 6

# 工件工序数据 (工件, 工序): [可选机器列表]
job_operations = {
    # 工件0
    (0, 0): [0, 1, 2, 3, 4, 5],
    (0, 1): [0, 1, 2, 3, 4, 5],
    (0, 2): [0, 1, 2, 3, 4, 5],
    (0, 3): [0, 1, 2, 3, 4, 5],
    (0, 4): [0, 1, 2, 3, 4, 5],
    (0, 5): [0, 1, 2, 3, 4, 5],

    # 工件1
    (1, 0): [0, 1, 2, 3, 4, 5],
    (1, 1): [0, 1, 2, 3, 4, 5],
    (1, 2): [0, 1, 2, 3, 4, 5],
    (1, 3): [0, 1, 2, 3, 4, 5],
    (1, 4): [0, 1, 2, 3, 4, 5],
    (1, 5): [0, 1, 2, 3, 4, 5],

    # 工件2
    (2, 0): [0, 1, 2, 3, 4, 5],
    (2, 1): [0, 1, 2, 3, 4, 5],
    (2, 2): [0, 1, 2, 3, 4, 5],
    (2, 3): [0, 1, 2, 3, 4, 5],
    (2, 4): [0, 1, 2, 3, 4, 5],
    (2, 5): [0, 1, 2, 3, 4, 5],

    # 工件3 (只有5道工序)
    (3, 0): [0, 1, 2, 3, 4, 5],
    (3, 1): [0, 1, 2, 3, 4, 5],
    (3, 2): [0, 1, 2, 3, 4, 5],
    (3, 3): [0, 1, 2, 3, 4, 5],
    (3, 4): [0, 1, 2, 3, 4, 5],

    # 工件4
    (4, 0): [0, 1, 2, 3, 4, 5],
    (4, 1): [0, 1, 2, 3, 4, 5],
    (4, 2): [0, 1, 2, 3, 4, 5],
    (4, 3): [0, 1, 2, 3, 4, 5],
    (4, 4): [0, 1, 2, 3, 4, 5],
    (4, 5): [0, 1, 2, 3, 4, 5],

    # 工件5
    (5, 0): [0, 1, 2, 3, 4, 5],
    (5, 1): [0, 1, 2, 3, 4, 5],
    (5, 2): [0, 1, 2, 3, 4, 5],
    (5, 3): [0, 1, 2, 3, 4, 5],
    (5, 4): [0, 1, 2, 3, 4, 5],
    (5, 5): [0, 1, 2, 3, 4, 5],

    # 工件6
    (6, 0): [0, 1, 2, 3, 4, 5],
    (6, 1): [0, 1, 2, 3, 4, 5],
    (6, 2): [0, 1, 2, 3, 4, 5],
    (6, 3): [0, 1, 2, 3, 4, 5],
    (6, 4): [0, 1, 2, 3, 4, 5],
    (6, 5): [0, 1, 2, 3, 4, 5],

    # 工件7
    (7, 0): [0, 1, 2, 3, 4, 5],
    (7, 1): [0, 1, 2, 3, 4, 5],
    (7, 2): [0, 1, 2, 3, 4, 5],
    (7, 3): [0, 1, 2, 3, 4, 5],
    (7, 4): [0, 1, 2, 3, 4, 5],
    (7, 5): [0, 1, 2, 3, 4, 5],

    # 工件8
    (8, 0): [0, 1, 2, 3, 4, 5],
    (8, 1): [0, 1, 2, 3, 4, 5],
    (8, 2): [0, 1, 2, 3, 4, 5],
    (8, 3): [0, 1, 2, 3, 4, 5],
    (8, 4): [0, 1, 2, 3, 4, 5],
    (8, 5): [0, 1, 2, 3, 4, 5],

    # 工件9
    (9, 0): [0, 1, 2, 3, 4, 5],
    (9, 1): [0, 1, 2, 3, 4, 5],
    (9, 2): [0, 1, 2, 3, 4, 5],
    (9, 3): [0, 1, 2, 3, 4, 5],
    (9, 4): [0, 1, 2, 3, 4, 5],
    (9, 5): [0, 1, 2, 3, 4, 5],
}

# 加工时间数据 (工件, 工序, 机器): 加工时间
# 为简化，这里使用随机生成的时间
processing_times = {}
for job_op, machines in job_operations.items():
    for machine in machines:
        # 随机生成1-10之间的加工时间
        processing_times[(job_op[0], job_op[1], machine)] = np.random.randint(1, 11)

# 创建操作索引
operations = list(job_operations.keys())
operation_to_idx = {op: idx for idx, op in enumerate(operations)}
num_operations = len(operations)


# -----------------------------
# 图构建函数
# -----------------------------

def build_disjunctive_graph():
    """构建析取图"""
    G = nx.DiGraph()

    # 添加节点 (操作)
    for op in operations:
        job, step = op
        G.add_node(operation_to_idx[op], job=job, step=step, type='operation')

    # 添加连接弧 (同一工件内的顺序约束)
    for job in range(10):  # 10个工件
        steps = [step for (j, step) in job_operations.keys() if j == job]
        steps.sort()
        for i in range(len(steps) - 1):
            src = (job, steps[i])
            dst = (job, steps[i + 1])
            G.add_edge(operation_to_idx[src], operation_to_idx[dst], type='sequence')

    # 添加析取弧 (同一机器上的操作对)
    machine_operations = {m: [] for m in range(num_machines)}
    for op, machines in job_operations.items():
        for m in machines:
            machine_operations[m].append(op)

    for m, ops in machine_operations.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                src_idx = operation_to_idx[ops[i]]
                dst_idx = operation_to_idx[ops[j]]
                # 添加双向边表示可能冲突
                G.add_edge(src_idx, dst_idx, type='disjunctive', machine=m)
                G.add_edge(dst_idx, src_idx, type='disjunctive', machine=m)

    return G


def build_simplified_disjunctive_graph():
    """构建简化析取图（工序顺序图）"""
    G = nx.DiGraph()

    # 添加节点 (操作)
    for op in operations:
        job, step = op
        G.add_node(operation_to_idx[op], job=job, step=step, type='operation')

    # 添加连接弧 (同一工件内的顺序约束)
    for job in range(10):  # 10个工件
        steps = [step for (j, step) in job_operations.keys() if j == job]
        steps.sort()
        for i in range(len(steps) - 1):
            src = (job, steps[i])
            dst = (job, steps[i + 1])
            G.add_edge(operation_to_idx[src], operation_to_idx[dst], type='sequence')

    # 在简化图中不添加析取弧
    return G


def build_operation_machine_bipartite():
    """构建操作-机器二分图"""
    G = nx.Graph()

    # 添加操作节点
    for op in operations:
        job, step = op
        G.add_node(f"op_{operation_to_idx[op]}", type='operation', job=job, step=step)

    # 添加机器节点
    for m in range(num_machines):
        G.add_node(f"machine_{m}", type='machine', machine_id=m)

    # 添加边 (操作和机器之间)
    for op, machines in job_operations.items():
        op_node = f"op_{operation_to_idx[op]}"
        for m in machines:
            machine_node = f"machine_{m}"
            # 添加边的权重为加工时间
            time = processing_times[(op[0], op[1], m)]
            G.add_edge(op_node, machine_node, weight=time, processing_time=time)

    return G


# -----------------------------
# 图可视化函数
# -----------------------------

def plot_and_save_graph(G, title, filename, pos=None, node_size=800):
    """绘制并保存图"""
    plt.figure(figsize=(12, 8))

    # 节点颜色和标签
    node_colors = []
    labels = {}

    for node in G.nodes():
        if 'type' in G.nodes[node]:
            if G.nodes[node]['type'] == 'operation':
                node_colors.append('lightblue')
                job = G.nodes[node]['job']
                step = G.nodes[node]['step']
                labels[node] = f"J{job}O{step}"
            elif G.nodes[node]['type'] == 'machine':
                node_colors.append('lightgreen')
                machine_id = G.nodes[node]['machine_id']
                labels[node] = f"M{machine_id}"
        else:
            node_colors.append('lightgray')
            labels[node] = node

    # 边样式
    edge_colors = []
    edge_styles = []

    for u, v, data in G.edges(data=True):
        if 'type' in data:
            if data['type'] == 'sequence':
                edge_colors.append('blue')
                edge_styles.append('solid')
            elif data['type'] == 'disjunctive':
                edge_colors.append('red')
                edge_styles.append('dashed')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')

    # 如果没有提供位置信息，则计算布局
    if pos is None:
        if isinstance(G, nx.DiGraph):
            pos = nx.spring_layout(G, k=0.5, iterations=50)
        else:
            pos = nx.spring_layout(G)

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.9)

    # 绘制边
    for i, (u, v, data) in enumerate(G.edges(data=True)):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=[edge_colors[i]],
                               style=[edge_styles[i]],
                               width=1.5)

    # 绘制标签
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    # 添加边标签（用于二分图的加工时间）
    if 'weight' in list(G.edges(data=True))[0][2]:
        edge_labels = {(u, v): f"{data['processing_time']}" for u, v, data in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()
    print(f"已保存图: {os.path.join(save_path, filename)}")


# -----------------------------
# GAT模型定义
# -----------------------------

class MultiGraphGAT(nn.Module):
    """多图融合的GAT模型"""

    def __init__(self, in_feats, hidden_size, num_heads, num_operations):
        super(MultiGraphGAT, self).__init__()
        # 析取图GAT层
        self.dg_gat1 = GATConv(in_feats, hidden_size, num_heads)
        self.dg_gat2 = GATConv(hidden_size * num_heads, hidden_size, 1)

        # 工序顺序图GAT层
        self.sog_gat1 = GATConv(in_feats, hidden_size, num_heads)
        self.sog_gat2 = GATConv(hidden_size * num_heads, hidden_size, 1)

        # 操作-机器二分图GAT层
        self.omb_gat1 = GATConv(in_feats, hidden_size, num_heads)
        self.omb_gat2 = GATConv(hidden_size * num_heads, hidden_size, 1)

        # 融合层
        self.fc = nn.Linear(hidden_size * 3, hidden_size)
        self.output = nn.Linear(hidden_size, num_operations)

        self.num_operations = num_operations
        self.hidden_size = hidden_size

    def forward(self, dg, sog, omb, features):
        # 析取图特征提取
        x_dg = F.elu(self.dg_gat1(dg, features))
        x_dg = x_dg.view(-1, x_dg.size(1) * x_dg.size(2))
        x_dg = F.elu(self.dg_gat2(dg, x_dg))
        x_dg = x_dg.squeeze(1)

        # 工序顺序图特征提取
        x_sog = F.elu(self.sog_gat1(sog, features))
        x_sog = x_sog.view(-1, x_sog.size(1) * x_sog.size(2))
        x_sog = F.elu(self.sog_gat2(sog, x_sog))
        x_sog = x_sog.squeeze(1)

        # 操作-机器二分图特征提取
        x_omb = F.elu(self.omb_gat1(omb, features))
        x_omb = x_omb.view(-1, x_omb.size(1) * x_omb.size(2))
        x_omb = F.elu(self.omb_gat2(omb, x_omb))
        x_omb = x_omb.squeeze(1)

        # 只取操作节点的特征
        x_dg = x_dg[:self.num_operations]
        x_sog = x_sog[:self.num_operations]
        x_omb = x_omb[:self.num_operations]

        # 特征融合
        x_fused = torch.cat((x_dg, x_sog, x_omb), dim=1)
        x_fused = F.elu(self.fc(x_fused))

        # 输出层
        out = self.output(x_fused)
        return out, x_fused


# -----------------------------
# 特征可视化函数
# -----------------------------

def visualize_features(features, labels, title, filename):
    """使用t-SNE可视化特征"""
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.detach().cpu().numpy())

    plt.figure(figsize=(10, 8))

    # 为每个工件创建颜色映射
    unique_jobs = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_jobs)))
    color_map = {job: colors[i] for i, job in enumerate(unique_jobs)}

    # 绘制每个点
    for i in range(len(features_2d)):
        job = labels[i]
        plt.scatter(features_2d[i, 0], features_2d[i, 1],
                    color=color_map[job], label=f'J{job}' if i == job else "")

    # 添加图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[job], markersize=10, label=f'J{job}')
        for job in unique_jobs]
    plt.legend(handles=handles, title="工件")

    plt.title(title)
    plt.xlabel("t-SNE特征1")
    plt.ylabel("t-SNE特征2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()
    print(f"已保存特征可视化: {os.path.join(save_path, filename)}")


# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    print("开始构建调度图结构...")

    # 构建三种图结构
    dg = build_disjunctive_graph()
    sog = build_simplified_disjunctive_graph()
    omb = build_operation_machine_bipartite()

    print(f"析取图节点数: {dg.number_of_nodes()}, 边数: {dg.number_of_edges()}")
    print(f"工序顺序图节点数: {sog.number_of_nodes()}, 边数: {sog.number_of_edges()}")
    print(f"操作-机器二分图节点数: {omb.number_of_nodes()}, 边数: {omb.number_of_edges()}")

    # 可视化图结构
    print("可视化图结构...")
    plot_and_save_graph(dg, "析取图 (Disjunctive Graph)", "disjunctive_graph.png")
    plot_and_save_graph(sog, "工序顺序图 (Simplified Disjunctive Graph)", "sequence_graph.png")
    plot_and_save_graph(omb, "操作-机器二分图 (Operation-Machine Bipartite Graph)", "bipartite_graph.png",
                        node_size=500)

    # 准备DGL图
    print("准备DGL图用于GAT...")

    # 将NetworkX图转换为DGL图
    dgl_dg = DGLGraph()
    dgl_dg.from_networkx(dg, edge_attrs=['type', 'machine'])

    dgl_sog = DGLGraph()
    dgl_sog.from_networkx(sog, edge_attrs=['type'])

    dgl_omb = DGLGraph()
    dgl_omb.from_networkx(omb, edge_attrs=['weight', 'processing_time'])

    # 创建节点特征
    # 特征: [工件ID, 工序步骤, 最小加工时间, 最大加工时间, 平均加工时间]
    features = torch.zeros(num_operations, 5)
    labels = []  # 用于可视化工件标签

    for op, idx in operation_to_idx.items():
        job, step = op
        times = [processing_times[(job, step, m)] for m in job_operations[op]]
        min_time = min(times)
        max_time = max(times)
        avg_time = sum(times) / len(times)

        features[idx] = torch.tensor([job, step, min_time, max_time, avg_time])
        labels.append(job)

    # 添加虚拟特征使维度匹配（对于二分图中的机器节点）
    num_omb_nodes = dgl_omb.number_of_nodes()
    # 操作节点在前，机器节点在后
    omb_features = torch.cat([
        features,
        torch.zeros(num_omb_nodes - num_operations, 5)
    ])

    # 创建模型
    hidden_size = 16
    num_heads = 2
    model = MultiGraphGAT(in_feats=5,
                          hidden_size=hidden_size,
                          num_heads=num_heads,
                          num_operations=num_operations)

    print("模型结构:")
    print(model)

    # 训练模型
    print("训练GAT模型...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 模拟训练（实际应用中应有真实标签）
    # 这里使用随机目标进行演示
    targets = torch.randn(num_operations, num_operations)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()

        # 前向传播
        output, fused_features = model(dgl_dg, dgl_sog, dgl_omb, features)

        # 计算损失（模拟任务）
        loss = F.mse_loss(output, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    print("训练完成!")

    # 提取融合后的特征
    with torch.no_grad():
        model.eval()
        _, fused_features = model(dgl_dg, dgl_sog, dgl_omb, features)
        print(f"融合特征维度: {fused_features.shape}")

    # 可视化特征
    print("可视化提取的特征...")
    visualize_features(fused_features, labels,
                       "GAT提取的特征空间 (t-SNE可视化)",
                       "gat_features_tsne.png")

    print("\n所有结果已保存至:", os.path.abspath(save_path))