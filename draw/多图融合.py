import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# === 柔性车间调度数据 ===
jobs = ['J1', 'J2', 'J3']
machines = ['M1', 'M2', 'M3']
operations = [
    'O11', 'O12', 'O13',
    'O21', 'O22', 'O23',
    'O31', 'O32', 'O33'
]

# 工序顺序 (有向边)
job_operation_edges = [
    ('O11', 'O12'), ('O12', 'O13'),
    ('O21', 'O22'), ('O22', 'O23'),
    ('O31', 'O32'), ('O32', 'O33')
]

# 机器冲突边（虚线红色）
conflict_edges = [('O11', 'O21'), ('O11', 'O31'), ('O21', 'O31')]

# 机器之间负载/物料流（有向边）
machine_edges = [('M1', 'M2'), ('M2', 'M3'), ('M3', 'M1')]  # 添加循环路径

# 操作对应可选机器
operation_machine_map = {
    'O11': ['M1', 'M2'],
    'O12': ['M2', 'M3'],
    'O13': ['M1'],
    'O21': ['M1', 'M3'],
    'O22': ['M2'],
    'O23': ['M2', 'M3'],
    'O31': ['M1'],
    'O32': ['M3'],
    'O33': ['M1', 'M2', 'M3']
}

# === 配色方案 ===
COLOR_OPERATION_NODE = '#1f77b4'    # 蓝
COLOR_CONFLICT_EDGE = '#d62728'     # 红
COLOR_MACHINE_NODE = '#2ca02c'      # 绿
COLOR_MACHINE_EDGE = '#ff7f0e'      # 橙
COLOR_JOB_NODE = '#9467bd'          # 紫
COLOR_JOB_EDGE = '#8c564b'          # 棕
COLOR_BIPARTITE_EDGE = '#7f7f7f'    # 灰

# 自定义渐变色谱：低负载（绿）→ 中负载（橙）→ 高负载（红）
cmap = LinearSegmentedColormap.from_list("machine_load", ["#4CAF50", "#FFA726", "#EF5350"])

# === 1. 析取图（Disjunctive Graph） ===
def draw_disjunctive_graph(save_path="disjunctive_graph.png"):
    G = nx.DiGraph()
    G.add_nodes_from(operations)
    G.add_edges_from(job_operation_edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_color=COLOR_OPERATION_NODE, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=job_operation_edges, edge_color='black', arrows=True)

    nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, edge_color=COLOR_CONFLICT_EDGE, style='dashed', width=2)

    plt.title("析取图 (Disjunctive Graph)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === 2. 工序顺序图（简化析取图） ===
def draw_job_graph(save_path="job_graph.png"):
    G = nx.DiGraph()
    G.add_nodes_from(operations)
    G.add_edges_from(job_operation_edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_color=COLOR_OPERATION_NODE, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=job_operation_edges, edge_color='black', arrows=True, width=2)

    plt.title("工序顺序图（简化析取图）")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === 改进的 3. 机器负载与物料流图 ===
def draw_machine_graph(save_path="machine_load_flow.png"):
    G = nx.DiGraph()
    G.add_nodes_from(machines)

    # 模拟负载数据（根据实际调度结果生成）
    load_data = {'M1': 0.6, 'M2': 0.8, 'M3': 0.4}  # 负载比例（0~1）

    # 模拟物料流动方向与流量
    flow_data = {
        ('M1', 'M2'): 0.7,
        ('M2', 'M3'): 0.5,
        ('M3', 'M1'): 0.3,
    }

    # 添加边
    for edge, weight in flow_data.items():
        G.add_edge(*edge, weight=weight)

    # 布局
    pos = nx.shell_layout(G)

    # 节点大小映射负载
    node_sizes = [1000 * (1 + load_data[node]) for node in G.nodes()]

    # 节点颜色映射负载
    node_colors = [cmap(load_data[node]) for node in G.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, node_shape='s')
    nx.draw_networkx_labels(G, pos, font_size=12)

    # 绘制边（箭头、宽度映射流量）
    for edge in G.edges(data=True):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(edge[0], edge[1])],
            width=2 + 4 * edge[2]['weight'],
            edge_color='black',
            arrows=True,
            arrowstyle='->',
            arrowsize=20
        )

    plt.title("机器负载与物料流图（柔性车间）")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === 4. 操作-机器二分图 ===
def draw_operation_machine_bipartite(save_path="operation_machine_bipartite.png"):
    B = nx.Graph()
    B.add_nodes_from(operations, bipartite=0)
    B.add_nodes_from(machines, bipartite=1)

    edges = []
    for op, ms in operation_machine_map.items():
        for m in ms:
            edges.append((op, m))
    B.add_edges_from(edges)

    pos = nx.spring_layout(B, seed=42)
    plt.figure(figsize=(10, 6))

    node_colors = [COLOR_OPERATION_NODE if n in operations else COLOR_MACHINE_NODE for n in B.nodes()]
    node_shapes = {'operation': 'o', 'machine': 's'}

    ops_nodes = [n for n in B.nodes() if n in operations]
    mach_nodes = [n for n in B.nodes() if n in machines]

    nx.draw_networkx_nodes(B, pos, nodelist=ops_nodes, node_color=COLOR_OPERATION_NODE, node_shape='o', node_size=700)
    nx.draw_networkx_nodes(B, pos, nodelist=mach_nodes, node_color=COLOR_MACHINE_NODE, node_shape='s', node_size=700)
    nx.draw_networkx_labels(B, pos, font_size=10)

    nx.draw_networkx_edges(B, pos, edge_color=COLOR_BIPARTITE_EDGE, width=1.5)

    plt.title("操作-机器二分图 (Operation-Machine Bipartite Graph)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# === 主程序入口 ===
if __name__ == '__main__':
    draw_disjunctive_graph()
    draw_job_graph()
    draw_machine_graph()
    draw_operation_machine_bipartite()
    print("✅ 图像已保存至当前目录：")
    print("1. disjunctive_graph.png - 析取图")
    print("2. job_graph.png - 工序顺序图")
    print("3. machine_load_flow.png - 改进的机器负载与物料流图")
    print("4. operation_machine_bipartite.png - 操作-机器二分图")