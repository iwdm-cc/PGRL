import networkx as nx
import matplotlib.pyplot as plt
# 设置 matplotlib 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


# ==== 统一实例数据 ====
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

# 机器之间负载/物料流（无向）
machine_edges = [('M1', 'M2'), ('M2', 'M3')]

# 工件间依赖 (有向)
job_edges = [('J1', 'J2'), ('J2', 'J3')]

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

# ==== 配色方案，参考SCI论文常用色，颜色码
COLOR_OPERATION_NODE = '#1f77b4'    # 蓝
COLOR_CONFLICT_EDGE = '#d62728'     # 红
COLOR_MACHINE_NODE = '#2ca02c'      # 绿
COLOR_MACHINE_EDGE = '#ff7f0e'      # 橙
COLOR_JOB_NODE = '#9467bd'          # 紫
COLOR_JOB_EDGE = '#8c564b'          # 棕
COLOR_BIPARTITE_EDGE = '#7f7f7f'    # 灰

# ==== 1. 析取图（Disjunctive Graph） ====
def draw_disjunctive_graph():
    G = nx.DiGraph()
    G.add_nodes_from(operations)
    G.add_edges_from(job_operation_edges)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_color=COLOR_OPERATION_NODE, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=job_operation_edges, edge_color='black', arrows=True)

    # 机器冲突边用红色虚线
    nx.draw_networkx_edges(G, pos, edgelist=conflict_edges, edge_color=COLOR_CONFLICT_EDGE, style='dashed', width=2)

    plt.title("析取图 (Disjunctive Graph)")
    plt.axis('off')
    plt.show()

# ==== 改进的 2. 工序顺序图（简化析取图）====
def draw_job_graph():
    G = nx.DiGraph()
    G.add_nodes_from(operations)  # 工序节点
    G.add_edges_from(job_operation_edges)  # 顺序依赖边

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    nx.draw_networkx_nodes(G, pos, node_color=COLOR_OPERATION_NODE, node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edgelist=job_operation_edges, edge_color='black', arrows=True, width=2)

    plt.title("工序顺序图（简化析取图）")
    plt.axis('off')
    plt.show()

# ==== 改进的 3. 机器负载与物料流图 ====
def draw_machine_graph():
    G = nx.Graph()
    G.add_nodes_from(machines)
    G.add_edges_from(machine_edges)

    pos = nx.circular_layout(G)
    plt.figure(figsize=(6, 6))

    nx.draw_networkx_nodes(G, pos, node_color=COLOR_MACHINE_NODE, node_shape='s', node_size=800)
    nx.draw_networkx_labels(G, pos, font_size=12)
    nx.draw_networkx_edges(G, pos, edge_color=COLOR_MACHINE_EDGE, width=2)

    plt.title("机器负载与物料流图")
    plt.axis('off')
    plt.show()

# ==== 4. 操作-机器二分图 (Operation-Machine Bipartite Graph) ====
def draw_operation_machine_bipartite():
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

    # 先画所有节点（因为形状不同，要分两次绘制）
    ops_nodes = [n for n in B.nodes() if n in operations]
    mach_nodes = [n for n in B.nodes() if n in machines]

    nx.draw_networkx_nodes(B, pos, nodelist=ops_nodes, node_color=COLOR_OPERATION_NODE, node_shape='o', node_size=700)
    nx.draw_networkx_nodes(B, pos, nodelist=mach_nodes, node_color=COLOR_MACHINE_NODE, node_shape='s', node_size=700)
    nx.draw_networkx_labels(B, pos, font_size=10)

    nx.draw_networkx_edges(B, pos, edge_color=COLOR_BIPARTITE_EDGE, width=1.5)

    plt.title("操作-机器二分图 (Operation-Machine Bipartite Graph)")
    plt.axis('off')
    plt.show()


# ==== 运行示例 ====
if __name__ == '__main__':
    draw_disjunctive_graph()
    draw_machine_graph()
    draw_job_graph()
    draw_operation_machine_bipartite()
