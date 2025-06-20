# dataset.py

import torch
from torch_geometric.data import Data
from collections import defaultdict


def parse_mk_file(filepath, has_header=True):
    """
    解析 MK 数据集，支持可选头信息
    参数:
        filepath: 文件路径
        has_header: 是否包含头信息行，默认True
    返回:
        job_data: List[List[List[Tuple[int,int]]]]
    """
    job_data = []

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    if has_header:
        # 读取头信息
        header_line = lines[0]
        job_num, machine_num, max_choices = map(int, header_line.split())
        print(f"[INFO] 头信息：作业数={job_num}, 机器数={machine_num}, 最大候选数={max_choices}")
        lines = lines[1:]  # 跳过头信息

    for line_idx, line in enumerate(lines):
        tokens = list(map(int, line.split()))
        pointer = 0
        op_num = tokens[pointer]
        pointer += 1

        job = []
        for op_id in range(op_num):
            if pointer >= len(tokens):
                raise ValueError(f"[Line {line_idx + 1 + (1 if has_header else 0)}] 候选机器数读取失败")

            alt_machine_num = tokens[pointer]
            pointer += 1

            op_machines = []
            for _ in range(alt_machine_num):
                if pointer + 1 >= len(tokens):
                    raise ValueError(
                        f"[Line {line_idx + 1 + (1 if has_header else 0)}] 工序 {op_id} 机器读取越界"
                    )
                machine_id = tokens[pointer]
                proc_time = tokens[pointer + 1]
                op_machines.append((machine_id, proc_time))
                pointer += 2

            job.append(op_machines)

        job_data.append(job)

    return job_data


def build_disjunction_graph(job_data):
    """
    构建析取图（Disjunction Graph）
    节点是工序（Operation），边表示工序的可选机器
    输入：
        job_data: List[List[List[(machine_id, proc_time)]]]
                  三级列表，job -> op -> [(machine, time), ...]
    返回：
        torch_geometric.data.Data 对象，包含节点特征和边信息
    """

    # 统计所有工序总数，用作节点数量
    total_ops = sum(len(job) for job in job_data)

    # 节点特征：这里我们简单用one-hot编码机器选择特征（或者你也可以用加工时间等）
    # 节点数量 = total_ops * 最大候选机器数（做虚拟扩展示例，后续根据需求调整）
    node_features = []
    edge_index = [[], []]  # 边索引，保存边的连接关系

    node_id = 0  # 节点编号
    op_id_map = []  # 记录每个工序对应的节点id列表（每个候选机器对应一个节点）

    for job_id, job in enumerate(job_data):
        for op_id, op_machines in enumerate(job):
            op_node_ids = []
            for (machine, time) in op_machines:
                # 构建节点特征，比如 one-hot 机器id，假设机器数不超过 10
                machine_feat = [0] * 10
                machine_feat[machine - 1] = 1  # 机器编号从1开始
                # 这里可以加上加工时间等特征，简单起见只用机器one-hot
                node_features.append(machine_feat)
                op_node_ids.append(node_id)
                node_id += 1

            op_id_map.append(op_node_ids)

    # 构建边（示例）：同一工序的候选机器节点之间互相连接（完全连通）
    for op_nodes in op_id_map:
        for i in range(len(op_nodes)):
            for j in range(i + 1, len(op_nodes)):
                edge_index[0].append(op_nodes[i])
                edge_index[1].append(op_nodes[j])
                edge_index[0].append(op_nodes[j])
                edge_index[1].append(op_nodes[i])

    # 转成张量
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


def build_operation_sequence_graph(job_data):
    """
    构建工序顺序图（Operation Sequence Graph）
    节点为工序-机器选择，边表示工序前后顺序依赖。
    输入：
        job_data: List[List[List[(machine_id, proc_time)]]]
    返回：
        torch_geometric.data.Data对象，包含节点特征和边信息
    """

    node_features = []
    edge_index = [[], []]

    node_id = 0
    op_id_map = []  # 记录每个工序对应节点id列表（候选机器个数）

    # 先生成所有节点和节点特征
    for job in job_data:
        for op_machines in job:
            op_node_ids = []
            for (machine, time) in op_machines:
                feat = [0] * 10
                feat[machine - 1] = 1
                # 可以加入加工时间作为特征，这里用归一化时间示例（假设最大时间100）
                feat.append(time / 100.0)
                node_features.append(feat)
                op_node_ids.append(node_id)
                node_id += 1
            op_id_map.append(op_node_ids)

    # 现在构建顺序边：
    # 注意op_id_map是按作业顺序排列的，但我们需要区分作业，先恢复每个作业的工序数量
    job_lengths = [len(job) for job in job_data]

    # 按作业拆分op_id_map
    pointer = 0
    for length in job_lengths:
        job_ops = op_id_map[pointer: pointer + length]
        pointer += length

        # 按顺序连接相邻工序的所有候选机器节点
        for i in range(len(job_ops) - 1):
            curr_op_nodes = job_ops[i]
            next_op_nodes = job_ops[i + 1]

            # 每个当前工序节点连到下个工序所有候选机器节点（全连接）
            for src in curr_op_nodes:
                for dst in next_op_nodes:
                    edge_index[0].append(src)
                    edge_index[1].append(dst)

    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data


def build_bipartite_graph(job_data, num_machines=5):
    """
    构建机器-工序二分图（Bipartite Graph）
    - 左侧节点为机器节点，共 num_machines 个
    - 右侧节点为工序节点，每个工序的每个候选机器对应一个节点
    - 节点特征：机器节点用 one-hot 编码 + 加工时间占位0；工序节点用 one-hot 机器编码 + 归一化加工时间
    - 边表示机器可加工该工序，边特征为加工时间的归一化值

    参数：
        job_data: List[List[List[(machine_id, proc_time)]]]
            三层列表，job->operation->候选机器列表(机器ID，加工时间)
        num_machines: int，机器总数，默认5

    返回：
        data: torch_geometric.data.Data，包含节点特征、边索引和边特征
    """

    # 构造机器节点特征，one-hot + 加工时间占位0，维度为 num_machines + 1
    machine_feats = []
    for m in range(num_machines):
        feat = [0] * (num_machines + 1)  # 预留一个维度用于加工时间
        feat[m] = 1  # 机器 one-hot 编码
        feat[-1] = 0.0  # 机器节点加工时间占位为0
        machine_feats.append(feat)

    node_features = []  # 存放所有工序节点特征
    edge_index = [[], []]  # 边的起点和终点
    edge_attr = []  # 边特征（加工时间归一化）

    node_id = num_machines  # 工序节点编号从机器节点编号后开始

    # 遍历所有作业和工序，生成工序节点和边
    for job in job_data:
        for op_machines in job:
            for (machine, proc_time) in op_machines:
                # 工序节点特征：机器one-hot + 加工时间归一化(假设最大加工时间100)
                feat = [0] * (num_machines + 1)
                feat[machine - 1] = 1
                feat[-1] = proc_time / 100.0

                node_features.append(feat)

                # 构建边：机器节点 -> 工序节点
                edge_index[0].append(machine - 1)  # 机器节点编号
                edge_index[1].append(node_id)  # 工序节点编号

                # 边特征为加工时间归一化
                edge_attr.append([proc_time / 100.0])

                node_id += 1

    # 合并节点特征：先是机器节点特征，再是工序节点特征
    x = torch.tensor(machine_feats + node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 构造图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data
