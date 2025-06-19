import numpy as np
import networkx as nx

num_machines = 6

job_operations = {(j, o): list(range(6)) for j in range(10) for o in range(6)}
job_operations.pop((3, 5))  # 工件3只有5道工序

processing_times = {
    (job, step, machine): np.random.randint(1, 11)
    for (job, step), machines in job_operations.items()
    for machine in machines
}

operations = list(job_operations.keys())
operation_to_idx = {op: idx for idx, op in enumerate(operations)}
num_operations = len(operations)


def build_disjunctive_graph():
    G = nx.DiGraph()
    for op in operations:
        job, step = op
        G.add_node(operation_to_idx[op], job=job, step=step, type='operation')
    for job in range(10):
        steps = [s for (j, s) in job_operations if j == job]
        steps.sort()
        for i in range(len(steps) - 1):
            src = (job, steps[i])
            dst = (job, steps[i + 1])
            G.add_edge(operation_to_idx[src], operation_to_idx[dst], type='sequence')
    machine_ops = {m: [] for m in range(num_machines)}
    for op, machines in job_operations.items():
        for m in machines:
            machine_ops[m].append(op)
    for m, ops in machine_ops.items():
        for i in range(len(ops)):
            for j in range(i + 1, len(ops)):
                u, v = operation_to_idx[ops[i]], operation_to_idx[ops[j]]
                G.add_edge(u, v, type='disjunctive', machine=m)
                G.add_edge(v, u, type='disjunctive', machine=m)
    return G


def build_simplified_disjunctive_graph():
    G = nx.DiGraph()
    for op in operations:
        job, step = op
        G.add_node(operation_to_idx[op], job=job, step=step, type='operation')
    for job in range(10):
        steps = [s for (j, s) in job_operations if j == job]
        steps.sort()
        for i in range(len(steps) - 1):
            u = (job, steps[i])
            v = (job, steps[i + 1])
            G.add_edge(operation_to_idx[u], operation_to_idx[v], type='sequence')
    return G


def build_operation_machine_bipartite():
    G = nx.Graph()
    for op in operations:
        job, step = op
        G.add_node(f"op_{operation_to_idx[op]}", type='operation', job=job, step=step)
    for m in range(num_machines):
        G.add_node(f"machine_{m}", type='machine', machine_id=m)
    for op, machines in job_operations.items():
        op_node = f"op_{operation_to_idx[op]}"
        for m in machines:
            G.add_edge(op_node, f"machine_{m}", weight=processing_times[(op[0], op[1], m)],
                       processing_time=processing_times[(op[0], op[1], m)])
    return G