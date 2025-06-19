
import os
import torch
import torch.nn.functional as F
import numpy as np
import dgl
from models.gat_model import MultiGraphGAT
from graphs.build_graphs import build_disjunctive_graph, build_simplified_disjunctive_graph, build_operation_machine_bipartite, encode_edge_types
from visualize import plot_and_save_graph, visualize_features

save_path = "scheduling_graphs"
os.makedirs(save_path, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# 构建图结构
dg = build_disjunctive_graph()
sog = build_simplified_disjunctive_graph()
omb = build_operation_machine_bipartite()
plot_and_save_graph(dg, "析取图", "dg.png")
plot_and_save_graph(sog, "工序顺序图", "sog.png")
plot_and_save_graph(omb, "二分图", "omb.png", node_size=500)

# 构建DGL图
dg = encode_edge_types(dg)
sog = encode_edge_types(sog)
omb = omb.to_directed()
dgl_dg = dgl.add_self_loop(dgl.from_networkx(dg, edge_attrs=['type', 'machine']))
dgl_sog = dgl.add_self_loop(dgl.from_networkx(sog, edge_attrs=['type']))
dgl_omb = dgl.add_self_loop(dgl.from_networkx(omb, edge_attrs=['weight', 'processing_time']))

from graphs.build_graphs import job_operations, processing_times, operation_to_idx
num_operations = len(operation_to_idx)
features = torch.zeros(num_operations, 5)
labels = []
for op, idx in operation_to_idx.items():
    job, step = op
    times = [processing_times[(job, step, m)] for m in job_operations[op]]
    features[idx] = torch.tensor([job, step, min(times), max(times), sum(times)/len(times)])
    labels.append(job)

omb_features = torch.cat([features, torch.zeros(dgl_omb.number_of_nodes() - num_operations, 5)])

# 模型与训练
model = MultiGraphGAT(in_feats=5, hidden_size=16, num_heads=2, num_operations=num_operations)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
targets = torch.randn(num_operations, num_operations)

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out, fused = model(dgl_dg, dgl_sog, dgl_omb, omb_features)
    loss = F.mse_loss(out, targets)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    _, fused = model(dgl_dg, dgl_sog, dgl_omb, omb_features)
visualize_features(fused, labels, "GAT特征可视化", "gat_tsne.png")
print("训练与可视化完成。")
