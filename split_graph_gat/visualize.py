import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import networkx as nx

def plot_and_save_graph(G, title, filename, save_path, pos=None, node_size=800):
    plt.figure(figsize=(12, 8))
    node_colors = []
    labels = {}

    for node in G.nodes():
        attrs = G.nodes[node]
        if attrs.get('type') == 'operation':
            node_colors.append('lightblue')
            labels[node] = f"J{attrs['job']}O{attrs['step']}"
        elif attrs.get('type') == 'machine':
            node_colors.append('lightgreen')
            labels[node] = f"M{attrs['machine_id']}"
        else:
            node_colors.append('lightgray')
            labels[node] = node

    edge_colors = []
    edge_styles = []
    for _, _, data in G.edges(data=True):
        edge_type = data.get('type', '')
        if edge_type == 'sequence':
            edge_colors.append('blue')
            edge_styles.append('solid')
        elif edge_type == 'disjunctive':
            edge_colors.append('red')
            edge_styles.append('dashed')
        else:
            edge_colors.append('gray')
            edge_styles.append('solid')

    if pos is None:
        pos = nx.spring_layout(G, k=0.5, iterations=50) if isinstance(G, nx.DiGraph) else nx.spring_layout(G)

    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, alpha=0.9)
    for i, (u, v) in enumerate(G.edges()):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=[edge_colors[i]], style=[edge_styles[i]], width=1.5)
    nx.draw_networkx_labels(G, pos, labels, font_size=10)

    if G.edges and 'weight' in list(G.edges(data=True))[0][2]:
        edge_labels = {(u, v): f"{d['processing_time']}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()


def visualize_features(features, labels, title, filename, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features.detach().cpu().numpy())
    plt.figure(figsize=(10, 8))

    jobs = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(jobs)))
    color_map = {job: colors[i] for i, job in enumerate(jobs)}

    for i in range(len(features_2d)):
        job = labels[i]
        plt.scatter(features_2d[i, 0], features_2d[i, 1], color=color_map[job], label=f'J{job}' if i == job else "")

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[j], markersize=10, label=f'J{j}')
               for j in jobs]
    plt.legend(handles=handles, title="工件")
    plt.title(title)
    plt.xlabel("t-SNE特征1")
    plt.ylabel("t-SNE特征2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename), dpi=300)
    plt.close()