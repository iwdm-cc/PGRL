# train.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
print(torch.__config__.show())

from dataset import (
    parse_mk_file,
    build_disjunction_graph,
    build_operation_sequence_graph,
    build_bipartite_graph
)
from model import MultiGraphFusion

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 设置中文字体和负号正常显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_embeddings(embeddings, save_path=None):
    """
    单纯无标签的t-SNE可视化
    """
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    plt.scatter(emb_2d[:,0], emb_2d[:,1], s=20)
    plt.title('t-SNE Visualization of Operation Embeddings')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] 嵌入可视化保存到 {save_path}")
    plt.show()

def visualize_embeddings_with_labels(embeddings, labels, save_path):
    """
    带节点类别标签的t-SNE可视化，节点颜色区分不同类别
    """
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_labels))

    for i, lbl in enumerate(unique_labels):
        idx = labels == lbl
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                    color=colors(i), label=f'类别 {lbl}', alpha=0.7, s=30)

    plt.legend()
    plt.title("t-SNE 可视化（带颜色区分）")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"[INFO] 嵌入可视化保存到 {save_path}")

def main():
    # 1. 读取数据与构建图结构
    mk_path = './data/BrandimarteMk1.fjs'  # 请替换为实际数据路径
    job_data = parse_mk_file(mk_path)

    data_disjunction = build_disjunction_graph(job_data)
    data_sequence = build_operation_sequence_graph(job_data)
    data_bipartite = build_bipartite_graph(job_data)

    # 2. 初始化模型和设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiGraphFusion(
        in_channels_dis=data_disjunction.x.shape[1],
        in_channels_seq=data_sequence.x.shape[1],
        in_channels_bip=data_bipartite.x.shape[1],
        hidden_channels=64,
        fusion_mode='concat'
    ).to(device)

    data_disjunction = data_disjunction.to(device)
    data_sequence = data_sequence.to(device)
    data_bipartite = data_bipartite.to(device)

    # 3. 推理得到工序节点嵌入
    model.eval()
    with torch.no_grad():
        op_embeddings = model(data_disjunction, data_sequence, data_bipartite)
    print(f"[INFO] 工序嵌入 shape: {op_embeddings.shape}")

    embeddings = op_embeddings.cpu().numpy()

    # 4. 嵌入数据统计信息
    print(f"[INFO] 嵌入均值：{embeddings.mean(axis=0)}")
    print(f"[INFO] 嵌入标准差：{embeddings.std(axis=0)}")
    print(f"[INFO] 前10个节点范数：{np.linalg.norm(embeddings, axis=1)[:10]}")

    # 5. 保存嵌入结果
    out_path = './outputs'
    os.makedirs(out_path, exist_ok=True)
    np.save(os.path.join(out_path, 'mk01_op_embeddings.npy'), embeddings)
    print(f"[INFO] 嵌入已保存至 {out_path}/mk01_op_embeddings.npy")

    # 6. 可视化（无标签）
    visualize_embeddings(embeddings, save_path=os.path.join(out_path, 'mk01_op_embeddings_tsne.png'))

    # 7. 可视化（带标签）——示例：模拟工序所属作业标签，每10个工序分为一组
    num_nodes = embeddings.shape[0]
    node_labels = np.array([i // 10 for i in range(num_nodes)])  # 示例标签

    visualize_embeddings_with_labels(embeddings, node_labels, save_path=os.path.join(out_path, 'mk01_op_embeddings_tsne_color.png'))

    # 8. 强化学习模块接口（预留）
    # TODO: 将 op_embeddings 作为状态输入，调用 RL 策略模块，执行调度动作
    # 示例：
    # action = rl_policy.select_action(state_embed=op_embeddings)
    # reward = env.step(action)
    # print(f"[INFO] 采取动作: {action}, 获得奖励: {reward}")

if __name__ == '__main__':
    main()
