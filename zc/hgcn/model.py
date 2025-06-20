import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GNNEncoder(nn.Module):
    """
    图神经网络编码器：基于GAT卷积
    适用于单一图结构的节点特征提取，如析取图、工序顺序图、二分图等
    """

    def __init__(self, in_channels, out_channels, heads=2, dropout=0.2):
        """
        初始化编码器
        参数：
            in_channels: 输入特征维度（节点特征维度）
            out_channels: 输出特征维度（每个头的输出维度）
            heads: 注意力头数量，默认2头多头注意力
            dropout: dropout概率，防止过拟合
        """
        super(GNNEncoder, self).__init__()
        # 第一层GATConv，输入维度 -> out_channels，使用多头注意力
        self.gat1 = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)
        # 第二层GATConv，整合多头输出，输出维度为out_channels，单头
        self.gat2 = GATConv(out_channels * heads, out_channels, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        前向传播
        参数：
            x: 节点特征矩阵，shape=[节点数, 输入特征维度]
            edge_index: 图的边索引，shape=[2, 边数]
        返回：
            节点的最终嵌入特征，shape=[节点数, out_channels]
        """
        # 第一层GAT卷积+激活+dropout
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # ELU激活函数，常用于GAT
        x = F.dropout(x, p=self.dropout, training=self.training)

        # 第二层GAT卷积，融合多头结果
        x = self.gat2(x, edge_index)
        return x


class MultiGraphFusion(nn.Module):
    """
    多图融合模块，将多个不同图结构的节点表示进行融合
    用于柔性作业车间调度中不同图的特征联合利用
    """

    def __init__(self, in_channels_dis, in_channels_seq, in_channels_bip, hidden_channels=32, fusion_mode='concat'):
        """
        初始化多图融合模型
        参数：
            in_channels: 输入节点特征维度
            hidden_channels: GAT输出特征维度
            fusion_mode: 融合方式，支持 'concat'（拼接） 或 'weighted_sum'（加权求和）
        """
        super(MultiGraphFusion, self).__init__()
        self.fusion_mode = fusion_mode
        self.hidden_channels = hidden_channels

        # 三个编码器分别处理不同图
        # 每个编码器的输入维度不同
        self.encoder_disjunction = GNNEncoder(in_channels_dis, hidden_channels)
        self.encoder_sequence = GNNEncoder(in_channels_seq, hidden_channels)
        self.encoder_bipartite = GNNEncoder(in_channels_bip, hidden_channels)

        # 如果选择加权融合，初始化三个可训练权重参数
        if fusion_mode == 'weighted_sum':
            self.alpha = nn.Parameter(torch.tensor([1 / 3, 1 / 3, 1 / 3]), requires_grad=True)

        # 融合后的特征通过MLP进一步映射
        fusion_output_dim = hidden_channels * 3 if fusion_mode == 'concat' else hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(fusion_output_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

    def forward(self, data_disjunction, data_sequence, data_bipartite):
        """
        前向传播，融合三种图的节点嵌入
        参数：
            data_disjunction: 析取图数据对象，含x和edge_index
            data_sequence: 工序顺序图数据对象
            data_bipartite: 机器-工序二分图数据对象
        返回：
            融合后的工序节点嵌入特征，shape=[工序节点数, hidden_channels]
        """
        # 分别提取三种图的节点嵌入

        x_dis = self.encoder_disjunction(data_disjunction.x, data_disjunction.edge_index)
        x_seq = self.encoder_sequence(data_sequence.x, data_sequence.edge_index)
        x_bip = self.encoder_bipartite(data_bipartite.x, data_bipartite.edge_index)

        def select_op_nodes(data, x):
            return x[data.node_type == 0] if hasattr(data, 'node_type') else x

        x_dis = select_op_nodes(data_disjunction, x_dis)
        x_seq = select_op_nodes(data_sequence, x_seq)
        x_bip = select_op_nodes(data_bipartite, x_bip)

        print(f"x_dis shape: {x_dis.shape}, x_seq shape: {x_seq.shape}, x_bip shape: {x_bip.shape}")

        # 融合前确保节点数对齐
        min_nodes = min(x_dis.size(0), x_seq.size(0), x_bip.size(0))
        x_dis = x_dis[:min_nodes]
        x_seq = x_seq[:min_nodes]
        x_bip = x_bip[:min_nodes]

        if self.fusion_mode == 'concat':
            x_fused = torch.cat([x_dis, x_seq, x_bip], dim=-1)
        elif self.fusion_mode == 'weighted_sum':
            alpha = F.softmax(self.alpha, dim=0)
            x_fused = alpha[0] * x_dis + alpha[1] * x_seq + alpha[2] * x_bip
        else:
            raise ValueError("Unsupported fusion_mode. Use 'concat' or 'weighted_sum'.")

        out = self.mlp(x_fused)
        return out
