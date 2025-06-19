
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GATConv

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

    def forward(self, dg, sog, omb, features):
        # 析取图特征
        x_dg = F.elu(self.dg_gat1(dg, features))
        x_dg = x_dg.view(-1, x_dg.size(1) * x_dg.size(2))
        x_dg = F.elu(self.dg_gat2(dg, x_dg)).squeeze(1)
        # 工序图特征
        x_sog = F.elu(self.sog_gat1(sog, features))
        x_sog = x_sog.view(-1, x_sog.size(1) * x_sog.size(2))
        x_sog = F.elu(self.sog_gat2(sog, x_sog)).squeeze(1)
        # 二分图特征
        x_omb = F.elu(self.omb_gat1(omb, features))
        x_omb = x_omb.view(-1, x_omb.size(1) * x_omb.size(2))
        x_omb = F.elu(self.omb_gat2(omb, x_omb)).squeeze(1)
        # 只取操作节点特征
        x_dg = x_dg[:self.num_operations]
        x_sog = x_sog[:self.num_operations]
        x_omb = x_omb[:self.num_operations]
        # 特征融合
        x_fused = torch.cat((x_dg, x_sog, x_omb), dim=1)
        x_fused = F.elu(self.fc(x_fused))
        out = self.output(x_fused)
        return out, x_fused
