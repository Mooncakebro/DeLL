import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json
import pickle
from utils import print_data_info


class FuseFeatFrontDoorEncoder(nn.Module):
    """
    fuse feat前门编码器 - 处理fuse feat特征和fuse feat anchor的交互
    fuse feat特征: (batch_size, 11, 256) 11个token序列
    fuse featanchor: (num_anchors, 11*256) 11*256维fuse feat向量
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.ff_de_dim if hasattr(config, 'ff_de_dim') else config.tf_de_dim
        
        # 读取fuse featanchor
        self._create_anchors(self.config.prior_fuseFeat_path)
        
        # Anchor投影层
        self.anchor_proj = nn.Sequential(
            nn.Linear(256*11, self.hidden_size),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_size)
        )
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 自注意力相关的归一化和前馈网络
        self.self_attn_norm = nn.LayerNorm(self.hidden_size)
        self.self_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1)
        )
        self.self_ffn_norm = nn.LayerNorm(self.hidden_size)
        
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 交叉注意力相关的归一化和前馈网络（新增缺失的层）
        self.cross_attn_norm = nn.LayerNorm(self.hidden_size)  
        self.cross_ffn = nn.Sequential(  # 原来的ffn
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1)
        )
        self.cross_ffn_norm = nn.LayerNorm(self.hidden_size)  
        
        # 门控机制（修正：添加缺失的ori_gate）
        self.aug_gate = nn.Linear(self.hidden_size, 1)
        self.ori_gate = nn.Linear(self.hidden_size, 1) 
        self.sigmoid = nn.Sigmoid()

    def _create_anchors(self, anchor_path_list=None):
        """
        从文件读取fuse featanchor
        Args:
            anchor_path_list: 存储fuse featanchor聚类结果的JSON文件路径列表
        """
        if anchor_path_list is None:
            print('No fuseFeat anchor path provided. Using random placeholder '
                  '(will be replaced by online DPMM).')
            anchors = torch.randn(1, 256 * 11) * 0.01
            self.register_buffer('anchors', anchors)
            print(f'[FuseFeatFrontDoor] Created 1 random placeholder anchor, shape: {anchors.shape}')
            return anchors
        
        # Handle both single path string and list of paths
        if isinstance(anchor_path_list, str):
            anchor_path_list = [anchor_path_list]
        
        all_clusters = []
        
        for anchor_path in anchor_path_list:
            if not os.path.isfile(anchor_path):
                print(f'[FuseFeatFrontDoor] File not found: {anchor_path}')
                continue
            with open(anchor_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract clusters from this file
            clusters = data.get('tracked_clusters', [])
            
            # If there are existing anchors, offset the cluster_ids to avoid collisions
            cluster_offset = max([c['cluster_id'] for c in all_clusters], default=-1) + 1
            for cluster in clusters:
                cluster['cluster_id'] = cluster['cluster_id'] + cluster_offset
            
            all_clusters.extend(clusters)
            
            print(f'Loaded {len(clusters)} anchors from {anchor_path}')
        
        if not all_clusters:
            print('No valid anchor data found. Using random placeholder anchors '
                  '(will be replaced by online DPMM).')
            anchors = torch.randn(1, 256 * 11) * 0.01
            self.register_buffer('anchors', anchors)
            print(f'[FuseFeatFrontDoor] Created 1 random placeholder anchor, shape: {anchors.shape}')
            return anchors
        
        # Sort all collected clusters by cluster_id
        all_clusters.sort(key=lambda x: x['cluster_id'])
        
        # Create feature tensors
        feat = [
            torch.tensor(c['mu'], dtype=torch.float32) 
            for c in all_clusters
        ]

        feat = torch.stack(feat)
        self.register_buffer('anchors', feat)
        
        print(f'Got {len(feat)} fuse feat anchors from {len(anchor_path_list)} file(s), shape: {feat.shape}')
        
        return feat

    def forward(self, fuseFeat_feats, fuseFeat_mask=None):
        """
        Args:
            fuseFeat_feats: (batch_size, 11, hidden_size) - fuse feat特征
            fuseFeat_mask: (batch_size, 11) - fuse feat特征mask（可选）
            
        Returns:
            enhanced_fuseFeat: (batch_size, 11, hidden_size) - 增强后的fuse feat特征
            gate_weights: (batch_size, 11) - 门控权重
        """
        batch_size, seq_len, hidden_size = fuseFeat_feats.shape
        num_anchors, anchor_dim = self.anchors.shape
        
        # 1. 转换输入格式为seq_first
        local_feats_t = fuseFeat_feats.transpose(0, 1)  # [11, batch_size, 256]
        
        # 2. 自注意力处理
        self_attn_out, _ = self.self_attn(
            query=local_feats_t,      # [11, batch_size, 256]
            key=local_feats_t,        # [11, batch_size, 256]
            value=local_feats_t,      # [11, batch_size, 256]
            key_padding_mask=fuseFeat_mask  # [batch_size, 11]
        )
        self_attn_out = self.self_attn_norm(local_feats_t + self_attn_out)
        
        # 3. 自注意力后的前馈网络
        ffn_out = self.self_ffn(self_attn_out)  # [11, batch_size, 256]
        self_attn_out = self.self_ffn_norm(self_attn_out + ffn_out)
        
        # 4. 投影fuse featanchor
        global_feats = self.anchor_proj(self.anchors)  # [num_anchors, 256]
        
        # 5. 扩展全局特征以匹配batch维度
        # 将global_feats扩展为[num_anchors, batch_size, 256]
        global_expanded = global_feats.unsqueeze(1)  # [num_anchors, 11, 256]
        global_expanded = global_expanded.repeat(1, batch_size, 1)  # [num_anchors, batch_size, 256]
        
        # 6. 交叉注意力
        # 注意：query和key/value的序列长度可以不同，但batch_size必须相同
        # query: [11, batch_size, 256] - 序列长度1
        # key/value: [num_anchors, batch_size, 256] - 序列长度num_anchors
        cross_attn_out, cross_weights = self.cross_attn(
            query=self_attn_out,      # [11, batch_size, 256]
            key=global_expanded,      # [num_anchors, batch_size, 256]
            value=global_expanded,    # [num_anchors, batch_size, 256]
        )
        
        # 7. 残差连接和归一化
        cross_attn_out = self.cross_attn_norm(self_attn_out + cross_attn_out)
        
        # 8. 交叉注意力后的前馈网络
        ffn_out = self.cross_ffn(cross_attn_out)  # [11, batch_size, 256]
        ffn_out = self.cross_ffn_norm(cross_attn_out + ffn_out)
        
        # 9. 门控融合
        aug_weight = self.aug_gate(ffn_out)  # [11, batch_size, 1]
        ori_weight = self.ori_gate(local_feats_t)  # [11, batch_size, 1]
        gate_weights = self.sigmoid(aug_weight + ori_weight)  # [11, batch_size, 1]
        
        # 10. 最终输出
        out_feats_t = gate_weights * ffn_out + (1 - gate_weights) * local_feats_t  # [11, batch_size, 256]
        
        # 11. 转换回batch_first格式
        enhanced_fuseFeat = out_feats_t.transpose(0, 1)  # [batch_size, 11, 256]
        gate_weights = gate_weights.squeeze(-1).transpose(0, 1)  # [batch_size, 11]
        
        return enhanced_fuseFeat, gate_weights

    @torch.no_grad()
    def update_anchors(self, new_anchors: torch.Tensor):
        """Hot-swap fused-feature anchors with new DPMM cluster means.

        Args:
            new_anchors: (num_anchors, 11*256) tensor of new fuseFeat anchors.
        """
        if new_anchors.shape[1] != self.anchors.shape[1]:
            raise ValueError(
                f"Anchor dim mismatch: new={new_anchors.shape[1]}, "
                f"existing={self.anchors.shape[1]}"
            )
        device = self.anchors.device
        self.anchors = new_anchors.to(device)
        print(f"[FuseFeatFrontDoor] Updated anchors: {new_anchors.shape[0]} anchors")


