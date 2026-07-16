import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import json

class PositionalEncoding(nn.Module):
    """位置编码层 (seq_len, batch_size, d_model格式)"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (seq_len, batch_size, d_model)
        Returns:
            (seq_len, batch_size, d_model) with positional encoding
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AnchorProjection(nn.Module):
    """Anchor投影层 - 将(53, 20)的anchor转换为(53, 10, 256)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 先将20维的anchor特征映射到hidden_size
        self.anchor_linear = nn.Sequential(
            nn.Linear(20, config.tf_de_dim),
            nn.ReLU(),
            nn.LayerNorm(config.tf_de_dim)
        )
        
        # 时间序列投影: 从1个时间步扩展到10个时间步
        self.temporal_proj = nn.Sequential(
            nn.Linear(config.tf_de_dim, config.tf_de_dim * 4),
            nn.ReLU(),
            nn.Linear(config.tf_de_dim * 4, config.tf_de_dim * 10),  # 输出10倍维度
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(config.tf_de_dim, max_len=10)

    def forward(self, anchors):
        """
        Args:
            anchors: (num_anchors, 20) - 原始anchor
        Returns:
            anchor_feats: (num_anchors, 10, 256) - 投影后的anchor特征
        """
        num_anchors = anchors.shape[0]
        
        # 1. 映射到hidden空间
        anchor_emb = self.anchor_linear(anchors)  # [53, 256]
        
        # 2. 时间序列扩展
        temporal_features = self.temporal_proj(anchor_emb)  # [53, 2560]
        temporal_features = temporal_features.view(num_anchors, 10, self.config.tf_de_dim)  # [53, 10, 256]
        
        # 3. 添加位置编码 (需要转换为seq_first格式)
        temporal_features_t = temporal_features.transpose(0, 1)  # [10, 53, 256]
        temporal_features_t = self.pos_encoding(temporal_features_t)  # [10, 53, 256]
        temporal_features = temporal_features_t.transpose(0, 1)  # [53, 10, 256]
        
        return temporal_features

class TrajFrontDoorEncoder(nn.Module):
    """
    前门编码器 - 处理GRU特征和anchor特征的交互
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.tf_de_dim  # 256

        self._create_anchors(self.config.prior_traj_path)
        
        # Anchor投影
        self.anchor_proj = AnchorProjection(config)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(self.hidden_size, max_len=10)
        
        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size, 
            num_heads=8,
            dropout=0.1,
            batch_first=False
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1)
        )
        
        # 门控机制
        self.aug_gate = nn.Linear(self.hidden_size, 1)
        self.ori_gate = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def _create_anchors(self, anchor_path=None):
        """
        Creates diverse trajectory anchors covering common driving maneuvers.
        When no valid path/file is found, creates a single random placeholder
        that will be replaced by the online DPMM once the buffer fills.
        """
        if anchor_path is None or not os.path.isfile(anchor_path):
            print(f'[TrajFrontDoor] No valid anchor path ({anchor_path}). '
                  f'Creating random placeholder (will be replaced by DPMM).')
            anchors = torch.randn(1, 20, dtype=torch.float32) * 0.01
            self.register_buffer('anchors', anchors)
            print(f'[TrajFrontDoor] Created {anchors.shape[0]} random placeholder anchors')
            return
        else:
            with open(anchor_path, 'r') as f:
                data = json.load(f)

            # Extract the first 20 elements of 'mu' from each cluster
            mu_list = [entry['mu'][:20] for entry in data]

            # Convert to a PyTorch tensor and register directly
            anchors = torch.tensor(mu_list, dtype=torch.float32)  # ✅ 第一次转换是 OK 的（list → tensor）
            self.register_buffer('anchors', anchors)  # ✅ 直接传 tensor，不要再用 torch.tensor()
            print(f'read anchors from {anchor_path}')
            # print(f'anchors shape: {torch.tensor(anchors, dtype=torch.float32).shape}')
            print(f"Created {len(anchors)} anchor trajectories")

    def forward(self, local_feats, local_mask=None):
        """
        Args:
            local_feats: (batch_size, seq_len=10, hidden_size=256) - GRU特征
            anchors: (num_anchors=53, anchor_dim=20) - 原始anchor
            local_mask: (batch_size, seq_len) - 局部特征mask
            
        Returns:
            enhanced_feats: (batch_size, seq_len=10, hidden_size=256) - 增强后的GRU特征
            gate_weights: (batch_size, seq_len=10) - 门控权重
        """
        batch_size, seq_len, hidden_size = local_feats.shape
        num_anchors, anchor_dim = self.anchors.shape
        
        # 1. 投影anchor到特征空间 (53, 20) -> (53, 10, 256)
        global_feats = self.anchor_proj(self.anchors)  # [53, 10, 256]
        
        # 2. 转换输入格式为seq_first
        local_feats_t = local_feats.transpose(0, 1)  # [10, 4, 256]
        global_feats_t = global_feats.transpose(0, 1)  # [10, 53, 256]
        
        # 3. 添加位置编码
        local_feats_t = self.pos_encoding(local_feats_t)  # [10, 4, 256]
        global_feats_t = self.pos_encoding(global_feats_t)  # [10, 53, 256]
        
        # 4. 局部特征自注意力
        self_attn_out, _ = self.self_attn(
            query=local_feats_t,      # [10, 4, 256]
            key=local_feats_t,        # [10, 4, 256]
            value=local_feats_t,      # [10, 4, 256]
            key_padding_mask=local_mask  # [4, 10]
        )
        self_attn_out = self.norm1(local_feats_t + self_attn_out)
        
        # 5. 交叉注意力：局部特征查询全局特征
        # 扩展局部特征以匹配全局特征的维度 [10, 4, 256] -> [10, 4*53, 256]
        local_expanded = self_attn_out.unsqueeze(2).repeat(1, 1, num_anchors, 1)
        local_expanded = local_expanded.view(seq_len, batch_size * num_anchors, hidden_size)
        
        # 扩展全局特征以匹配局部特征的维度 [10, 53, 256] -> [10, 4*53, 256]
        global_expanded = global_feats_t.unsqueeze(2).repeat(1, 1, batch_size, 1)
        global_expanded = global_expanded.view(seq_len, num_anchors * batch_size, hidden_size)
        
        # 交叉注意力
        cross_attn_out, cross_weights = self.cross_attn(
            query=local_expanded,     # [10, 4*53, 256]
            key=global_expanded,      # [10, 4*53, 256]
            value=global_expanded,    # [10, 4*53, 256]
        )
        
        # 6. 恢复原始维度并聚合anchor信息
        cross_attn_out = cross_attn_out.view(seq_len, batch_size, num_anchors, hidden_size)
        cross_attn_out = torch.mean(cross_attn_out, dim=2)  # [10, 4, 256] - 平均所有anchor
        cross_attn_out = self.norm2(self_attn_out + cross_attn_out)
        
        # 7. 前馈网络
        ffn_out = self.ffn(cross_attn_out)  # [10, 4, 256]
        ffn_out = self.norm3(cross_attn_out + ffn_out)
        
        # 8. 门控融合
        aug_weight = self.aug_gate(ffn_out)  # [10, 4, 1]
        ori_weight = self.ori_gate(local_feats_t)  # [10, 4, 1]
        gate_weights = self.sigmoid(aug_weight + ori_weight)  # [10, 4, 1]
        
        # 9. 最终输出
        out_feats_t = gate_weights * ffn_out + (1 - gate_weights) * local_feats_t  # [10, 4, 256]
        
        # 10. 转换回batch_first格式
        enhanced_feats = out_feats_t.transpose(0, 1)  # [4, 10, 256]
        gate_weights = gate_weights.squeeze(-1).transpose(0, 1)  # [4, 10]
        
        return enhanced_feats, gate_weights

    @torch.no_grad()
    def update_anchors(self, new_anchors: torch.Tensor):
        """Hot-swap trajectory anchors with new DPMM cluster means.

        Args:
            new_anchors: (num_anchors, 20) tensor of new trajectory anchors.
        """
        if new_anchors.shape[1] != self.anchors.shape[1]:
            raise ValueError(
                f"Anchor dim mismatch: new={new_anchors.shape[1]}, "
                f"existing={self.anchors.shape[1]}"
            )
        device = self.anchors.device
        self.anchors = new_anchors.to(device)
        print(f"[TrajFrontDoor] Updated anchors: {new_anchors.shape[0]} anchors")