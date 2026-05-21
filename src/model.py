# # # # # -*- codeing = utf-8 -*-
# # # # # @time ： 2025/8/10
# # # # # @author : likun
# # # # # @file : model.py
# # # # # @software : PyCharm
# # # # import torch
# # # # import torch.nn as nn
# # # # import torch.optim as optim
# # # # from copy import deepcopy
# # # # import numpy as np
# # # # import random
# -*- coding: utf-8 -*-
"""
model.py
- CNN: 局部时序特征提取
- TimesNet: FFT周期感知 + 周期卷积建模
- iTransformer: 变量维 inverted attention
- 保留 FusionModel / CNNBaseline / iTransformerBaseline / CNNiTransformerBaseline
"""

import math
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# 工具函数
# =========================
def _max_valid_heads(d_model, max_heads_candidate):
    max_h = min(int(max_heads_candidate), int(d_model))
    for h in range(max_h, 0, -1):
        if d_model % h == 0:
            return h
    return 1


def _ensure_valid_params(params, search_space):
    clipped = {}
    for k, v in params.items():
        if k not in search_space:
            clipped[k] = int(v) if isinstance(v, (int, np.integer)) else float(v)
            continue
        low, high = search_space[k]
        if isinstance(low, int) and isinstance(high, int):
            val = int(round(v))
            val = max(low, min(high, val))
        else:
            val = float(v)
            val = max(float(low), min(float(high), val))
        clipped[k] = val

    if 'd_model' in clipped:
        clipped['d_model'] = int(clipped['d_model'])
    if 'd_ff' in clipped:
        clipped['d_ff'] = int(clipped['d_ff'])
    if 'num_heads' in clipped:
        clipped['num_heads'] = int(max(1, clipped['num_heads']))
    if 'cnn_kernel' in clipped:
        clipped['cnn_kernel'] = int(max(1, clipped['cnn_kernel']))
    if 'batch_size' in clipped:
        clipped['batch_size'] = int(max(1, clipped['batch_size']))
    if 'dropout' in clipped:
        clipped['dropout'] = float(clipped['dropout'])
    if 'learning_rate' in clipped:
        clipped['learning_rate'] = float(clipped['learning_rate'])

    if 'd_model' in clipped and 'num_heads' in clipped:
        d_model = clipped['d_model']
        nh = clipped['num_heads']
        valid_heads = _max_valid_heads(d_model, nh)
        clipped['num_heads'] = valid_heads

    return clipped


# =========================
# CNN 分支
# =========================
class TemporalCNNBranch(nn.Module):
    def __init__(self, input_dim, d_model, cnn_kernel=3, dropout=0.1):
        super().__init__()
        pad = cnn_kernel // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=cnn_kernel, padding=pad),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)      # [B, C, T]
        x = self.net(x)            # [B, d_model, T]
        x = self.pool(x).squeeze(-1)  # [B, d_model]
        return x


# =========================
# TimesNet 分支
# FFT找周期 + 周期卷积
# =========================
class TimesBlock(nn.Module):
    def __init__(self, d_model, d_ff, top_k=3, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.period_conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU()
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _detect_periods(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        # 用时间维均值做 FFT，得到 batch 级周期候选
        fft_amp = torch.abs(torch.fft.rfft(x.mean(dim=-1), dim=1)).mean(dim=0)  # [F]
        if fft_amp.numel() <= 1:
            return [1], torch.tensor([1.0], device=x.device)

        fft_amp[0] = 0
        k = min(self.top_k, fft_amp.numel() - 1)
        vals, idxs = torch.topk(fft_amp, k=k)

        periods = []
        for idx in idxs:
            # 频率索引 -> 周期近似
            period = max(1, T // int(idx.item()))
            periods.append(period)

        weights = torch.softmax(vals, dim=0)
        return periods, weights

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        base = x.mean(dim=1)  # [B, D]

        periods, weights = self._detect_periods(x)
        feats = []

        for p, w in zip(periods, weights):
            pad_len = (p - (T % p)) % p
            x_pad = F.pad(x, (0, 0, 0, pad_len))  # pad time dimension
            t_new = x_pad.shape[1] // p

            # [B, T, D] -> [B, D, t_new, p]
            z = x_pad.view(B, t_new, p, D).permute(0, 3, 1, 2).contiguous()
            z = self.period_conv(z)
            z = z.mean(dim=(2, 3))  # [B, D]
            feats.append(w * z)

        if len(feats) == 0:
            period_feat = torch.zeros_like(base)
        else:
            period_feat = torch.stack(feats, dim=0).sum(dim=0)

        out = self.norm(base + self.dropout(self.ffn(period_feat)))
        return out


# =========================
# iTransformer 分支
# 变量维 token + attention
# =========================
class InvertedAttentionBranch(nn.Module):
    def __init__(self, input_dim, d_model, num_heads=4, dropout=0.1):
        super().__init__()
        self.var_encoder = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        B, C, T = x.shape

        # 每个变量当作一个 token
        x = x.reshape(B * C, 1, T)
        x = self.var_encoder(x).squeeze(-1)  # [B*C, d_model]
        x = x.view(B, C, -1)                 # [B, C, d_model]

        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        x = x.mean(dim=1)  # [B, d_model]
        return x


# =========================
# 主模型：CNN + TimesNet + iTransformer
# =========================
class FusionModel(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, num_heads, cnn_kernel, dropout, pred_len):
        super().__init__()

        d_model = int(d_model)
        if d_model <= 0:
            raise ValueError("d_model must be positive")

        num_heads = int(num_heads)
        num_heads = _max_valid_heads(d_model, num_heads)

        # 输入特征映射
        self.value_embedding = nn.Linear(input_dim, d_model)

        # 三个分支
        self.cnn_branch = TemporalCNNBranch(input_dim, d_model, cnn_kernel=cnn_kernel, dropout=dropout)
        self.times_branch = TimesBlock(d_model, d_ff, top_k=3, dropout=dropout)
        self.itrans_branch = InvertedAttentionBranch(input_dim, d_model, num_heads=num_heads, dropout=dropout)

        # 融合输出
        self.fusion_head = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        # x: [B, T, C]
        x_emb = self.value_embedding(x)   # [B, T, d_model]

        cnn_feat = self.cnn_branch(x)            # [B, d_model]
        times_feat = self.times_branch(x_emb)    # [B, d_model]
        itr_feat = self.itrans_branch(x)         # [B, d_model]

        feat = torch.cat([cnn_feat, times_feat, itr_feat], dim=-1)
        out = self.fusion_head(feat)
        return out


# =========================
# 基线模型
# =========================
class CNNBaseline(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, num_heads, cnn_kernel, dropout, pred_len):
        super().__init__()
        self.branch = TemporalCNNBranch(input_dim, d_model, cnn_kernel=cnn_kernel, dropout=dropout)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.branch(x)
        return self.fc_out(x)


class iTransformerBaseline(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, num_heads, cnn_kernel, dropout, pred_len):
        super().__init__()
        self.branch = InvertedAttentionBranch(input_dim, d_model, num_heads=num_heads, dropout=dropout)
        self.fc_out = nn.Linear(d_model, pred_len)

    def forward(self, x):
        x = self.branch(x)
        return self.fc_out(x)


class CNNiTransformerBaseline(nn.Module):
    def __init__(self, input_dim, d_model, d_ff, num_heads, cnn_kernel, dropout, pred_len):
        super().__init__()
        self.cnn_branch = TemporalCNNBranch(input_dim, d_model, cnn_kernel=cnn_kernel, dropout=dropout)
        self.itr_branch = InvertedAttentionBranch(input_dim, d_model, num_heads=num_heads, dropout=dropout)
        self.fc_out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x):
        c = self.cnn_branch(x)
        i = self.itr_branch(x)
        return self.fc_out(torch.cat([c, i], dim=-1))


# =========================
# 训练 & 验证
# =========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device).float()
        y_batch = y_batch.to(device).float()

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        bs = x_batch.size(0)
        total_loss += loss.item() * bs
        total_samples += bs

    if total_samples == 0:
        return float('inf')
    return total_loss / total_samples


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            preds = model(x_batch)
            loss = criterion(preds, y_batch)

            bs = x_batch.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

    if total_samples == 0:
        return float('inf')
    return total_loss / total_samples

