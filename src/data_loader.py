# # # # -*- codeing = utf-8 -*-
# # # # @time ： 2025/8/9
# # # # @author : likun
# # # # @file : data_loader.py
# # # # @software : PyCharm
# -*- coding: utf-8 -*-
"""
data_loader.py

功能：
1. 为单井构造滑动窗口 DataLoader
2. 为多井合并构造滑动窗口 DataLoader
3. 兼容 top_features_per_well.csv:
   well_id, feature_1, feature_2, ...
4. 兼容 top_features_detailed.csv:
   well_id, rank, feature, importance_score
5. 支持两种预处理文件：
   - 单井文件：preprocessed_data/Sheet1_preprocessed.csv
   - 合并文件：preprocessed_data.csv 或 all_wells_preprocessed.csv
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset


# =========================
# Dataset
# =========================
class WellSequenceDataset(Dataset):
    """
    滑动窗口数据集。
    X_windows: (N, seq_len, C)
    y_windows: (N, pred_len)
    """
    def __init__(self, X_windows, y_windows):
        assert len(X_windows) == len(y_windows)
        self.X = X_windows
        self.y = y_windows

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )


# =========================
# 工具函数
# =========================
def _normalize_date_column(df):
    """
    统一时间列为 date。
    """
    df = df.copy()
    if "date" not in df.columns:
        date_candidates = [
            c for c in df.columns
            if "date" in str(c).lower()
            or "time" in str(c).lower()
            or "时间" in str(c)
            or "年月" in str(c)
            or "年" in str(c)
        ]
        if len(date_candidates) > 0:
            df = df.rename(columns={date_candidates[0]: "date"})
        else:
            df["date"] = np.arange(len(df)).astype(str)
    return df


def _normalize_well_column(df):
    """
    统一井号列为 well。
    """
    df = df.copy()
    if "well" not in df.columns:
        for cand in ["well_id", "井号", "井名", "井"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "well"})
                break
    return df


def _read_top_features(top_features_path, well_id):
    """
    从 top_features_per_well.csv 或 top_features_detailed.csv 中读取某口井的特征列表。

    支持格式 1:
        well_id, feature_1, feature_2, feature_3 ...

    支持格式 2:
        well_id, rank, feature, importance_score
    """
    if not os.path.exists(top_features_path):
        raise FileNotFoundError(f"找不到特征文件: {top_features_path}")

    features_df = pd.read_csv(top_features_path, encoding="utf-8-sig")

    if "well_id" not in features_df.columns:
        # 兼容旧格式
        if "well" in features_df.columns:
            features_df = features_df.rename(columns={"well": "well_id"})
        elif "井号" in features_df.columns:
            features_df = features_df.rename(columns={"井号": "well_id"})
        else:
            return []

    # 匹配井号
    row_df = features_df[features_df["well_id"].astype(str) == str(well_id)]
    if row_df.empty:
        row_df = features_df[
            features_df["well_id"].astype(str).str.lower() == str(well_id).lower()
        ]

    if row_df.empty:
        return []

    # 格式 1: feature_1, feature_2...
    feature_cols = [
        c for c in features_df.columns
        if str(c).lower().startswith("feature_")
    ]

    if len(feature_cols) > 0:
        row = row_df.iloc[0]
        feats = []
        for c in feature_cols:
            v = row.get(c, None)
            if pd.isna(v):
                continue
            v = str(v).strip()
            if v != "":
                feats.append(v)
        return feats

    # 格式 2: detailed format
    if "feature" in features_df.columns:
        sub = row_df.copy()
        if "rank" in sub.columns:
            sub = sub.sort_values("rank")
        return sub["feature"].astype(str).tolist()

    # 兼容旧 top1, top2...
    old_cols = [
        c for c in features_df.columns
        if str(c).lower().startswith("top")
    ]
    if len(old_cols) > 0:
        row = row_df.iloc[0]
        feats = []
        for c in old_cols:
            v = row.get(c, None)
            if pd.isna(v):
                continue
            v = str(v).strip()
            if v != "":
                feats.append(v)
        return feats

    return []


def _resolve_column_name(df_cols, requested_name):
    """
    在实际数据列中匹配特征名。
    """
    df_cols = list(df_cols)

    if requested_name in df_cols:
        return requested_name

    low_map = {str(c).lower(): c for c in df_cols}
    if str(requested_name).lower() in low_map:
        return low_map[str(requested_name).lower()]

    synonyms = {
        "月天数": "month_days",
        "month_days": "month_days",
        "月份": "month_idx",
        "month_idx": "month_idx",
        "month_sin": "month_sin",
        "month_cos": "month_cos",
        "月产油量": "月产油量",
        "月产油量_log": "月产油量_log",
    }

    if requested_name in synonyms and synonyms[requested_name] in df_cols:
        return synonyms[requested_name]

    key = str(requested_name).lower()
    if key in synonyms and synonyms[key] in df_cols:
        return synonyms[key]

    return None


def _get_target_col(df):
    """
    获取目标列。
    """
    if "月产油量_log" in df.columns:
        return "月产油量_log"
    if "月产油量" in df.columns:
        return "月产油量"
    raise ValueError("数据中没有目标列：月产油量_log 或 月产油量")


def _get_input_features(df, top_features_path, well_id, target_col):
    """
    获取输入特征列表。
    """
    selected = _read_top_features(top_features_path, well_id)

    # 如果没有 top 特征，使用候选特征回退
    if len(selected) == 0:
        selected = [
            c for c in [
                "地层压力", "井底压力", "套压", "油压", "回压",
                "month_days", "month_sin", "month_cos", "month_idx"
            ]
            if c in df.columns
        ]

    input_feats = []
    for feat in selected:
        # 防止目标泄漏
        if feat in ["月产油量", "月产油量_log", target_col]:
            continue

        resolved = _resolve_column_name(df.columns, feat)
        if resolved is not None:
            if resolved not in ["月产油量", "月产油量_log", target_col]:
                if resolved not in input_feats:
                    input_feats.append(resolved)

    # 如果仍为空，回退到所有数值列
    if len(input_feats) == 0:
        drop_cols = {"date", "dataset", "well", "well_id", "井号", "月产油量", "月产油量_log", target_col}
        numeric_cols = [
            c for c in df.columns
            if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
        input_feats = numeric_cols

    if len(input_feats) == 0:
        raise ValueError(f"井 {well_id} 没有可用输入特征")

    return input_feats


def _make_windows(X_all, y_all, seq_len, pred_len):
    """
    生成滑动窗口。
    """
    T = len(X_all)
    window_start_max = T - seq_len - pred_len + 1

    if window_start_max <= 0:
        return (
            np.zeros((0, seq_len, X_all.shape[1]), dtype=np.float32),
            np.zeros((0, pred_len), dtype=np.float32)
        )

    X_windows, y_windows = [], []

    for i in range(window_start_max):
        X_windows.append(X_all[i:i + seq_len])
        y_windows.append(y_all[i + seq_len:i + seq_len + pred_len])

    return (
        np.asarray(X_windows, dtype=np.float32),
        np.asarray(y_windows, dtype=np.float32)
    )


def _split_windows_by_time(X_windows, y_windows, T, seq_len, pred_len, train_ratio, val_ratio):
    """
    按原始时间边界划分窗口。
    """
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_end = max(train_end, seq_len + 1)
    val_end = max(val_end, train_end + 1)

    if val_end >= T:
        val_end = min(T - pred_len, train_end + 1)

    train_idx_end = max(0, train_end - seq_len - pred_len + 1)
    val_idx_end = max(0, val_end - seq_len - pred_len + 1)

    X_train = X_windows[:train_idx_end]
    y_train = y_windows[:train_idx_end]

    X_val = X_windows[train_idx_end:val_idx_end]
    y_val = y_windows[train_idx_end:val_idx_end]

    X_test = X_windows[val_idx_end:]
    y_test = y_windows[val_idx_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


# =========================
# 单井 DataLoader
# =========================
def create_dataloaders(
    preprocessed_path,
    top_features_path,
    well_id,
    batch_size=32,
    seq_len=12,
    pred_len=3,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
):
    """
    为单口井构造 train/val/test DataLoader。

    preprocessed_path 可以是：
    1. 单井文件：preprocessed_data/Sheet1_preprocessed.csv
    2. 合并文件：preprocessed_data.csv
    """
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"找不到预处理文件: {preprocessed_path}")

    df_all = pd.read_csv(preprocessed_path, encoding="utf-8-sig")
    df_all = _normalize_date_column(df_all)
    df_all = _normalize_well_column(df_all)

    # 如果有 well 列，按井筛选；否则认为整个文件就是单井
    if "well" in df_all.columns:
        df_well = df_all[df_all["well"].astype(str) == str(well_id)].copy()
        if df_well.empty:
            df_well = df_all[
                df_all["well"].astype(str).str.lower() == str(well_id).lower()
            ].copy()
        if df_well.empty:
            raise ValueError(f"在 {preprocessed_path} 中找不到井 {well_id}")
    else:
        df_well = df_all.copy()

    df_well = df_well.sort_values("date").reset_index(drop=True)

    T = len(df_well)
    if T < seq_len + pred_len + 1:
        raise ValueError(
            f"井 {well_id} 数据太短，T={T}, 需要至少 {seq_len + pred_len + 1}"
        )

    target_col = _get_target_col(df_well)
    input_feats = _get_input_features(df_well, top_features_path, well_id, target_col)

    X_all = df_well[input_feats].astype(float).values
    y_all = df_well[target_col].astype(float).values

    # 如果输入特征维度太低，补 0，保证模型稳定
    if X_all.shape[1] < 2:
        pad = np.zeros((X_all.shape[0], 2 - X_all.shape[1]), dtype=X_all.dtype)
        X_all = np.concatenate([X_all, pad], axis=1)

    # 时间划分边界
    train_end = int(T * train_ratio)
    train_end = max(train_end, seq_len + 1)

    # 标准化只使用训练段
    X_train_raw = X_all[:train_end]
    mean = X_train_raw.mean(axis=0, keepdims=True)
    std = X_train_raw.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_all_scaled = (X_all - mean) / std

    # 生成窗口
    X_windows, y_windows = _make_windows(X_all_scaled, y_all, seq_len, pred_len)

    # 划分窗口
    X_train, y_train, X_val, y_val, X_test, y_test = _split_windows_by_time(
        X_windows, y_windows, T, seq_len, pred_len, train_ratio, val_ratio
    )

    if len(X_train) == 0:
        raise ValueError(f"井 {well_id} 训练窗口为空")

    train_ds = WellSequenceDataset(X_train, y_train)
    val_ds = WellSequenceDataset(X_val, y_val)
    test_ds = WellSequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = X_all.shape[1]
    return train_loader, val_loader, test_loader, input_dim


# =========================
# 多井合并 DataLoader
# =========================
def create_combined_dataloaders(
    preprocessed_path,
    top_features_path,
    well_ids=None,
    batch_size=32,
    seq_len=12,
    pred_len=3,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    min_windows_total=50,
    max_wells=None
):
    """
    多井合并 DataLoader，用于 WOA 快速评估。
    """
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"找不到合并预处理文件: {preprocessed_path}")

    df_all = pd.read_csv(preprocessed_path, encoding="utf-8-sig")
    df_all = _normalize_date_column(df_all)
    df_all = _normalize_well_column(df_all)

    if "well" not in df_all.columns:
        raise ValueError("合并数据必须包含 well 列")

    if well_ids is None:
        well_ids = df_all["well"].astype(str).unique().tolist()
    else:
        well_ids = [str(w) for w in well_ids]

    if max_wells is not None and len(well_ids) > max_wells:
        import random
        well_ids = random.sample(well_ids, max_wells)

    # 先收集每口井的输入特征
    per_well_feats = {}
    union_features = []

    for well_id in well_ids:
        df_w = df_all[df_all["well"].astype(str) == str(well_id)].copy()
        if df_w.empty:
            continue

        df_w = df_w.sort_values("date").reset_index(drop=True)

        try:
            target_col = _get_target_col(df_w)
            feats = _get_input_features(df_w, top_features_path, well_id, target_col)
        except Exception:
            continue

        per_well_feats[well_id] = feats

        for f in feats:
            if f not in union_features:
                union_features.append(f)

    if len(per_well_feats) == 0:
        raise ValueError("没有可用井用于合并训练")

    if len(union_features) == 0:
        raise ValueError("合并特征集合为空")

    combined_X_train, combined_y_train = [], []
    combined_X_val, combined_y_val = [], []
    combined_X_test, combined_y_test = [], []

    for well_id, feats in per_well_feats.items():
        df_w = df_all[df_all["well"].astype(str) == str(well_id)].copy()
        df_w = df_w.sort_values("date").reset_index(drop=True)

        T = len(df_w)
        if T < seq_len + pred_len + 1:
            continue

        try:
            target_col = _get_target_col(df_w)
        except Exception:
            continue

        # 按 union_features 构造 X，缺失列补 0
        X_cols = []
        for f in union_features:
            if f in df_w.columns:
                X_cols.append(df_w[f].astype(float).values.reshape(-1, 1))
            else:
                X_cols.append(np.zeros((T, 1), dtype=float))

        X_all = np.concatenate(X_cols, axis=1)
        y_all = df_w[target_col].astype(float).values

        train_end = int(T * train_ratio)
        train_end = max(train_end, seq_len + 1)

        X_train_raw = X_all[:train_end]
        mean = X_train_raw.mean(axis=0, keepdims=True)
        std = X_train_raw.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0

        X_scaled = (X_all - mean) / std

        X_windows, y_windows = _make_windows(X_scaled, y_all, seq_len, pred_len)

        X_train, y_train, X_val, y_val, X_test, y_test = _split_windows_by_time(
            X_windows, y_windows, T, seq_len, pred_len, train_ratio, val_ratio
        )

        if len(X_train) > 0:
            combined_X_train.append(X_train)
            combined_y_train.append(y_train)

        if len(X_val) > 0:
            combined_X_val.append(X_val)
            combined_y_val.append(y_val)

        if len(X_test) > 0:
            combined_X_test.append(X_test)
            combined_y_test.append(y_test)

    def _concat_x(lst):
        if len(lst) == 0:
            return np.zeros((0, seq_len, len(union_features)), dtype=np.float32)
        return np.concatenate(lst, axis=0).astype(np.float32)

    def _concat_y(lst):
        if len(lst) == 0:
            return np.zeros((0, pred_len), dtype=np.float32)
        return np.concatenate(lst, axis=0).astype(np.float32)

    X_train = _concat_x(combined_X_train)
    y_train = _concat_y(combined_y_train)

    X_val = _concat_x(combined_X_val)
    y_val = _concat_y(combined_y_val)

    X_test = _concat_x(combined_X_test)
    y_test = _concat_y(combined_y_test)

    total_windows = len(X_train) + len(X_val)

    if total_windows < min_windows_total:
        raise ValueError(
            f"合并后训练/验证窗口数太少: {total_windows} < {min_windows_total}"
        )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = len(union_features)
    return train_loader, val_loader, test_loader, input_dim
