# -*- codeing = utf-8 -*-
# @time ： 2025/8/11 22:46
# @author : likun
# @file : all_wells_metrics.py
# @software : PyCharm
# -*- coding: utf-8 -*-
# train_all_wells_test.py
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
all_wells_metrics.py

功能：
- 批量测试所有已经训练好的井模型；
- 每口井：
    1. 加载对应模型；
    2. 在测试集上预测；
    3. 保存 CSV；
    4. 保存预测曲线图；
    5. 打印 log 空间与 exp 空间指标。

注意：
- preprocess.py 中使用的是 np.log1p(月产油量)
- 因此反变换必须使用 np.expm1()
"""

import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import FusionModel
from data_loader import create_dataloaders

warnings.filterwarnings("ignore")


# =========================
# 配置
# =========================
SAVE_DIR = "results"
os.makedirs(SAVE_DIR, exist_ok=True)

PREPROCESSED_DIR = "preprocessed_data"
TOP_FEATURES_PATH = "top_features_per_well.csv"

SEQ_LEN = 12
PRED_LEN = 3
BATCH_SIZE = 32

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 工具函数
# =========================
def read_well_ids(top_features_path):
    """
    从 top_features_per_well.csv 中读取井号。
    """
    if not os.path.exists(top_features_path):
        raise FileNotFoundError(f"找不到特征文件：{top_features_path}")

    df = pd.read_csv(top_features_path, encoding="utf-8-sig")

    for col in ["well_id", "well", "井号", "井名"]:
        if col in df.columns:
            return df[col].astype(str).unique().tolist()

    raise ValueError(
        f"{top_features_path} 中找不到井号列，"
        f"需要 well_id / well / 井号 / 井名 中的一个。"
    )


def safe_load_state(path):
    """
    兼容不同 PyTorch 版本的权重加载。
    """
    try:
        return torch.load(path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=DEVICE)


def calc_metrics(y_true, y_pred):
    """
    计算 MSE、RMSE、MAE、MAPE、R²。
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    mask = np.abs(y_true) > 1e-8

    if mask.sum() == 0:
        mape = np.nan
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    return mse, rmse, mae, mape, r2


def infer_model_config_from_state(state, input_dim_from_data):
    """
    根据 state_dict 推断 FusionModel 的关键结构参数。
    """
    # input_dim / d_model
    if "value_embedding.weight" in state:
        d_model = state["value_embedding.weight"].shape[0]
        input_dim = state["value_embedding.weight"].shape[1]
    elif "cnn_branch.net.0.weight" in state:
        w = state["cnn_branch.net.0.weight"]
        d_model = w.shape[0]
        input_dim = w.shape[1]
    else:
        d_model = 64
        input_dim = input_dim_from_data

    # pred_len
    if "fusion_head.3.weight" in state:
        pred_len = state["fusion_head.3.weight"].shape[0]
    elif "fc_out.weight" in state:
        pred_len = state["fc_out.weight"].shape[0]
    else:
        pred_len = PRED_LEN

    # d_ff
    if "times_branch.ffn.0.weight" in state:
        d_ff = state["times_branch.ffn.0.weight"].shape[0]
    else:
        d_ff = max(128, d_model * 4)

    # cnn_kernel
    if "cnn_branch.net.0.weight" in state:
        cnn_kernel = state["cnn_branch.net.0.weight"].shape[2]
    else:
        cnn_kernel = 3

    # num_heads 无法从 MultiheadAttention 权重直接可靠推断
    # 用 1 最稳，不影响权重 shape 加载
    num_heads = 1

    return {
        "input_dim": int(input_dim),
        "d_model": int(d_model),
        "d_ff": int(d_ff),
        "num_heads": int(num_heads),
        "cnn_kernel": int(cnn_kernel),
        "dropout": 0.1,
        "pred_len": int(pred_len)
    }


def get_test_dates(preprocessed_path, seq_len=12, pred_len=3, train_ratio=0.7, val_ratio=0.15):
    """
    复现 data_loader.py 里的时间切分逻辑，用于获得测试集日期。
    """
    df_w = pd.read_csv(preprocessed_path, encoding="utf-8-sig")

    if "date" not in df_w.columns:
        date_candidates = [
            c for c in df_w.columns
            if "date" in str(c).lower()
            or "time" in str(c).lower()
            or "时间" in str(c)
            or "年" in str(c)
        ]

        if len(date_candidates) > 0:
            df_w = df_w.rename(columns={date_candidates[0]: "date"})
        else:
            df_w["date"] = np.arange(len(df_w)).astype(str)

    df_w = df_w.sort_values("date").reset_index(drop=True)

    T = len(df_w)

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_end = max(train_end, seq_len + 1)
    val_end = max(val_end, train_end + 1)

    if val_end >= T:
        val_end = min(T - pred_len, train_end + 1)

    window_start_max = T - seq_len - pred_len + 1
    val_idx_end = max(0, val_end - seq_len - pred_len + 1)

    test_window_starts = list(range(val_idx_end, window_start_max))

    test_dates = []

    for i in test_window_starts:
        idx = i + seq_len
        if idx < len(df_w):
            test_dates.append(str(df_w["date"].iloc[idx]))

    return test_dates


# =========================
# 主流程
# =========================
def main():
    print(f"使用设备：{DEVICE}")

    well_ids = read_well_ids(TOP_FEATURES_PATH)

    print("待评估井号：")
    for w in well_ids:
        print(f"  - {w}")

    summary_rows = []

    for well_id in well_ids:
        print(f"\n===== 测试井: {well_id} =====")

        preprocessed_path = os.path.join(PREPROCESSED_DIR, f"{well_id}_preprocessed.csv")
        model_path = os.path.join(SAVE_DIR, f"best_model_well_{well_id}.pt")

        if not os.path.exists(preprocessed_path):
            print(f"[跳过] 找不到预处理文件: {preprocessed_path}")
            continue

        if not os.path.exists(model_path):
            print(f"[跳过] 找不到模型文件: {model_path}")
            continue

        # 加载 DataLoader
        try:
            train_loader, val_loader, test_loader, input_dim_from_data = create_dataloaders(
                preprocessed_path=preprocessed_path,
                top_features_path=TOP_FEATURES_PATH,
                well_id=well_id,
                batch_size=BATCH_SIZE,
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN
            )
        except Exception as e:
            print(f"[跳过] 井 {well_id} 数据加载失败: {e}")
            continue

        if len(test_loader.dataset) == 0:
            print(f"[跳过] 井 {well_id} 测试集为空")
            continue

        # 加载模型权重
        try:
            state = safe_load_state(model_path)
        except Exception as e:
            print(f"[跳过] 模型权重加载失败: {e}")
            continue

        # 推断模型结构
        cfg = infer_model_config_from_state(state, input_dim_from_data)

        net = FusionModel(
            input_dim=cfg["input_dim"],
            d_model=cfg["d_model"],
            d_ff=cfg["d_ff"],
            num_heads=cfg["num_heads"],
            cnn_kernel=cfg["cnn_kernel"],
            dropout=cfg["dropout"],
            pred_len=cfg["pred_len"]
        ).to(DEVICE)

        try:
            net.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"[WARN] strict=True 加载失败，尝试 strict=False。原因：{e}")
            net.load_state_dict(state, strict=False)

        net.eval()

        # 测试集预测
        all_preds = []
        all_trues = []

        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE).float()

                preds = net(xb)

                all_preds.append(preds.cpu().numpy())
                all_trues.append(yb.cpu().numpy())

        if len(all_preds) == 0:
            print("[WARN] 无预测结果，跳过。")
            continue

        all_preds = np.concatenate(all_preds, axis=0)
        all_trues = np.concatenate(all_trues, axis=0)

        # 取第一步预测
        pred_log = all_preds[:, 0].flatten()
        true_log = all_trues[:, 0].flatten()

        test_dates = get_test_dates(
            preprocessed_path,
            seq_len=SEQ_LEN,
            pred_len=PRED_LEN
        )

        N = min(len(test_dates), len(true_log), len(pred_log))

        test_dates = test_dates[:N]
        true_log = true_log[:N]
        pred_log = pred_log[:N]

        # 反变换：log1p -> expm1
        true_exp = np.expm1(true_log)
        pred_exp = np.expm1(pred_log)

        # 指标
        mse_log, rmse_log, mae_log, mape_log, r2_log = calc_metrics(true_log, pred_log)
        mse_exp, rmse_exp, mae_exp, mape_exp, r2_exp = calc_metrics(true_exp, pred_exp)

        # 保存 CSV
        df_out = pd.DataFrame({
            "date": test_dates,
            "true_log": true_log,
            "pred_log": pred_log,
            "true_exp": true_exp,
            "pred_exp": pred_exp
        })

        out_csv = os.path.join(SAVE_DIR, f"predictions_well_{well_id}.csv")
        df_out.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"[保存预测 CSV] {out_csv}")

        # 保存图像
        plt.figure(figsize=(10, 4))
        plt.plot(test_dates, true_exp, label="True")
        plt.plot(test_dates, pred_exp, label="Pred")
        plt.xticks(rotation=45)
        plt.xlabel("Date")
        plt.ylabel("Oil Production")
        plt.legend()
        plt.tight_layout()

        out_png = os.path.join(SAVE_DIR, f"plot_well_{well_id}.png")
        plt.savefig(out_png, dpi=300)
        plt.close()

        print(f"[保存图像] {out_png}")

        # 打印指标
        print(f"[{well_id}] 测试集指标 (log space):")
        print(f"  MSE:  {mse_log:.6f}")
        print(f"  RMSE: {rmse_log:.6f}")
        print(f"  MAE:  {mae_log:.6f}")
        print(f"  MAPE: {mape_log:.2f}%")
        print(f"  R²:   {r2_log:.6f}")

        print(f"[{well_id}] 测试集指标 (exp space):")
        print(f"  MSE:  {mse_exp:.6f}")
        print(f"  RMSE: {rmse_exp:.6f}")
        print(f"  MAE:  {mae_exp:.6f}")
        print(f"  MAPE: {mape_exp:.2f}%")
        print(f"  R²:   {r2_exp:.6f}")

        summary_rows.append({
            "well_id": well_id,
            "mse_log": mse_log,
            "rmse_log": rmse_log,
            "mae_log": mae_log,
            "mape_log": mape_log,
            "r2_log": r2_log,
            "mse_exp": mse_exp,
            "rmse_exp": rmse_exp,
            "mae_exp": mae_exp,
            "mape_exp": mape_exp,
            "r2_exp": r2_exp,
            "n_test_windows": N
        })

    # 保存汇总指标
    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(SAVE_DIR, "all_wells_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 汇总指标已保存：{summary_path}")
    else:
        print("\n[WARN] 没有生成任何井的评估结果。")


if __name__ == "__main__":
    main()