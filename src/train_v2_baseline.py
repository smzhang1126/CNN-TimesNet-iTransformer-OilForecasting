# # -*- codeing = utf-8 -*-
# # @time ： 2025/8/25 17:07
# # @author : likun
# # @file : train_v2_baseline.py
# # @software : PyCharm
# # -*- coding: utf-8 -*-
# """
# train_v2_baselines.py
# 一次性训练并评估 FusionModel + CNNBaseline + iTransformerBaseline + CNNiTransformerBaseline
# - 自动确保预处理和特征选择已完成
# - 对每口井、每个模型独立运行 WOA（验证集 MSE 作为目标）
# - 保存每井最优模型与测试集指标（MSE / MAPE / R2）
# """
#
# import os
# import math
# import random
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# import preprocess
# import top_features_selector
# from data_loader import create_dataloaders
# from model import (
#     FusionModel, CNNBaseline, iTransformerBaseline, CNNiTransformerBaseline,
#     train_one_epoch, evaluate_model, woa_optimize
# )
#
# # ---------------- 全局配置 ----------------
# RAW_DATA_PATH = "init_data.xlsx"
# PREPROCESSED_PATH = "preprocessed_data.csv"
# TOP_FEATURES_PATH = "top_features_per_well.csv"
#
# RESULTS_ROOT = "results_baselines"    # 不覆盖你原来的 results 目录
# os.makedirs(RESULTS_ROOT, exist_ok=True)
#
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED = 42
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
#
# # 统一的窗口与训练控制
# SEQ_LEN = 12
# PRED_LEN = 3
# EPOCHS = 50
# EARLY_STOP_PATIENCE = 8
#
# # 与主实验一致的搜索空间（可按需调整）
# SEARCH_SPACE = {
#     "learning_rate": (1e-4, 5e-3),
#     "d_model": (16, 128),
#     "d_ff": (64, 512),
#     "dropout": (0.05, 0.5),
#     "num_heads": (1, 8),
#     "cnn_kernel": (2, 6),
#     "batch_size": (8, 64),
# }
#
# # ---------------- 指标函数 ----------------
# def calculate_mape(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     mask = y_true != 0
#     if mask.sum() == 0:
#         return np.nan
#     return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
#
# def calculate_r2(y_true, y_pred):
#     y_true = np.asarray(y_true, dtype=float)
#     y_pred = np.asarray(y_pred, dtype=float)
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     if ss_tot == 0:
#         return np.nan
#     return 1.0 - ss_res / ss_tot
#
# # ---------------- 模型集合 ----------------
# MODEL_REGISTRY = {
#     "FusionModel": FusionModel,
#     "CNNBaseline": CNNBaseline,
#     "iTransformerBaseline": iTransformerBaseline,
#     "CNNiTransformerBaseline": CNNiTransformerBaseline,
# }
#
# # ---------------- 确保预处理与特征文件 ----------------
# def ensure_data_ready():
#     if not os.path.exists(PREPROCESSED_PATH):
#         print("预处理文件不存在，开始运行 preprocess.preprocess_data ...")
#         preprocess.preprocess_data(RAW_DATA_PATH, PREPROCESSED_PATH)
#         print(f"已生成：{PREPROCESSED_PATH}")
#
#     if not os.path.exists(TOP_FEATURES_PATH):
#         print("Top 特征文件不存在，开始运行 top_features_selector.select_top_features_per_well ...")
#         top_features_selector.select_top_features_per_well(PREPROCESSED_PATH, top_n=5, output_path=TOP_FEATURES_PATH)
#         print(f"已生成：{TOP_FEATURES_PATH}")
#
# # ---------------- 训练与评估（单井、单模型） ----------------
# def run_one_model_on_one_well(model_name, model_cls, well_id):
#     """
#     对 (模型, 井) 运行：WOA -> 训练 -> 测试。保存模型与指标。
#     """
#     result_dir = os.path.join(RESULTS_ROOT, model_name, well_id)
#     os.makedirs(result_dir, exist_ok=True)
#
#     # --- 定义供 WOA 使用的评估函数（注意：按候选 batch_size 重新构造 DataLoader） ---
#     def eval_params(params):
#         d_model   = int(params.get("d_model", 32))
#         d_ff      = int(params.get("d_ff", 128))
#         num_heads = int(params.get("num_heads", 1))
#         cnn_ks    = int(params.get("cnn_kernel", 3))
#         dropout   = float(params.get("dropout", 0.1))
#         lr        = float(params.get("learning_rate", 1e-3))
#         bs        = int(params.get("batch_size", 16))
#
#         # 用候选的 batch_size 构建 DataLoaders
#         train_loader, val_loader, _, input_dim = create_dataloaders(
#             preprocessed_path=PREPROCESSED_PATH,
#             top_features_path=TOP_FEATURES_PATH,
#             well_id=well_id,
#             batch_size=bs,
#             seq_len=SEQ_LEN,
#             pred_len=PRED_LEN
#         )
#
#         if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
#             return float("inf")
#
#         # 构建模型与优化器
#         model = model_cls(
#             input_dim=input_dim,
#             d_model=d_model,
#             d_ff=d_ff,
#             num_heads=num_heads,
#             cnn_kernel=cnn_ks,
#             dropout=dropout,
#             pred_len=PRED_LEN
#         ).to(DEVICE)
#
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=lr)
#
#         # 轻量训练若干轮，返回验证集 MSE 作为适应度
#         try:
#             for _ in range(3):
#                 train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
#             val_loss = evaluate_model(model, val_loader, criterion, DEVICE)
#             return float(val_loss)
#         except Exception as e:
#             print(f"[{model_name}-{well_id}] 评估异常：{e}")
#             return float("inf")
#
#     # --- 运行 WOA 搜索 ---
#     best_params, best_fitness = woa_optimize(
#         train_func=lambda p, **kw: eval_params(p),
#         search_space=SEARCH_SPACE,
#         pop_size=8,
#         max_iter=12
#     )
#     print(f"[{model_name}-{well_id}] 最优验证集 MSE：{best_fitness:.6f}")
#     print(f"[{model_name}-{well_id}] 最优参数：{best_params}")
#
#     # --- 用最优参数重新构建 DataLoaders & 模型，完整训练 + 早停 ---
#     bs = int(best_params["batch_size"])
#     lr = float(best_params["learning_rate"])
#     d_model   = int(best_params["d_model"])
#     d_ff      = int(best_params["d_ff"])
#     num_heads = int(best_params["num_heads"])
#     cnn_ks    = int(best_params["cnn_kernel"])
#     dropout   = float(best_params["dropout"])
#
#     train_loader, val_loader, test_loader, input_dim = create_dataloaders(
#         preprocessed_path=PREPROCESSED_PATH,
#         top_features_path=TOP_FEATURES_PATH,
#         well_id=well_id,
#         batch_size=bs,
#         seq_len=SEQ_LEN,
#         pred_len=PRED_LEN
#     )
#
#     model = model_cls(
#         input_dim=input_dim,
#         d_model=d_model,
#         d_ff=d_ff,
#         num_heads=num_heads,
#         cnn_kernel=cnn_ks,
#         dropout=dropout,
#         pred_len=PRED_LEN
#     ).to(DEVICE)
#
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#
#     best_val = float("inf")
#     best_wts = None
#     patience = 0
#
#     for ep in range(1, EPOCHS + 1):
#         tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
#         va_loss = evaluate_model(model, val_loader, criterion, DEVICE)
#         print(f"[{model_name}-{well_id}] Epoch {ep:02d}/{EPOCHS} | Train {tr_loss:.6f} | Val {va_loss:.6f}")
#
#         if va_loss < best_val:
#             best_val = va_loss
#             best_wts = model.state_dict()
#             patience = 0
#         else:
#             patience += 1
#             if patience >= EARLY_STOP_PATIENCE:
#                 print(f"[{model_name}-{well_id}] Early stopping at epoch {ep}")
#                 break
#
#     # 保存最佳模型
#     if best_wts is not None:
#         model.load_state_dict(best_wts)
#     torch.save(model.state_dict(), os.path.join(result_dir, "best_model.pth"))
#
#     # --- 测试集评估（MSE / MAPE / R2） ---
#     model.eval()
#     with torch.no_grad():
#         test_mse = evaluate_model(model, test_loader, criterion, DEVICE)
#         y_true_all, y_pred_all = [], []
#         for xb, yb in test_loader:
#             xb = xb.to(DEVICE).float()
#             yb = yb.to(DEVICE).float()
#             preds = model(xb)                    # [B, pred_len]
#             y_true_all.extend(yb.cpu().numpy().reshape(-1))
#             y_pred_all.extend(preds.cpu().numpy().reshape(-1))
#
#     test_mape = calculate_mape(y_true_all, y_pred_all)
#     test_r2   = calculate_r2(y_true_all, y_pred_all)
#
#     # 保存单井结果
#     single_res = {
#         "model": model_name,
#         "well_id": well_id,
#         "best_val_mse": float(best_val),
#         "test_mse": float(test_mse),
#         "test_mape(%)": float(test_mape),
#         "test_r2": float(test_r2),
#         "best_params": best_params
#     }
#     pd.DataFrame([single_res]).to_csv(os.path.join(result_dir, "summary.csv"), index=False, encoding="utf-8-sig")
#     print(f"[{model_name}-{well_id}] Test MSE={test_mse:.6f} | MAPE={test_mape:.2f}% | R2={test_r2:.4f}")
#
#     return single_res
#
# # ---------------- 主程序 ----------------
# if __name__ == "__main__":
#     ensure_data_ready()
#
#     # 读取井清单
#     wells_df = pd.read_csv(TOP_FEATURES_PATH, encoding="utf-8-sig")
#     well_ids = wells_df["well_id"].astype(str).unique().tolist()
#
#     all_results = []
#
#     for well in well_ids:
#         print(f"\n========== Processing {well} ==========\n")
#         for model_name, model_cls in MODEL_REGISTRY.items():
#             res = run_one_model_on_one_well(model_name, model_cls, well)
#             all_results.append(res)
#
#     # 汇总各模型-各井测试指标
#     df_all = pd.DataFrame(all_results)
#     df_all.to_csv(os.path.join(RESULTS_ROOT, "all_models_all_wells_summary.csv"),
#                   index=False, encoding="utf-8-sig")
#     print(f"\n已保存汇总表：{os.path.join(RESULTS_ROOT, 'all_models_all_wells_summary.csv')}")
# -*- coding: utf-8 -*-
"""
train_v2_baseline.py

一次性训练/测试 FusionModel 与三种 baseline（CNNBaseline, iTransformerBaseline, CNNiTransformerBaseline）
- 对每口井分别训练（使用相同超参或可调整）
- 保存每口井逐时间步预测结果 CSV： date, true_log, pred_log, true_exp, pred_exp
- 输出并汇总逐井指标（exp-space MSE, RMSE, MAE, MAPE, R2）
"""

import os
import math
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import create_dataloaders
from model import (
    FusionModel,
    CNNBaseline,
    iTransformerBaseline,
    CNNiTransformerBaseline,
    train_one_epoch,
    evaluate_model,
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------- CONFIG ----------------
PREPROCESSED_DIR = "preprocessed_data"                  # 每井 CSV 存放目录（每口井文件名应为 {well}_preprocessed.csv）
TOP_FEATURES_PATH = "top_features_per_well.csv"         # 用于读取 well_id 列
RESULTS_DIR = "results_baselines"                       # 保存结果的根目录（会为每个模型建子目录）
os.makedirs(RESULTS_DIR, exist_ok=True)

# training hyperparams (可按需修改)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN = 12
PRED_LEN = 3
BATCH_SIZE = 32
EPOCHS = 50
PATIENCE = 6
LEARNING_RATE = 1e-3

# model default hyperparams (可用 WOA/手动替换)
D_MODEL = 64
D_FF = 128
NUM_HEADS = 4
CNN_KERNEL = 3
DROPOUT = 0.1

# baseline/model list
MODEL_LIST = [
    ("Fusion", FusionModel),
    ("CNN", CNNBaseline),
    ("iTransformer", iTransformerBaseline),
    ("CNNiTransformer", CNNiTransformerBaseline),
]

# ---------------- helpers ----------------
def safe_read_wells(top_features_path):
    """从 top_features_per_well.csv 获取井列表（well_id 列）"""
    df = pd.read_csv(top_features_path, encoding='utf-8-sig')
    col = None
    for cand in ["well_id", "well", "井号"]:
        if cand in df.columns:
            col = cand
            break
    if col is None:
        raise ValueError(f"{top_features_path} 中未找到井号列 (期待 'well_id'/'well'/'井号')。")
    wells = df[col].astype(str).unique().tolist()
    return wells

def get_test_dates_for_well(per_well_csv, seq_len=SEQ_LEN, pred_len=PRED_LEN, train_ratio=0.7, val_ratio=0.15):
    """复现 create_dataloaders 的切分逻辑，返回测试窗口对应的日期（预测时刻，用窗口结束后的第一个预测步）"""
    df = pd.read_csv(per_well_csv, encoding='utf-8-sig')
    # find date column
    if 'date' not in df.columns:
        date_candidates = [c for c in df.columns if 'date' in c.lower() or '时间' in c or 'time' in c.lower()]
        if date_candidates:
            df = df.rename(columns={date_candidates[0]: 'date'})
        else:
            df['date'] = df.index.astype(str)
    df = df.sort_values('date').reset_index(drop=True)
    T = len(df)
    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))
    train_end = max(train_end, seq_len + 1)
    val_end = max(val_end, train_end + 1)
    if val_end >= T:
        val_end = min(T - pred_len, train_end + 1)
    window_start_max = T - seq_len - pred_len + 1
    train_idx_end = max(0, train_end - seq_len - pred_len + 1)
    val_idx_end = max(0, val_end - seq_len - pred_len + 1)
    test_window_starts = list(range(val_idx_end, window_start_max))
    test_dates = [str(df['date'].iloc[i + seq_len]) for i in test_window_starts]
    return test_dates

def safe_metrics(y_true, y_pred):
    """计算 exp-space 的常用回归指标（接受一维 numpy 数组）"""
    y_true = np.array(y_true).astype(float).flatten()
    y_pred = np.array(y_pred).astype(float).flatten()
    # 对长度进行保护
    if len(y_true) == 0:
        return dict(MSE=np.nan, RMSE=np.nan, MAE=np.nan, MAPE=np.nan, R2=np.nan)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    # MAPE: mask zeros
    mask = (y_true != 0)
    if mask.sum() == 0:
        mape = float('nan')
    else:
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float('nan')
    return dict(MSE=mse, RMSE=rmse, MAE=mae, MAPE=mape, R2=r2)

# ---------------- core: run one model on one well ----------------
def run_one_model_on_one_well(model_cls, model_name, well_id):
    """
    Train & test one model on one well.
    Returns:
      metrics (dict) in exp-space, and saves per-window predictions CSV to RESULTS_DIR/{model_name}/predictions_{well_id}.csv
    """
    print(f"\n========== {model_name} - {well_id} ==========")
    per_well_csv = os.path.join(PREPROCESSED_DIR, f"{well_id}_preprocessed.csv")
    if not os.path.exists(per_well_csv):
        raise FileNotFoundError(f"找不到单井预处理文件：{per_well_csv}")

    # create dataloaders (single-well file)
    train_loader, val_loader, test_loader, input_dim = create_dataloaders(
        preprocessed_path=per_well_csv,
        top_features_path=TOP_FEATURES_PATH,
        well_id=well_id,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN
    )

    # instantiate model (统一参数签名)
    model = model_cls(
        input_dim=input_dim,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_heads=NUM_HEADS,
        cnn_kernel=CNN_KERNEL,
        dropout=DROPOUT,
        pred_len=PRED_LEN
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # training with early stopping (using val_loss)
    best_val_loss = float('inf')
    best_path = os.path.join(RESULTS_DIR, f"{model_name}_{well_id}_best.pt")
    patience = 0
    for epoch in range(1, EPOCHS + 1):
        # 注意 train_one_epoch 的参数顺序： (model, dataloader, criterion, optimizer, device)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss = evaluate_model(model, val_loader, criterion, DEVICE)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping (patience={PATIENCE})")
                break

    # if no model saved (rare) -> save last state
    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)

    # load best and predict on test set
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE).float()
            preds = model(xb)             # shape (batch, pred_len)
            all_preds.append(preds.cpu().numpy())
            all_trues.append(yb.cpu().numpy())

    if len(all_preds) == 0:
        print(f"[WARN] {model_name} {well_id} 测试集为空，跳过输出。")
        return None

    all_preds = np.concatenate(all_preds, axis=0).reshape(-1, PRED_LEN)
    all_trues = np.concatenate(all_trues, axis=0).reshape(-1, PRED_LEN)

    # 取第一步预测进行对齐（与当前 pipeline 一致）
    pred_series = all_preds[:, 0].flatten()
    true_series = all_trues[:, 0].flatten()

    # 获取测试窗口对应的日期（与 create_dataloaders 的时间切分一致）
    test_dates = get_test_dates_for_well(per_well_csv, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    N_test_windows = len(test_dates)
    # 对齐长度（安全截断或填充）
    pred_series = pred_series[:N_test_windows]
    true_series = true_series[:N_test_windows]
    test_dates = test_dates[:N_test_windows]

    # 反对数变换（preprocess.py 使用 log1p -> inverse is expm1）
    pred_exp = np.expm1(pred_series)
    true_exp = np.expm1(true_series)

    # save per-window CSV
    out_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"predictions_{well_id}.csv")
    df_out = pd.DataFrame({
        "date": test_dates,
        "true_log": true_series,
        "pred_log": pred_series,
        "true_exp": true_exp,
        "pred_exp": pred_exp
    })
    df_out.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"[保存预测] {out_csv}")

    # compute metrics in exp-space
    metrics_exp = safe_metrics(true_exp, pred_exp)
    # also compute in log-space (for convenience/comparison)
    metrics_log = safe_metrics(true_series, pred_series)

    metrics = {
        "model": model_name,
        "well_id": well_id,
        # exp space
        "mse_exp": metrics_exp["MSE"],
        "rmse_exp": metrics_exp["RMSE"],
        "mae_exp": metrics_exp["MAE"],
        "mape_exp": metrics_exp["MAPE"],
        "r2_exp": metrics_exp["R2"],
        # log space
        "mse_log": metrics_log["MSE"],
        "rmse_log": metrics_log["RMSE"],
        "mae_log": metrics_log["MAE"],
        "mape_log": metrics_log["MAPE"],
        "r2_log": metrics_log["R2"],
        "n_test_windows": N_test_windows,
        "best_val_loss": best_val_loss
    }

    print(f"[{model_name} - {well_id}] MSE(exp)={metrics['mse_exp']:.6f}, MAPE(exp)={metrics['mape_exp']:.2f}%, R2(exp)={metrics['r2_exp']:.4f}")
    return metrics

# ---------------- main ----------------
def main():
    # wells list from top_features_per_well.csv
    wells = safe_read_wells(TOP_FEATURES_PATH)
    print("Found wells:", wells)

    all_results = []
    # for each model run across wells
    for model_name, model_class in MODEL_LIST:
        print("\n\n" + "="*20 + f" Running model: {model_name} " + "="*20)
        for well in wells:
            try:
                metrics = run_one_model_on_one_well(model_class, model_name, well)
                if metrics is not None:
                    all_results.append(metrics)
            except Exception as e:
                print(f"[ERROR] {model_name} on {well} failed: {e}")

    # save summary
    if len(all_results) > 0:
        df_res = pd.DataFrame(all_results)
        out_summary = os.path.join(RESULTS_DIR, "all_models_all_wells_summary.csv")
        df_res.to_csv(out_summary, index=False, encoding='utf-8-sig')
        print(f"\nSaved summary: {out_summary}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
