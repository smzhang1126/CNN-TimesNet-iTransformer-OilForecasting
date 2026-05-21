# -*- codeing = utf-8 -*-
# @time ： 2025/8/11 23:15
# @author : likun
# @file : train_v2.py
# @software : PyCharm
# -*- coding: utf-8 -*-
"""
train_v2.py

增强版训练脚本：
1. 数据预处理
2. Top-N 特征选择
3. WOA 搜索超参数
4. 逐井训练 CNN-TimesNet-iTransformer
5. 测试阶段保存：
   - 预测 CSV
   - 测试集 log 空间指标
   - 测试集 exp 空间指标
   - final_results_v2.csv

注意：
- preprocess.py 中使用 np.log1p(月产油量)
- 因此反变换必须使用 np.expm1()
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

import preprocess
import top_features_selector
import data_loader
import model


# =========================
# 全局配置
# =========================
RAW_DATA_PATH = "init_data.xlsx"

PREPROCESSED_DIR = "preprocessed_data"
PREPROCESSED_PATH = "preprocessed_data.csv"

TOP_FEATURES_PATH = "top_features_per_well.csv"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

SEQ_LEN = 12
PRED_LEN = 3
EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 工具函数
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_d_model_for_heads(d_model, num_heads):
    d_model = int(d_model)
    num_heads = int(num_heads)

    if num_heads <= 0:
        return d_model

    if d_model % num_heads == 0:
        return d_model

    return ((d_model + num_heads - 1) // num_heads) * num_heads


def read_wells_from_top_features(top_features_path):
    df = pd.read_csv(top_features_path, encoding="utf-8-sig")
    if "well_id" not in df.columns:
        if "well" in df.columns:
            df = df.rename(columns={"well": "well_id"})
        elif "井号" in df.columns:
            df = df.rename(columns={"井号": "well_id"})
        else:
            raise ValueError("top_features_per_well.csv 中缺少 well_id 列")
    return df["well_id"].astype(str).unique().tolist()


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


def get_test_dates(per_well_path, seq_len=12, pred_len=3, train_ratio=0.7, val_ratio=0.15):
    """
    复现 data_loader 的切分逻辑，获取测试窗口对应日期。
    """
    df = pd.read_csv(per_well_path, encoding="utf-8-sig")

    if "date" not in df.columns:
        date_candidates = [
            c for c in df.columns
            if "date" in str(c).lower()
            or "time" in str(c).lower()
            or "时间" in str(c)
            or "年" in str(c)
        ]
        if len(date_candidates) > 0:
            df = df.rename(columns={date_candidates[0]: "date"})
        else:
            df["date"] = np.arange(len(df)).astype(str)

    df = df.sort_values("date").reset_index(drop=True)

    T = len(df)

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_end = max(train_end, seq_len + 1)
    val_end = max(val_end, train_end + 1)

    if val_end >= T:
        val_end = min(T - pred_len, train_end + 1)

    window_start_max = T - seq_len - pred_len + 1
    val_idx_end = max(0, val_end - seq_len - pred_len + 1)

    test_window_starts = list(range(val_idx_end, window_start_max))

    dates = []
    for i in test_window_starts:
        idx = i + seq_len
        if idx < len(df):
            dates.append(str(df["date"].iloc[idx]))

    return dates


def predict_on_loader(net, loader, device):
    """
    返回第一步预测和真实值。
    """
    net.eval()

    all_preds = []
    all_trues = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device).float()
            preds = net(xb)

            all_preds.append(preds.cpu().numpy())
            all_trues.append(yb.cpu().numpy())

    if len(all_preds) == 0:
        return np.array([]), np.array([])

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    pred_series = all_preds[:, 0].flatten()
    true_series = all_trues[:, 0].flatten()

    return true_series, pred_series


# =========================
# WOA 适应度函数
# =========================
def train_func(
    params,
    preprocessed_path,
    top_features_path,
    device,
    seq_len=12,
    pred_len=3,
    sample_wells=5,
    min_windows_total=80,
    max_wells=8
):
    import torch.nn as nn
    import torch.optim as optim

    try:
        lr = float(params.get("learning_rate", 1e-3))
        d_model = int(params.get("d_model", 64))
        d_ff = int(params.get("d_ff", 128))
        dropout = float(params.get("dropout", 0.1))
        num_heads = int(params.get("num_heads", 1))
        cnn_kernel = int(params.get("cnn_kernel", 3))
        batch_size = int(params.get("batch_size", 32))
    except Exception as e:
        print(f"[WOA] 参数解析失败: {e}")
        return 1e6

    d_model = adjust_d_model_for_heads(d_model, num_heads)

    try:
        wells = read_wells_from_top_features(top_features_path)
    except Exception as e:
        print(f"[WOA] 读取井列表失败: {e}")
        return 1e6

    if len(wells) == 0:
        return 1e6

    if sample_wells is not None and sample_wells < len(wells):
        selected_wells = random.sample(wells, sample_wells)
    else:
        selected_wells = wells

    try:
        train_loader, val_loader, _, input_dim = data_loader.create_combined_dataloaders(
            preprocessed_path=preprocessed_path,
            top_features_path=top_features_path,
            well_ids=selected_wells,
            batch_size=batch_size,
            seq_len=seq_len,
            pred_len=pred_len,
            min_windows_total=min_windows_total,
            max_wells=max_wells
        )
    except Exception as e:
        print(f"[WOA] 构造合并 DataLoader 失败: {e}")
        return 1e6

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        return 1e6

    try:
        net = model.FusionModel(
            input_dim=input_dim,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            cnn_kernel=cnn_kernel,
            dropout=dropout,
            pred_len=pred_len
        ).to(device)
    except Exception as e:
        print(f"[WOA] 模型构建失败: {e}")
        return 1e6

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    val_loss = 1e6

    try:
        for _ in range(2):
            model.train_one_epoch(net, train_loader, criterion, optimizer, device)
            val_loss = model.evaluate_model(net, val_loader, criterion, device)
    except Exception as e:
        print(f"[WOA] 训练失败: {e}")
        return 1e6

    if not np.isfinite(val_loss):
        return 1e6

    return float(val_loss)


# =========================
# 主流程
# =========================
if __name__ == "__main__":
    set_seed(42)

    print(f"使用设备: {DEVICE}")

    # 1. 预处理
    print("\n====== 步骤 1：数据预处理 ======")
    preprocess.preprocess_data(
        input_excel_path=RAW_DATA_PATH,
        combined_csv_path=PREPROCESSED_PATH,
        output_dir=PREPROCESSED_DIR
    )

    # 2. 特征选择
    print("\n====== 步骤 2：Top-N 特征选择 ======")
    top_features_selector.select_top_features_per_well(
        data_dir=PREPROCESSED_DIR,
        top_n=5,
        output_path=TOP_FEATURES_PATH,
        detailed_output="top_features_detailed.csv"
    )

    # 3. WOA 搜索
    print("\n====== 步骤 3：WOA 超参数优化 ======")

    search_space = {
        "learning_rate": (1e-5, 1e-2),
        "d_model": (16, 128),
        "d_ff": (64, 512),
        "dropout": (0.05, 0.5),
        "num_heads": (1, 8),
        "cnn_kernel": (2, 6),
        "batch_size": (8, 64)
    }

    best_params, best_fitness = model.woa_optimize(
        train_func,
        search_space=search_space,
        pop_size=12,
        max_iter=20,
        preprocessed_path=PREPROCESSED_PATH,
        top_features_path=TOP_FEATURES_PATH,
        device=DEVICE,
        seq_len=SEQ_LEN,
        pred_len=PRED_LEN,
        sample_wells=5,
        min_windows_total=80,
        max_wells=8
    )

    print("\n====== WOA 优化完成 ======")
    print(f"最优验证 Loss: {best_fitness:.6f}")
    print("最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 4. 逐井训练和评估
    print("\n====== 步骤 4：逐井训练和测试 ======")

    wells = read_wells_from_top_features(TOP_FEATURES_PATH)
    final_results = []

    for well_id in wells:
        print(f"\n---------- 训练井 {well_id} ----------")

        per_well_path = os.path.join(PREPROCESSED_DIR, f"{well_id}_preprocessed.csv")
        if not os.path.exists(per_well_path):
            print(f"[跳过] 找不到单井文件: {per_well_path}")
            continue

        try:
            train_loader, val_loader, test_loader, input_dim = data_loader.create_dataloaders(
                preprocessed_path=per_well_path,
                top_features_path=TOP_FEATURES_PATH,
                well_id=well_id,
                batch_size=int(best_params.get("batch_size", 32)),
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN
            )
        except Exception as e:
            print(f"[跳过] 数据加载失败: {e}")
            continue

        d_model = adjust_d_model_for_heads(
            int(best_params.get("d_model", 64)),
            int(best_params.get("num_heads", 1))
        )

        net = model.FusionModel(
            input_dim=input_dim,
            d_model=d_model,
            d_ff=int(best_params.get("d_ff", 128)),
            num_heads=int(best_params.get("num_heads", 1)),
            cnn_kernel=int(best_params.get("cnn_kernel", 3)),
            dropout=float(best_params.get("dropout", 0.1)),
            pred_len=PRED_LEN
        ).to(DEVICE)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=float(best_params.get("learning_rate", 1e-3))
        )

        best_val_loss = float("inf")
        save_path = os.path.join(RESULTS_DIR, f"best_model_well_{well_id}.pt")

        for epoch in range(1, EPOCHS + 1):
            train_loss = model.train_one_epoch(net, train_loader, criterion, optimizer, DEVICE)

            if len(val_loader.dataset) > 0:
                val_loss = model.evaluate_model(net, val_loader, criterion, DEVICE)
            else:
                val_loss = train_loss

            print(
                f"井 {well_id} | Epoch {epoch:03d} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(net.state_dict(), save_path)

        # 加载最佳模型
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path, map_location=DEVICE))

        # 测试集预测
        if len(test_loader.dataset) == 0:
            print(f"井 {well_id} 测试集为空")
            continue

        true_log, pred_log = predict_on_loader(net, test_loader, DEVICE)

        # 日期对齐
        test_dates = get_test_dates(
            per_well_path,
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

        print(f"[{well_id}] Log 空间指标:")
        print(f"  MSE:  {mse_log:.6f}")
        print(f"  RMSE: {rmse_log:.6f}")
        print(f"  MAE:  {mae_log:.6f}")
        print(f"  MAPE: {mape_log:.2f}%")
        print(f"  R²:   {r2_log:.6f}")

        print(f"[{well_id}] Exp 空间指标:")
        print(f"  MSE:  {mse_exp:.6f}")
        print(f"  RMSE: {rmse_exp:.6f}")
        print(f"  MAE:  {mae_exp:.6f}")
        print(f"  MAPE: {mape_exp:.2f}%")
        print(f"  R²:   {r2_exp:.6f}")

        # 保存预测 CSV
        pred_df = pd.DataFrame({
            "date": test_dates,
            "true_log": true_log,
            "pred_log": pred_log,
            "true_exp": true_exp,
            "pred_exp": pred_exp
        })

        pred_csv = os.path.join(RESULTS_DIR, f"predictions_well_{well_id}.csv")
        pred_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
        print(f"[保存预测 CSV] {pred_csv}")

        # 保存图
        plt.figure(figsize=(10, 4))
        plt.plot(test_dates, true_exp, label="True")
        plt.plot(test_dates, pred_exp, label="Pred")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(RESULTS_DIR, f"plot_well_{well_id}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"[保存预测图] {plot_path}")

        # 汇总结果
        final_results.append({
            "well_id": well_id,
            "best_val_mse": best_val_loss,

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

    # 5. 保存总结果
    result_df = pd.DataFrame(final_results)
    out_path = os.path.join(RESULTS_DIR, "final_results_v2.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n全部完成，结果保存至: {out_path}")
