# # # # -*- codeing = utf-8 -*-
# # # # @time ： 2025/8/10
# # # # @author : likun
# # # # @file : train.py
# # # # @software : PyCharm
# # # # train.py
# -*- coding: utf-8 -*-
"""
train.py

主训练脚本：
1. 数据预处理
2. 单井 Top-N 特征选择
3. 使用 WOA 在多井合并数据上搜索超参数
4. 使用最优超参数逐井训练 CNN-TimesNet-iTransformer 模型
5. 保存每口井最优模型与测试集 MSE

说明：
- 模型定义在 model.py 中
- 数据加载在 data_loader.py 中
- 特征选择在 top_features_selector.py 中
"""

import os
import random
import numpy as np
import pandas as pd
import torch

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
    """
    调整 d_model，使其可以被 num_heads 整除。
    """
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
    """
    WOA 调用的适应度函数。
    使用多井合并数据快速训练 2 个 epoch，返回验证集 loss。
    """
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
    if hasattr(preprocess, "preprocess_data"):
        preprocess.preprocess_data(
            input_excel_path=RAW_DATA_PATH,
            combined_csv_path=PREPROCESSED_PATH,
            output_dir=PREPROCESSED_DIR
        )
    else:
        raise AttributeError("preprocess.py 中未找到 preprocess_data")

    print(f"合并预处理文件: {PREPROCESSED_PATH}")
    print(f"单井预处理目录: {PREPROCESSED_DIR}")

    # 2. 特征选择
    print("\n====== 步骤 2：Top-N 特征选择 ======")
    if hasattr(top_features_selector, "select_top_features_per_well"):
        top_features_selector.select_top_features_per_well(
            data_dir=PREPROCESSED_DIR,
            top_n=5,
            output_path=TOP_FEATURES_PATH,
            detailed_output="top_features_detailed.csv"
        )
    else:
        raise AttributeError("top_features_selector.py 中未找到 select_top_features_per_well")

    print(f"Top 特征文件: {TOP_FEATURES_PATH}")

    # 3. WOA 超参数优化
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
    print(f"最优验证集 Loss: {best_fitness:.6f}")
    print("最优参数:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 4. 逐井训练
    print("\n====== 步骤 4：逐井训练并保存模型 ======")

    wells = read_wells_from_top_features(TOP_FEATURES_PATH)
    final_results = []

    for well_id in wells:
        print(f"\n---------- 训练井 {well_id} ----------")

        per_well_path = os.path.join(PREPROCESSED_DIR, f"{well_id}_preprocessed.csv")
        if not os.path.exists(per_well_path):
            print(f"[跳过] 找不到单井文件: {per_well_path}")
            continue

        batch_size = int(best_params.get("batch_size", 32))

        try:
            train_loader, val_loader, test_loader, input_dim = data_loader.create_dataloaders(
                preprocessed_path=per_well_path,
                top_features_path=TOP_FEATURES_PATH,
                well_id=well_id,
                batch_size=batch_size,
                seq_len=SEQ_LEN,
                pred_len=PRED_LEN
            )
        except Exception as e:
            print(f"[跳过] 井 {well_id} 数据加载失败: {e}")
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

        # 测试
        if os.path.exists(save_path):
            net.load_state_dict(torch.load(save_path, map_location=DEVICE))

        if len(test_loader.dataset) > 0:
            test_mse = model.evaluate_model(net, test_loader, criterion, DEVICE)
            print(f"井 {well_id} 测试集 MSE: {test_mse:.6f}")
        else:
            test_mse = np.nan
            print(f"井 {well_id} 测试集为空")

        final_results.append({
            "well_id": well_id,
            "best_val_mse": best_val_loss,
            "test_mse": test_mse
        })

    # 5. 保存结果
    result_df = pd.DataFrame(final_results)
    out_path = os.path.join(RESULTS_DIR, "final_results.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n训练完成，结果已保存: {out_path}")