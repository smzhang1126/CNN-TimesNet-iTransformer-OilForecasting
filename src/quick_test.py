import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEED = 42
DATA_PATH = "data/example_data.csv"
RESULT_DIR = "results"
RESULT_PATH = os.path.join(RESULT_DIR, "quick_test_predictions.csv")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuickForecastModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        self.cnn = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.input_projection(x)

        cnn_input = x.transpose(1, 2)
        cnn_output = self.cnn(cnn_input).transpose(1, 2)

        transformer_output = self.transformer(cnn_output)

        last_hidden = transformer_output[:, -1, :]
        output = self.regressor(last_hidden)

        return output.squeeze(-1)


def create_sliding_windows(df, feature_cols, target_col, seq_len=6):
    x_list = []
    y_list = []
    date_list = []
    well_list = []

    for well_id, group in df.groupby("well_id"):
        group = group.sort_values("date").reset_index(drop=True)

        features = group[feature_cols].values
        target = group[target_col].values
        dates = group["date"].values

        for i in range(len(group) - seq_len):
            x_list.append(features[i:i + seq_len])
            y_list.append(target[i + seq_len])
            date_list.append(dates[i + seq_len])
            well_list.append(well_id)

    return (
        np.array(x_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
        np.array(date_list),
        np.array(well_list)
    )


def main():
    set_seed(SEED)

    print("Loading example data...")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. Please make sure data/example_data.csv exists."
        )

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["well_id", "date"]).reset_index(drop=True)

    target_col = "oil_production"

    feature_cols = [
        "liquid_production",
        "water_cut",
        "formation_pressure",
        "bottom_hole_pressure",
        "casing_pressure",
        "tubing_pressure",
        "back_pressure",
        "injection_rate"
    ]

    seq_len = 6

    df[target_col] = np.log1p(df[target_col])

    x, y, dates, wells = create_sliding_windows(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        seq_len=seq_len
    )

    if len(x) < 5:
        raise ValueError("Not enough samples for quick test. Please add more rows to example_data.csv.")

    split_idx = int(len(x) * 0.8)

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = dates[split_idx:]
    wells_test = wells[split_idx:]

    scaler = StandardScaler()

    n_train, t, f = x_train.shape
    n_test = x_test.shape[0]

    x_train_2d = x_train.reshape(-1, f)
    x_test_2d = x_test.reshape(-1, f)

    x_train_scaled = scaler.fit_transform(x_train_2d).reshape(n_train, t, f)
    x_test_scaled = scaler.transform(x_test_2d).reshape(n_test, t, f)

    x_train_tensor = torch.tensor(x_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    print("Building CNN-TimesNet-iTransformer quick-test model...")

    model = QuickForecastModel(
        input_dim=len(feature_cols),
        hidden_dim=32,
        num_heads=4,
        num_layers=1,
        dropout=0.1
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Running quick test...")

    model.train()
    epochs = 10

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(x_train_tensor)
        loss = criterion(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        pred_test_log = model(x_test_tensor).cpu().numpy()

    y_test_real = np.expm1(y_test)
    pred_test_real = np.expm1(pred_test_log)

    mae = mean_absolute_error(y_test_real, pred_test_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, pred_test_real))
    r2 = r2_score(y_test_real, pred_test_real)

    print("Quick test completed successfully.")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2:   {r2:.4f}")

    os.makedirs(RESULT_DIR, exist_ok=True)

    result_df = pd.DataFrame({
        "date": dates_test,
        "well_id": wells_test,
        "true_oil_production": y_test_real,
        "predicted_oil_production": pred_test_real
    })

    result_df.to_csv(RESULT_PATH, index=False)

    print(f"Prediction results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()