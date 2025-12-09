import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred.float(), target.float()).item()

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred.float(), target.float()).item()

def direct_full_series(model, series: torch.Tensor, seq_len: int, device: torch.device) -> tuple[float, float]:
    series = series.squeeze()
    T = len(series)
    mses, maes = [], []

    with torch.no_grad():
        for i in range(seq_len, T):
            x = series[i-seq_len:i].unsqueeze(-1).unsqueeze(0).to(device)
            pred = model(x).squeeze().cpu()
            target = series[i]

            mses.append(mse(pred, target))
            maes.append(mae(pred, target))

    return np.mean(mses), np.mean(maes)

def direct_category_window(model, series: torch.Tensor, seq_len: int, device: torch.device) -> tuple[float, float]:
    series = series.squeeze()
    tail = series[-2*seq_len:]
    return direct_full_series(model, tail, seq_len, device)

def save_metrics_csv(results: dict, csv_path: str, metadata_csv: str | None = None) -> None:
    ticker_to_category = {}
    if metadata_csv is not None and os.path.exists(metadata_csv):
        meta = pd.read_csv(metadata_csv)
        if "ticker" in meta.columns and "category" in meta.columns:
            ticker_to_category = (
                meta[["ticker", "category"]]
                .drop_duplicates()
                .set_index("ticker")["category"]
                .to_dict()
            )

    # Build tall DataFrame first (one row per ticker)
    rows = []
    for ticker, m in results.items():
        rows.append({
            "ticker": ticker,
            "category": ticker_to_category.get(ticker, "unknown"),
            "lstm_full_mse":  m["lstm_full_mse"],
            "lstm_full_mae":  m["lstm_full_mae"],
            "lstm_cat_mse":   m["lstm_cat_mse"],
            "lstm_cat_mae":   m["lstm_cat_mae"],
            "trans_full_mse": m["trans_full_mse"],
            "trans_full_mae": m["trans_full_mae"],
            "trans_cat_mse":  m["trans_cat_mse"],
            "trans_cat_mae":  m["trans_cat_mae"],
        })

    df = pd.DataFrame(rows)
    df = df.set_index("ticker").T
    df.insert(0, "ticker", df.index)
    df = df.reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved wide metrics to {csv_path}")
