import numpy as np
import torch
import torch.nn.functional as F

def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.mse_loss(pred.float(), target.float()).item()

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return F.l1_loss(pred.float(), target.float()).item()

def direct_full_series(model, series: torch.Tensor, seq_len: int, stride: int, device: torch.device) -> tuple[float, float]:
    series = series.squeeze()
    T = len(series)
    mses, maes = [], []

    for i in range(0, T-seq_len, stride):
        x = series[i:i+seq_len].unsqueeze(-1).unsqueeze(0).to(device)
        pred = model(x).squeeze().cpu()[:seq_len]
        target = series[i+seq_len:i+2*seq_len]

        mses.append(mse(pred, target))
        maes.append(mae(pred, target))
    
    return np.mean(mses), np.mean(maes)

def direct_category_window(model, series: torch.Tensor, seq_len: int, device: torch.device) -> tuple[float, float]:
    series = series.squeeze()
    x = series[-2*seq_len:-seq_len].unsqueeze(0).unsqueeze(-1).to(device)
    pred = model(x).squeeze().cpu()[:seq_len]
    target = series[-seq_len:].unsqueeze(0)

    return mse(pred, target), mae(pred, target)