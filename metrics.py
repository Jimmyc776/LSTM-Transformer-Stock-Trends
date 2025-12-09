import numpy as np
import torch
import torch.nn.functional as F

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