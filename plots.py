import torch
import numpy as np
import matplotlib.pyplot as plt
from metrics import mse, mae

def plot_autoregressive_chain(lstm_model, transformer_model, ticker: str, series: torch.Tensor, seq_len: int, device: torch.device):
    series = series.squeeze()
    T = len(series)

    steps = T - seq_len
    lstm_context = series[:seq_len].clone()
    transformer_context = series[:seq_len].clone()
    lstm_preds = []
    transformer_preds = []

    with torch.no_grad():
        lstm_cur = lstm_context.clone()
        transformer_cur = transformer_context.clone()
        for _ in range(steps):
            lstm_x = lstm_cur[-seq_len:].unsqueeze(-1).unsqueeze(0).to(device)
            transformer_x = transformer_cur[-seq_len:].unsqueeze(-1).unsqueeze(0).to(device)
            lstm_pred = lstm_model(lstm_x).squeeze().cpu()
            transformer_pred = transformer_model(transformer_x).squeeze().cpu()
            lstm_preds.append(lstm_pred)
            transformer_preds.append(transformer_pred)
            lstm_cur = torch.cat([lstm_cur, lstm_pred.view(1)], dim=0)
            transformer_cur = torch.cat([transformer_cur, transformer_pred.view(1)], dim=0)

    lstm_preds = torch.stack(lstm_preds)
    transformer_preds = torch.stack(transformer_preds)

    true_tail = series[seq_len:seq_len+steps]
    lstm_mse = mse(lstm_preds, true_tail)
    lstm_mae = mae(lstm_preds, true_tail)
    transformer_mse = mse(transformer_preds, true_tail)
    transformer_mae = mae(transformer_preds, true_tail)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(series.numpy(), color='gray', alpha=0.4, label='True (full)')
    t_axis=range(seq_len, seq_len+steps)
    ax.plot(t_axis, true_tail.numpy(), color='black', linestyle='--', linewidth=2.0, label='True (evaluated)')


    ax.plot(t_axis, lstm_preds.numpy(), color='C1', linestyle='--', linewidth=2.5, label='LSTM prediction')
    ax.plot(t_axis, transformer_preds.numpy(), color='C2', linestyle='--', linewidth=2.5, label='Transformer prediction')

    ax.set_title(f"{ticker} – LSTM vs Transformer Auto-regressive Roll-out\n"
                f"LSTM: MSE={lstm_mse:.4f}, MAE={lstm_mae:.4f} | "
                f"Transformer: MSE={transformer_mse:.4f}, MAE={transformer_mae:.4f}", fontsize=12)
    ax.set_xlabel("Time index")
    ax.set_ylabel("Scaled price")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig

def plot_direct_full_series(lstm_model, transformer_model, ticker: str, series: torch.Tensor, seq_len: int, device: torch.device,) -> plt.Figure:
    series = series.squeeze()
    T = len(series)

    ts = np.arange(seq_len, T)
    true_vals = series[seq_len:T]

    lstm_preds = []
    transformer_preds = []

    with torch.no_grad():
        for t in range(seq_len, T):
            x = series[t-seq_len:t].unsqueeze(-1).unsqueeze(0).to(device)
            lstm_preds.append(lstm_model(x).squeeze().cpu())
            transformer_preds.append(transformer_model(x).squeeze().cpu())

    lstm_preds = torch.stack(lstm_preds)
    transformer_preds = torch.stack(transformer_preds)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, true_vals.numpy(), label="True", color="black")
    ax.plot(ts, lstm_preds.numpy(), label="LSTM", color="C1")
    ax.plot(ts, transformer_preds.numpy(), label="Transformer", color="C2")

    ax.set_title(f"{ticker} -- Direct one-step predictions")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Scaled price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_direct_category_window(lstm_model, transformer_model, ticker: str, series: torch.Tensor, seq_len: int, device: torch.device) -> plt.Figure:
    # Restrict to last 2*seq_len (context + category window)
    tail = series.squeeze()[-2*seq_len:]
    tail_T = len(tail)
    
    # Predictions only for the category window (last seq_len of tail)
    true_vals = tail[seq_len:tail_T]
    ts = np.arange(1, len(true_vals) + 1)

    lstm_preds = []
    transformer_preds = []

    with torch.no_grad():
        for t in range(seq_len, tail_T):
            x = tail[t-seq_len:t].unsqueeze(-1).unsqueeze(0).to(device)
            lstm_preds.append(lstm_model(x).squeeze().cpu())
            transformer_preds.append(transformer_model(x).squeeze().cpu())

    lstm_preds = torch.stack(lstm_preds)
    transformer_preds = torch.stack(transformer_preds)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(ts, true_vals.numpy(), label="True (category window)", color="black")
    ax.plot(ts, lstm_preds.numpy(), label="LSTM", color="C1")
    ax.plot(ts, transformer_preds.numpy(), label="Transformer", color="C2")

    ax.set_title(f"{ticker} -- Direct one-step predictions (classification window)")
    ax.set_xlabel("Time index within category window")
    ax.set_ylabel("Scaled price")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_category_window_errors(lstm_model, transformer_model, ticker: str, series: torch.Tensor, seq_len: int, device: torch.device) -> plt.Figure:
    series = series.squeeze()
    T = len(series)

    # Tail: 1250 context + 1250 category window
    tail = series[-2*seq_len:]   # [2*seq_len]
    tail_T = len(tail)

    true_vals = tail[seq_len:tail_T]  # category window targets

    lstm_preds = []
    trans_preds = []

    with torch.no_grad():
        for t in range(seq_len, tail_T):
            x = tail[t-seq_len:t].unsqueeze(-1).unsqueeze(0).to(device)
            lstm_preds.append(lstm_model(x).squeeze().cpu())
            trans_preds.append(transformer_model(x).squeeze().cpu())

    lstm_preds = torch.stack(lstm_preds)
    trans_preds = torch.stack(trans_preds)

    # Absolute error over time
    lstm_abs_err = (lstm_preds - true_vals).abs()
    trans_abs_err = (trans_preds - true_vals).abs()

    fig, ax = plt.subplots(figsize=(12, 5))
    local_ts = np.arange(len(true_vals))

    ax.plot(local_ts, lstm_abs_err.numpy(), label="LSTM |error|", color="C1")
    ax.plot(local_ts, trans_abs_err.numpy(), label="Transformer |error|", color="C2")

    ax.set_title(f"{ticker} -- Absolute error over category window (last 5y)")
    ax.set_xlabel("Time index within category window")
    ax.set_ylabel("Absolute error (scaled)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig
