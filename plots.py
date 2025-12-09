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