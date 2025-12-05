import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Union, List, Optional

def plot_time_series(data_source: Union[str, torch.Tensor, np.ndarray],
                     tickers: Optional[List[str]]=None,
                     num_stocks: Optional[int]=None,
                     separate: bool=False, 
                     max_samples: int=10000,
                     title: str="Stock Time Series") -> None:
    """
    Plots multiple time series either together or separately

    Args:
        data_source: CSV path, torch.Tensor, or nd.array
        tickers: Specific stock tickers to plot (validated against data)
        num_stocks: Max stocks to plot if tickers=None (overrides data length)
        separate: Plot each stock separately
        max_samples: Limit total points plotted (for 20yr data)
        title: Plot title
    """
    plot_tickers = []

    if isinstance(data_source, str):
        # Load from CSV
        df = pd.read_csv(data_source, index_col=0, parse_dates=True)
        available_tickers = [col for col in df.columns if col != 'Date']

        # Validate tickers
        if tickers is not None:
            valid_tickers = [t for t in tickers if t in available_tickers]
            plot_tickers = valid_tickers[:num_stocks] if num_stocks else valid_tickers
        else:
            plot_tickers = available_tickers[:num_stocks] if num_stocks else available_tickers
        
        if not plot_tickers:
            raise ValueError("No valid tickers found in CSV to plot.")
        
        plot_df = df[plot_tickers].iloc[:max_samples]
    
    elif isinstance(data_source, (torch.Tensor, np.ndarray)):
        if isinstance(data_source, torch.Tensor):
            data = data_source.cpu().numpy()
        else:
            data = np.asarray(data_source)

        data = data.astype(np.float32)
        data = np.nan_to_num(data, nan=0.0)
        
        if data.ndim == 1:
            if tickers is None:
                plot_tickers = ["Time Series"]
            else:
                plot_tickers = [tickers[0]]
            plot_df = pd.DataFrame({plot_tickers[0]: data[:max_samples]})
        elif data.ndim == 2:
            num_available = data.shape[0]
            if tickers is None:
                plot_tickers = [f"Series {i+1}" for i in range(min(num_available, num_stocks or num_available))]
            else:
                plot_tickers = tickers[:num_stocks or len(tickers)]
            
            plot_df = pd.DataFrame(data[:len(plot_tickers), :max_samples].T, columns=plot_tickers)
    
    else:
        raise ValueError("data_source must be str (CSV), torch.Tensor, or np.ndarray.")

    # Plot
    if separate:
        fig, axes = plt.subplots(1, len(plot_tickers), figsize=(4*len(plot_tickers), 4))
        if len(plot_tickers) == 1: axes = [axes]
        for i, ticker in enumerate(plot_tickers):
            axes[i].plot(plot_df[ticker])
            axes[i].set_title(ticker)
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Value")
        plt.tight_layout()
    else:
        plt.figure(figsize=(15, 8))
        for ticker in plot_tickers:
            plt.plot(plot_df[ticker], label=ticker, alpha=0.8)
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.show()