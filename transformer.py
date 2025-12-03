import torch.nn as nn
import torch
from typing import Tuple, Optional
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000, dropout: float=0.1):
      super().__init__()
      self.dropout=nn.Dropout(dropout)
      pe=torch.zeros(max_len,d_model)
      position = torch.arange(0,max_len).unsqueeze(1).float()
      div_term = torch.exp(-math.log(10000.0)*torch.arange(0,d_model,2).float()/d_model)
      pe[:,0::2] = torch.sin(position*div_term)
      pe[:,1::2] = torch.cos(position*div_term)
      pe=pe.unsqueeze(0)
      self.register_buffer('pe',pe)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
      x=x+self.pe[:,: x.size(1)]
      return self.dropout(x)

    
class StockTransformer(nn.Module):
    def __init__(self, inp_dim: int, d_model: int=64, n_heads: int=4, n_layers: int=3,
                 dim_feedforward: int=128, dropout: float=0.1, output_dim: int=1, max_len: int=500):
        super().__init__()
        self.d_model=d_model
        self.input_proj = nn.Linear(inp_dim, d_model)
        self.pos_encoding=PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        encoder = nn.TransformerEncoderLayer(d_model=d_model,
                                             nhead=n_heads,
                                             dim_feedforward=dim_feedforward,
                                             dropout=dropout,
                                             batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, num_layers=n_layers)
        self.readout = nn.Linear(d_model, output_dim)

    def _generate_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
       mask=torch.triu(torch.ones(T, T, device=device), diagonal=1)
       mask=mask.masked_fill(mask==1, float('-inf')).masked_fill(mask==0,0.0)
       return mask
        

    def forward(self, x: torch.Tensor, return_attn: bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        '''
        x                the inputs. shape: (B x T x dim)

        
        Outputs:
        attn_output      shape: (B x T x dim)
        attn_alphas      If return_attn is False, return None. Otherwise return the attention weights
                         of each of each of the attention heads for each of the layers.
                         shape: (B x Num_layers x Num_heads x T x T)

        output, collected_attns = None, None
        '''
        device=x.device
        B,T,_=x.shape

        x=self.input_proj(x)*math.sqrt(self.d_model)
        x=self.pos_encoding(x)

        causal_mask=self._generate_causal_mask(T,device)
        encoded=self.encoder(x,mask=causal_mask)
        last_hidden = encoded[:,-1,:]
        y_pred=self.readout(last_hidden)

        if return_attn:
           return y_pred,encoded
        return y_pred,None
