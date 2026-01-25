import torch
import torch.nn as nn

class TRA(nn.Module):
    """Temporal Recurrent Attention (Basic version - returns x * Attention)"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at = self.att_fc(at).transpose(1,2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At

class DynamicTRA(nn.Module):
    """Temporal Recurrent Attention (Dynamic version - returns Attentions explicitly)"""
    def __init__(self, channels, kernels, kat_heads):
        super().__init__()
        self.kernels = kernels
        self.kat_heads = kat_heads
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc1 = nn.Linear(channels*2, channels)
        self.att_act1 = nn.Sigmoid()
        self.att_fc2 = nn.Linear(channels * 2, kat_heads * kernels)
        self.att_act2 = nn.Softmax(dim=2)

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1,2))[0]
        at1 = self.att_fc1(at).transpose(1,2)
        at1 = self.att_act1(at1)
        At = at1[..., None]  # (B,C,T,1)
        kat = self.att_fc2(at).transpose(1,2) # (B,kat_heads*kernels,T)
        kat = kat.reshape(x.shape[0], self.kat_heads, self.kernels, x.shape[2])  # (B,kat_heads,kernels,T)
        kat = self.att_act2(kat)
        kat = torch.unbind(kat, dim=1)  # kat_heads * (B,kernels,T)

        return At, kat
