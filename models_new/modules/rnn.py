import torch
import torch.nn as nn

class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h== None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2, x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers, x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h
    
    
class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size, hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size, hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)
    
    def forward(self, x):
        """x: (B, C, T, F)"""
        ## Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size) # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        ## Inter RNN
        x = intra_out.permute(0,2,1,3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3]) 
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size) # (B,F,T,C)
        inter_x = inter_x.permute(0,2,1,3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_out = torch.add(intra_out, inter_x)
        
        dual_out = inter_out.permute(0,3,1,2)  # (B,C,T,F)
        
        return dual_out
