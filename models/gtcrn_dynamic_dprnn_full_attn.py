"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params -> 93.28 MMac 89.429 K params ?
"""
import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.profiler import record_function

calculate_macs_mode = False


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
                                                / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                                                    / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2])  + 1e-12) \
                                                    / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1- erb_filters[-2, bins[-2]:bins[-1]+1]
        
        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))
    
    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1,kernel_size), stride=(1, stride), padding=(0, (kernel_size-1)//2))
        
    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class Conv2dAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), kernel_choices=1, use_deconv=False, groups=1, pad_left=0):
        super().__init__()
        self.pad_left = pad_left
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert out_channels % groups == 0 and in_channels % groups == 0
        self.use_deconv = use_deconv
        # candidates: (K, C_in, C_out, K_T, K_F)
        # candidates_bias: (K, C_out)

        self.candidates_bias = nn.Parameter(torch.empty((kernel_choices, out_channels), requires_grad=True))
        if use_deconv:
            in_channels, out_channels = out_channels, in_channels
        self.candidates = nn.Parameter(torch.empty((kernel_choices, in_channels//groups, out_channels, *kernel_size)), requires_grad=True)

        nn.init.kaiming_normal_(self.candidates)
        nn.init.zeros_(self.candidates_bias)

    def conv(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        unfolded = F.unfold(x_pad, (self.kernel_size[0], 1), dilation=(self.dilation[0], 1), padding=(self.padding[0], 0), stride=(self.stride[0], 1))
        unfolded2 = rearrange(unfolded, "b (c p) (t f) -> 1 (b t c) p f", c=x.shape[1], p=self.kernel_size[0], t=x.shape[2], f=x.shape[3])
        # x: (B, C, T, F)
        # unfolded: (B, C * K_T, T * F) -> (1, B*T*C, K_T, F)
        # attention: (B, K, T)
        grouped_kernels = torch.einsum("kiopq, bkt -> btiopq", self.candidates, attention)
        grouped_kernels = rearrange(grouped_kernels, "b t i o p q -> (b t o) i p q")
        grouped_bias = torch.einsum("ko, bkt -> bto", self.candidates_bias, attention)
        grouped_bias = rearrange(grouped_bias, "b t o -> (b t o)")
        out = F.conv2d(unfolded2, grouped_kernels, bias=grouped_bias, stride=self.stride, padding=(0, self.padding[1]), groups=x.shape[0]*x.shape[2]*self.groups)
        out = rearrange(out, "1 (b t o) 1 f -> b o t f", b=x.shape[0], t=x.shape[2], o=self.out_channels)
        return out
    
    def conv_and_sum(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        candidates = rearrange(self.candidates, "k i o p q -> (o k) i p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out = F.conv2d(x_pad, candidates, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        out = rearrange(out, "b (o k) t f -> b o k t f", o=self.out_channels, k=self.candidates_bias.shape[0])
        out = torch.einsum("b o k t f, b k t -> b o t f", out, attention)
        return out

    def deconv(self, out, attention):
        out = F.pad(out, [0, 0, self.pad_left, 0])  # pad left for causality
        attention = F.pad(attention, [self.pad_left, 0], "replicate")  # pad left for causality
        grouped_kernels = torch.einsum("kiopq, bkt -> btiopq", self.candidates, attention)
        grouped_kernels = rearrange(grouped_kernels, "b t i o p q -> (b t o) i p q")
        grouped_bias = torch.einsum("ko, bkt -> bto", self.candidates_bias, attention)
        grouped_bias = rearrange(grouped_bias, "b t o -> (b t o)")
        out1 = rearrange(out, "b o t f -> 1 (b t o) 1 f")
        unfolded2 = F.conv_transpose2d(out1, grouped_kernels, bias=grouped_bias, stride=self.stride, padding=(0, self.padding[1]), groups=out.shape[0]*out.shape[2]*self.groups)
        unfolded = rearrange(unfolded2, "1 (b t c) p f -> b (c p) (t f)", b=out.shape[0], t=out.shape[2], c=self.out_channels)
        x = F.fold(unfolded, (out.shape[2] - self.pad_left, out.shape[3]), (self.kernel_size[0], 1), dilation=(self.dilation[0], 1), padding=(self.padding[0], 0), stride=(self.stride[0], 1))
        return x
    
    def deconv_and_sum(self, out, attention):
        out = F.pad(out, [0, 0, self.pad_left, 0])  # pad left for causality
        # attention = F.pad(attention, [self.pad_left, 0], "replicate")  # pad left for causality
        candidates = rearrange(self.candidates, "k o i p q -> i (o k) p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out = F.conv_transpose2d(out, candidates, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        out = rearrange(out, "b (o k) t f -> b o k t f", o=self.out_channels, k=self.candidates_bias.shape[0])
        out = torch.einsum("b o k t f, b k t -> b o t f", out, attention)
        return out
    
    def forward(self, x, attention):
        if self.use_deconv:
            if not calculate_macs_mode:
                return self.deconv_and_sum(x, attention)
            return self.deconv(x, attention)
        if not calculate_macs_mode:
            return self.conv_and_sum(x, attention)
        return self.conv(x, attention)


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, kernel_choices=8, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        # conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)

        self.ln = nn.LayerNorm((in_channels//2*3, 33))
        
        self.point_conv1 = Conv2dAttention(in_channels//2*3, hidden_channels, (1, 1), use_deconv=use_deconv, kernel_choices=kernel_choices)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = Conv2dAttention(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels, use_deconv=use_deconv, pad_left=self.pad_size, 
                                            kernel_choices=kernel_choices)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = Conv2dAttention(hidden_channels, in_channels//2, (1, 1), use_deconv=use_deconv, kernel_choices=kernel_choices)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2, kernel_choices, 3)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        mask, (kat1, kat2, kat3) = self.tra(x1)
        x1 = self.sfe(x1)
        x1 = x1.permute(0,2,1,3)  # (B,T,C,F)
        x1 = self.ln(x1)
        x1 = x1.permute(0,2,1,3)  # (B,C,T,F)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1, kat1)))
        # h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1, kat2)))
        h1 = self.point_bn2(self.point_conv2(h1, kat3))

        h1 = h1 * mask

        x =  self.shuffle(h1, x2)
        
        return x


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


class TFMultiheadAttention(nn.Module):
    """
    Time-Frequency joint multi-head attention with 2D positional encoding.
    - Temporal: relative rotary position encoding (RoPE) applied to Q/K.
    - Frequency: GRU additive encoding added to token embeddings.
    - Causality: fixed-length causal window over time (attend to last W_t frames across all F freqs).
    - Linformer-style low-rank: project K/V along windowed sequence length to rank r via learnable E/F.

    Inputs: x in shape (B, C, T, F)
    Outputs: same shape
    """
    def __init__(self, embed_dim: int, num_heads: int, width: int, window_t: int = 4, linformer_rank: int = 16,
                 dropout: float = 0.0, rope_theta: float = 10000.0,
                 separable_linformer: bool = False, rank_t=None, rank_f=None):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.width = width  # frequency bins in latent (F)
        self.window_t = window_t  # temporal window in frames
        self.window_tokens = window_t * width
        self.linformer_rank = linformer_rank
        self.rope_theta = rope_theta

        # Frequency additive encoding (RNN across frequency for each time step)
        self.freq_rnn = nn.GRU(embed_dim, embed_dim, num_layers=1, batch_first=True, bidirectional=False)

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(dropout)

        self.separable_linformer = separable_linformer
        self.linformer_rank = linformer_rank

        if not self.separable_linformer:
            # Standard Linformer over window tokens W*F
            self.E = nn.Parameter(torch.empty(self.window_tokens, linformer_rank))  # for K
            self.Fp = nn.Parameter(torch.empty(self.window_tokens, linformer_rank)) # for V
            nn.init.xavier_uniform_(self.E)
            nn.init.xavier_uniform_(self.Fp)
        else:
            # Separable Linformer: (W x F) ≈ (W -> r_t) ⊗ (F -> r_f)
            if rank_t is None and rank_f is None:
                rt = max(1, min(window_t, int(round(math.sqrt(linformer_rank)))))
                rf = max(1, min(width, linformer_rank // rt))
                if rt * rf == 0:
                    rt, rf = 1, max(1, linformer_rank)
            else:
                rt = rank_t if rank_t is not None else max(1, min(window_t, linformer_rank))
                rf = rank_f if rank_f is not None else max(1, min(width, linformer_rank // rt))
            self.rank_t = rt
            self.rank_f = rf
            # Keys projections
            self.Et_k = nn.Parameter(torch.empty(window_t, rt))  # time proj W -> r_t
            self.Ef_k = nn.Parameter(torch.empty(width, rf))     # freq proj F -> r_f
            # Values projections
            self.Et_v = nn.Parameter(torch.empty(window_t, rt))
            self.Ef_v = nn.Parameter(torch.empty(width, rf))
            for p in [self.Et_k, self.Ef_k, self.Et_v, self.Ef_v]:
                nn.init.xavier_uniform_(p)

        # Precompute rotary inverse frequencies for temporal RoPE
        inv_idx = torch.arange(0, self.head_dim, 2).float()
        self.register_buffer('rope_inv_freq', 1.0 / (self.rope_theta ** (inv_idx / self.head_dim)), persistent=False)

    def _apply_rope(self, q: torch.Tensor, k: torch.Tensor, pos_q: torch.Tensor, pos_k: torch.Tensor):
        """
        Apply 1D RoPE along time for Q and K.
        q: (B,H,Nq,D), k: (B,H,Nk,D)
        pos_q: (Nq,), pos_k: (Nk,) integer positions (time indices)
        """
        device = q.device
        # Build cos/sin embeddings
        def build_cos_sin(pos):
            # pos: (N,)
            freqs = torch.einsum('n,d->nd', pos.float().to(device), self.rope_inv_freq.to(device))  # (N, D/2)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            # expand to D by interleaving
            cos = torch.stack([cos, cos], dim=-1).reshape(pos.shape[0], -1)  # (N, D)
            sin = torch.stack([sin, sin], dim=-1).reshape(pos.shape[0], -1)  # (N, D)
            return cos, sin

        def rotate_half(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
            return x_rot

        cos_q, sin_q = build_cos_sin(pos_q)  # (Nq, D)
        cos_k, sin_k = build_cos_sin(pos_k)  # (Nk, D)

        # reshape for broadcasting across batch and heads
        cos_q = cos_q.unsqueeze(0).unsqueeze(0)  # (1,1,Nq,D)
        sin_q = sin_q.unsqueeze(0).unsqueeze(0)
        cos_k = cos_k.unsqueeze(0).unsqueeze(0)  # (1,1,Nk,D)
        sin_k = sin_k.unsqueeze(0).unsqueeze(0)

        q_rope = (q * cos_q) + (rotate_half(q) * sin_q)
        k_rope = (k * cos_k) + (rotate_half(k) * sin_k)
        return q_rope, k_rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, F)
        returns: (B, C, T, F)
        """
        B, C, T, Fr = x.shape
        assert Fr == self.width, f"Expected width {self.width}, got {Fr}"

        # Prepare additive frequency RNN encoding per time step
        x_tfc = x.permute(0, 2, 3, 1).contiguous()  # (B, T, F, C)
        x_flat = x_tfc.reshape(B * T, Fr, C)
        freq_enc, _ = self.freq_rnn(x_flat)  # (B*T, F, C)
        freq_enc = freq_enc.reshape(B, T, Fr, C)
        y = x_tfc + freq_enc  # (B, T, F, C)

        # Projections
        q_all = self.q_proj(y)  # (B,T,F,C)
        k_all = self.k_proj(y)  # (B,T,F,C)
        v_all = self.v_proj(y)  # (B,T,F,C)

        # Split heads -> (B,H,T,F,D)
        def split_heads(t):
            return t.view(B, T, Fr, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4).contiguous()

        qh = split_heads(q_all)
        kh = split_heads(k_all)
        vh = split_heads(v_all)

        W = self.window_t
        # Build causal windows and projections
        def build_cos_sin(pos_mat: torch.Tensor):  # pos_mat: (T, M)
            freqs = torch.einsum('tm,d->tmd', pos_mat.float(), self.rope_inv_freq)  # (T,M,D/2)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            cos = torch.stack([cos, cos], dim=-1).reshape(freqs.shape[0], freqs.shape[1], -1)  # (T,M,D)
            sin = torch.stack([sin, sin], dim=-1).reshape(freqs.shape[0], freqs.shape[1], -1)  # (T,M,D)
            return cos, sin

        def rotate_half_nd(x):
            x1 = x[..., ::2]
            x2 = x[..., 1::2]
            return torch.stack((-x2, x1), dim=-1).reshape_as(x)

        t_idx = torch.arange(T, device=x.device)
        offsets = torch.arange(-W + 1, 1, device=x.device)

        if not self.separable_linformer:
            # Standard Linformer path with (W,F) windows
            kh_bhdtf = kh.permute(0, 1, 4, 2, 3).contiguous().view(B * self.num_heads, self.head_dim, T, Fr)
            vh_bhdtf = vh.permute(0, 1, 4, 2, 3).contiguous().view(B * self.num_heads, self.head_dim, T, Fr)
            pad2d = (0, 0, W - 1, 0)
            kh_pad = F.pad(kh_bhdtf, pad2d)
            vh_pad = F.pad(vh_bhdtf, pad2d)
            k_cols = F.unfold(kh_pad, kernel_size=(W, Fr), padding=(0, 0), stride=(1, 1))  # (B*H, D*W*F, T)
            v_cols = F.unfold(vh_pad, kernel_size=(W, Fr), padding=(0, 0), stride=(1, 1))
            k_win = k_cols.view(B, self.num_heads, self.head_dim, W * Fr, T).permute(0, 1, 4, 3, 2).contiguous()
            v_win = v_cols.view(B, self.num_heads, self.head_dim, W * Fr, T).permute(0, 1, 4, 3, 2).contiguous()

            pos_q_mat = t_idx[:, None].expand(T, Fr)  # (T,F)
            pos_k_time = t_idx[:, None] + offsets[None, :]  # (T,W)
            pos_k_mat = pos_k_time.repeat_interleave(Fr, dim=1)  # (T, W*F)
            cos_q, sin_q = build_cos_sin(pos_q_mat)
            cos_k, sin_k = build_cos_sin(pos_k_mat)

            q_rope = (qh * cos_q.unsqueeze(0).unsqueeze(0)) + (rotate_half_nd(qh) * sin_q.unsqueeze(0).unsqueeze(0))
            k_rope = (k_win * cos_k.unsqueeze(0).unsqueeze(0)) + (rotate_half_nd(k_win) * sin_k.unsqueeze(0).unsqueeze(0))

            k_proj = torch.einsum('b h t n d, n r -> b h t r d', k_rope, self.E)
            v_proj = torch.einsum('b h t n d, n r -> b h t r d', v_win, self.Fp)
            attn_scores = torch.einsum('b h t f d, b h t r d -> b h t f r', q_rope, k_proj) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn_scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('b h t f r, b h t r d -> b h t f d', attn, v_proj)
        else:
            # Separable Linformer path: time-only windows then freq projection
            kh_bhdtf = kh.permute(0, 1, 4, 2, 3).contiguous().view(B * self.num_heads, self.head_dim, T, Fr)
            vh_bhdtf = vh.permute(0, 1, 4, 2, 3).contiguous().view(B * self.num_heads, self.head_dim, T, Fr)
            pad2d = (0, 0, W - 1, 0)
            kh_pad = F.pad(kh_bhdtf, pad2d)
            vh_pad = F.pad(vh_bhdtf, pad2d)
            # Unfold only along time: kernel (W,1)
            k_cols = F.unfold(kh_pad, kernel_size=(W, 1), padding=(0, 0), stride=(1, 1))  # (B*H, D*W, T*F)
            v_cols = F.unfold(vh_pad, kernel_size=(W, 1), padding=(0, 0), stride=(1, 1))
            # (B,H,T,F,W,D)
            k_win_t = k_cols.view(B, self.num_heads, self.head_dim, W, T, Fr).permute(0, 1, 4, 5, 3, 2).contiguous()
            v_win_t = v_cols.view(B, self.num_heads, self.head_dim, W, T, Fr).permute(0, 1, 4, 5, 3, 2).contiguous()

            # RoPE only needs time indices; share across F
            # Q RoPE with (T,D)
            cos_q_t, sin_q_t = build_cos_sin(t_idx[:, None])  # (T,1,D)
            q_rope = (qh * cos_q_t.unsqueeze(0).unsqueeze(0)) + (rotate_half_nd(qh) * sin_q_t.unsqueeze(0).unsqueeze(0))
            # K RoPE with (T,W,D), broadcast over F
            pos_k_time = t_idx[:, None] + offsets[None, :]  # (T,W)
            cos_k_t, sin_k_t = build_cos_sin(pos_k_time)
            k_rope_t = (k_win_t * cos_k_t.unsqueeze(0).unsqueeze(0).unsqueeze(3)) + (rotate_half_nd(k_win_t) * sin_k_t.unsqueeze(0).unsqueeze(0).unsqueeze(3))

            # Temporal projection W -> r_t
            k_tp = torch.einsum('b h t f w d, w r -> b h t f r d', k_rope_t, self.Et_k)
            v_tp = torch.einsum('b h t f w d, w r -> b h t f r d', v_win_t, self.Et_v)
            # Frequency projection F -> r_f
            k_proj = torch.einsum('b h t f r d, f n -> b h t n r d', k_tp, self.Ef_k)
            v_proj = torch.einsum('b h t f r d, f n -> b h t n r d', v_tp, self.Ef_v)
            # Flatten rank dims
            r = self.rank_t * self.rank_f
            k_proj = k_proj.reshape(B, self.num_heads, T, r, self.head_dim)
            v_proj = v_proj.reshape(B, self.num_heads, T, r, self.head_dim)

            attn_scores = torch.einsum('b h t f d, b h t r d -> b h t f r', q_rope, k_proj) / math.sqrt(self.head_dim)
            attn = torch.softmax(attn_scores, dim=-1)
            attn = self.attn_drop(attn)
            out = torch.einsum('b h t f r, b h t r d -> b h t f d', attn, v_proj)

        # Merge heads -> (B,T,F,C) and project out
        out = out.permute(0, 2, 3, 1, 4).contiguous().view(B, T, Fr, self.embed_dim)
        out = self.out_proj(out)

        # Return to (B,C,T,F)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out


# === RetNet Retention机制 ===
class TFMultiheadRetention(nn.Module):
    """
    Time-Frequency joint multi-head Retention (RetNet) module.
    - Only time axis has causal/decay (no mask/decay across frequency).
    - Training and streaming both use O(T) recurrence along time (no O(T^2) masks).
    - Optional low-rank projection across frequency (F -> r -> F) to reduce compute/memory.
    Args:
        embed_dim: input/output channel dim
        num_heads: number of heads
        width: frequency bins
        decay: float in (0,1), retention decay factor (lambda)
        low_rank: if >0, use low-rank projection across frequency
        mode: 'train'|'streaming' (default 'train')
    """
    def __init__(self, embed_dim, num_heads, width, decay=0.9, low_rank=0, mode='train'):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.width = width
        self.decay = decay
        self.low_rank = low_rank
        self.mode = mode

        # Frequency positional encoding and frequency self-attention (per time step)
        self.freq_pos = nn.Parameter(torch.zeros(width, embed_dim))  # learnable absolute PE over F
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        self.fq_q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.fq_k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.fq_v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.fq_out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # Time retention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        if low_rank > 0:
            # Project frequency axis to low_rank (F -> r) and back (r -> F)
            self.E = nn.Parameter(torch.empty(width, low_rank))  # (F, r)
            nn.init.xavier_uniform_(self.E)
        else:
            self.E = None

        # For streaming mode: state cache for each head/freq
        self.register_buffer('_streaming_state', None, persistent=False)

    def reset_streaming_state(self, device=None, batch_size=1):
        # Called before streaming inference
        state = torch.zeros(batch_size, self.num_heads, self.width, self.head_dim, device=device)
        self._streaming_state = state

    def forward(self, x, streaming_update=False):
        """
        x: (B, C, T, F)
        streaming_update: if True, run in streaming (stateful) mode and still return all T outputs.
        returns: (B, C, T, F)
        """
        B, C, T, F = x.shape
        assert F == self.width
        # Prepare (B,T,F,C)
        x_tfc = x.permute(0, 2, 3, 1).contiguous()

        # 1) Frequency self-attention per time step (no mask/decay across frequencies)
        x_f = x_tfc + self.freq_pos.unsqueeze(0).unsqueeze(0)  # add F-pos
        # Projections for frequency attention
        Qf = self.fq_q_proj(x_f)  # (B,T,F,C)
        Kf = self.fq_k_proj(x_f)
        Vf = self.fq_v_proj(x_f)
        B_, T_, F_ = B, x_f.shape[1], x_f.shape[2]
        # Split heads: (B,T,H,F,D)
        def split_heads_f(t):
            return t.view(B_, T_, F_, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4).contiguous()
        Qfh = split_heads_f(Qf)
        Kfh = split_heads_f(Kf)
        Vfh = split_heads_f(Vf)
        # Attention across frequency axis per time step
        attn_f = torch.einsum('b t h f d, b t h g d -> b t h f g', Qfh, Kfh) / math.sqrt(self.head_dim)
        attn_f = nn.functional.softmax(attn_f, dim=-1)
        Of = torch.einsum('b t h f g, b t h g d -> b t h f d', attn_f, Vfh)  # (B,T,H,F,D)
        Of = Of.permute(0, 1, 3, 2, 4).contiguous().view(B_, T_, F_, self.embed_dim)
        y_f = self.fq_out_proj(Of)  # (B,T,F,C)
        # Residual
        y = x_tfc + y_f

        # 2) Time retention along T per frequency bin
        Q = self.q_proj(y)
        K = self.k_proj(y)
        V = self.v_proj(y)
        # Split heads: (B, T, F, H, D)
        def split_heads(t):
            return t.view(B, T, F, self.num_heads, self.head_dim)
        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Common permutations for both modes
        Qh = Qh.permute(0, 3, 2, 1, 4).contiguous()  # (B,H,F,T,D)
        Kh = Kh.permute(0, 3, 2, 1, 4).contiguous()
        Vh = Vh.permute(0, 3, 2, 1, 4).contiguous()

        if (self.mode == 'streaming') or streaming_update:
            # Stateful recurrence; return all T outputs and update internal state at the end
            decay = self.decay
            if self.low_rank > 0:
                # Project to r
                Qp = torch.einsum('bhftd,fr->bhrtd', Qh, self.E)  # (B,H,r,T,D)
                Kp = torch.einsum('bhftd,fr->bhrtd', Kh, self.E)
                Vp = torch.einsum('bhftd,fr->bhrtd', Vh, self.E)
                r = self.low_rank
                # init or reuse state: (B,H,r,D)
                if self._streaming_state is None or self._streaming_state.shape[0] != B or self._streaming_state.shape[2] != r:
                    self._streaming_state = torch.zeros(B, self.num_heads, r, self.head_dim, device=x.device, dtype=Qp.dtype)
                s = self._streaming_state  # (B,H,r,D)
                y_list = []
                for t in range(T):
                    s = decay * s + Kp[:, :, :, t, :] * Vp[:, :, :, t, :]
                    y_t = Qp[:, :, :, t, :] * s
                    y_list.append(y_t)
                Y = torch.stack(y_list, dim=3)  # (B,H,r,T,D)
                # Project back to F
                out = torch.einsum('bhrtd,fr->bhftd', Y, self.E)
                # update state
                self._streaming_state = s.detach()
            else:
                # State over F
                if self._streaming_state is None or self._streaming_state.shape[0] != B or self._streaming_state.shape[2] != F:
                    self._streaming_state = torch.zeros(B, self.num_heads, F, self.head_dim, device=x.device, dtype=Qh.dtype)
                s = self._streaming_state  # (B,H,F,D)
                y_list = []
                for t in range(T):
                    s = self.decay * s + Kh[:, :, :, t, :] * Vh[:, :, :, t, :]
                    y_t = Qh[:, :, :, t, :] * s
                    y_list.append(y_t)
                out = torch.stack(y_list, dim=3)  # (B,H,F,T,D)
                self._streaming_state = s.detach()
        else:
            # Training mode: non-stateful O(T^2) retention using lower-tri decay along time
            idx = torch.arange(T, device=x.device)
            diff = (idx[None, :] - idx[:, None]).to(Qh.dtype)  # (T,T)
            decay_mask = torch.where(diff >= 0, (self.decay ** diff), torch.zeros_like(diff))  # (T,T)
            if self.low_rank > 0:
                Qp = torch.einsum('bhftd,fr->bhrtd', Qh, self.E)  # (B,H,r,T,D)
                Kp = torch.einsum('bhftd,fr->bhrtd', Kh, self.E)
                Vp = torch.einsum('bhftd,fr->bhrtd', Vh, self.E)
                # qk: (B,H,r,T,T,D)
                qk = Qp.unsqueeze(3) * Kp.unsqueeze(2)
                qk = qk * decay_mask[None, None, None, :, :, None]
                Y = torch.einsum('bhrttd,bhrtd->bhrtd', qk, Vp)
                out = torch.einsum('bhrtd,fr->bhftd', Y, self.E)
            else:
                # (B,H,F,T,D)
                qk = Qh.unsqueeze(3) * Kh.unsqueeze(2)  # (B,H,F,T,T,D)
                qk = qk * decay_mask[None, None, None, :, :, None]
                out = torch.einsum('bhfttd,bhftd->bhftd', qk, Vh)  # sum over s

        # (B,H,F,T,D) -> (B,T,F,H,D) -> (B,T,F,C) -> (B,C,T,F)
        out = out.permute(0, 3, 2, 1, 4).contiguous().view(B, T, F, self.embed_dim)
        out = self.out_proj(out)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(5*2,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(1*2,1), dilation=(1,1), use_deconv=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class GTCRN(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()
        self.tf_att1 = TFMultiheadRetention(16, 4, 33, decay=0.9, low_rank=10, mode="training")
        self.tf_att2 = TFMultiheadRetention(16, 4, 33, decay=0.9, low_rank=10, mode="training")

        self.decoder = Decoder()

        self.mask = Mask()

    def forward(self, x):
        """
        x: (B, L)
        """
        device = x.device
        n_samples = x.shape[1]
        
        stft_kwargs = {'n_fft': self.n_fft, 'hop_length': self.hop_len, 'win_length': self.win_len,
                       'window': torch.hann_window(self.win_len).to(device), 'onesided': True}
        
        spec = torch.stft(x,  **stft_kwargs, return_complex=True)
        spec = torch.view_as_real(spec)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        
        spec = spec.permute(0,3,2,1)  # (B,2,T,F)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        feat, en_outs = self.encoder(feat)
        
        feat = self.tf_att1(feat) # (B,16,T,33)
        feat = self.tf_att2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


if __name__ == "__main__":
    calculate_macs_mode = True
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    try:
        flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                                print_per_layer_stat=True, verbose=False, backend='aten')
        params = 0
        for p in model.parameters():
            params += p.numel()
        print(flops, params/1e3) 
    except ValueError:
        print("ptflops not support aten backend")   

    """causality check"""
    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    y1 = model(x1)[0]
    y2 = model(x2)[0]

    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
