"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN (Mamba)
Dynamic Version
"""
import profile
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function
from mamba_ssm import Mamba
import torch.nn.functional as F

CALCULATE_MACS_MODE = False
DO_CALCULATION_CONPENSATION = True

CHANNELS = 16


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
     
    def deconv(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        unfolded = F.unfold(x_pad, (self.kernel_size[0] * self.dilation[0] - self.dilation[0] + 1, 1), dilation=(1, 1), padding=(0, 0), stride=(self.stride[0], 1))
        unfolded2 = rearrange(unfolded, "b (c p) (t f) -> 1 (b t c) p f", c=x.shape[1], p=self.kernel_size[0] * self.dilation[0] - self.dilation[0] + 1, t=x.shape[2], f=x.shape[3])
        # x: (B, C, T, F)
        # unfolded: (B, C * K_T, T * F) -> (1, B*T*C, K_T, F)
        # attention: (B, K, T)
        grouped_kernels = torch.einsum("kiopq, bkt -> btiopq", self.candidates, attention)
        grouped_kernels = rearrange(grouped_kernels, "b t i o p q -> (b t o) i p q")
        grouped_bias = torch.einsum("ko, bkt -> bto", self.candidates_bias, attention)
        grouped_bias = rearrange(grouped_bias, "b t o -> (b t o)")
        out = F.conv_transpose2d(unfolded2, grouped_kernels, bias=grouped_bias, stride=self.stride, padding=self.padding, groups=x.shape[0]*x.shape[2]*self.groups, dilation=self.dilation)
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
            if not CALCULATE_MACS_MODE:
                return self.deconv_and_sum(x, attention)
            return self.deconv(x, attention)
        if not CALCULATE_MACS_MODE:
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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, CHANNELS, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            with record_function(f"Encoder_layer_{i}: " + "ConvBlock" if i<2 else "GTConvBlock"):
                x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(5*2,1), dilation=(5,1), use_deconv=True),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(1*2,1), dilation=(1,1), use_deconv=True),
            ConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(CHANNELS, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            with record_function(f"Decoder_layer_{i}: " + "GTConvBlock" if i<3 else "ConvBlock"):
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
    

class BiMamba(Mamba):
    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        """
        if CALCULATE_MACS_MODE:
            return self.forward_python(hidden_states)
        else:
            return super().forward(hidden_states, inference_params=inference_params)

    def forward_python(self, hidden_states):
        """
        Slow, python-based implementation for FLOPs counting.
        """
        batch, seqlen, dim = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        xz = self.in_proj(hidden_states) # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1) # (B, L, d_inner)
        
        # to (B, d_inner, L)
        x = x.transpose(1, 2)
        x = self.conv1d(x)[:, :, :seqlen]
        x = x.transpose(1, 2)
        
        x = F.silu(x)
        
        # x_proj: (B, L, d_inner) -> (B, L, dt_rank + 2*d_state)
        x_dbl = self.x_proj(x)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        dt = self.dt_proj(dt) # (B, L, d_inner)
        dt = F.softplus(dt)
        
        # A_log is (d_inner, d_state)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)
        
        h = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype) # (B, D, N)
        
        y_list = []
        
        # Scan loop
        for t in range(seqlen):
            xt = x[:, t, :] # (B, D)
            dt_t = dt[:, t, :] # (B, D)
            Bt = B[:, t, :] # (B, N)
            Ct = C[:, t, :] # (B, N)
            
            # shapes:
            # A: (D, N)
            # h: (B, D, N)
            
            # dt_t: (B, D) -> (B, D, 1)
            dt_t_exp = dt_t.unsqueeze(-1) 
            
            # dA = exp(dt * A)
            dA = torch.exp(dt_t_exp * A) # (B, D, N)
            # dB = dt * B
            dB = dt_t_exp * Bt.unsqueeze(1) # (B, D, N)
            
            # Update h: h = h * dA + dB * x
            h = h * dA + dB * xt.unsqueeze(-1) 
            
            # Compute y = (h * C).sum(-1)
            yt = (h * Ct.unsqueeze(1)).sum(dim=-1) # (B, D)
            
            y_list.append(yt)
            
        y = torch.stack(y_list, dim=1) # (B, L, D)
        
        y = y + x * self.D # Residual
        
        y = y * F.silu(z)
        
        # Scan MACs approx: L * D * N * 3 (update state + compute output)
        if CALCULATE_MACS_MODE and DO_CALCULATION_CONPENSATION:
            scan_macs = int(seqlen * self.d_inner * self.d_state * 3) # (1, K) @ (K, 1) = K MACs
            if scan_macs > 0:
                dummy_v = torch.zeros(1, scan_macs, device=device)
                dummy_w = torch.zeros(scan_macs, 1, device=device)
                torch.mm(dummy_v, dummy_w)
        
        out = self.out_proj(y)
        return out


class PureMambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        
        self.mamba = BiMamba(
            d_model=d_model,
            d_state=16, 
            d_conv=4,   
            expand=2
        )

    def forward(self, x):
        # x: (B, T, D)
        res = x
        x = self.norm(x)
        x = self.mamba(x)
        return res + x

class MambaEnhancer(nn.Module):
    def __init__(
        self,
        n_fft=512,
        hop_len=256,
        win_len=512,
        d_model=128,
        n_layers=3
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb = ERB(65, 64) 
        self.sfe = SFE(3, 1)   
        
        # (B, 9, T, 129) -> (B, T, d_model)
        self.input_conv = nn.Conv2d(CHANNELS, 3, kernel_size=1) 
        self.input_fc = nn.Linear(33 * 3, d_model) 
        self.input_act = nn.PReLU()

        # Mamba
        self.backbone = nn.Sequential(*[
            PureMambaBlock(d_model) for _ in range(n_layers)
        ])

        # (B, T, d_model) -> (B, 2, T, 129)
        self.output_fc = nn.Linear(d_model, 33 * CHANNELS)
        
        self.output_act = nn.Tanh() 

        self.mask = Mask()

    def forward(self, x):
        # (B, C, T, F) -> (B, 3, T, F)
        x_in = self.input_conv(x) 
        # (B, 3, T, 33) -> (B, T, 99)
        x_in = x_in.permute(0, 2, 1, 3).reshape(x_in.shape[0], x_in.shape[2], -1)
        # (B, T, 99) -> (B, T, d_model)
        x_in = self.input_act(self.input_fc(x_in))

        # (B, T, d_model)
        x_mem = self.backbone(x_in)

        # (B, T, d_model) -> (B, T, 258)
        x_out = self.output_act(self.output_fc(x_mem))
        
        # Reshape to (B, C, T, 33)
        output = x_out.view(x_out.shape[0], x_out.shape[1], CHANNELS, 33).permute(0, 2, 1, 3)
        
        return output



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
        
        self.mamba_enhancer = MambaEnhancer()
        
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

        with record_function("Encoder"):
            feat, en_outs = self.encoder(feat)
        
        with record_function("DPGRNN"):
            feat = self.mamba_enhancer(feat)

        with record_function("Decoder"):
            m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


if __name__ == "__main__":
    CALCULATE_MACS_MODE = True
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                            print_per_layer_stat=True, verbose=False, backend="aten")
    params = 0
    for p in model.parameters():
        params += p.numel()
    print(flops, params/1e3)

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
