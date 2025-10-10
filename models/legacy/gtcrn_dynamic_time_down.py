"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params -> 93.28 MMac 89.429 K params ?
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torch.profiler import record_function


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
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False, 
                 time_padding=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.time_padding = time_padding
        if not self.time_padding:
            self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        if self.time_padding and not use_deconv:
            self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding=(0,padding[1]), groups=groups)
        else:
            self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding=(padding[0] * 2, padding[1]), groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        self.kernel_size = kernel_size
        self.stride = stride
        self.transposed = use_deconv
        self.padding = padding
    def forward(self, x):
        if self.time_padding and not self.transposed:
            padding_t = (- (x.shape[2] + self.padding[0] * 2 - self.kernel_size[0] + 1 - 1)) % self.stride[0]
            x = F.pad(x, [0, 0, self.padding[0] * 2, padding_t])
        if self.time_padding and self.transposed:
            x_padded_t_size = self.stride[0] * (x.shape[2] - 1) + self.kernel_size[0] - 2 * self.padding[0]
            re_padding_t = (- (x_padded_t_size + self.padding[0] * 4 - self.kernel_size[0] + 1 - 1)) % self.stride[0]
            # print("Re_padding_t:", re_padding_t)
            x_pretend_size = x_padded_t_size + re_padding_t + self.padding[0] * 4
            target_size = (x_pretend_size - self.kernel_size[0]) // self.stride[0] + 1
            x = F.pad(x, [0, 0, target_size - x.shape[2], 0])
            pass
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
            return self.deconv_and_sum(x, attention)
        return self.conv_and_sum(x, attention)


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


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, 16, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(16, 16, (5,1), stride=(5,1), padding=(2,0), groups=2, use_deconv=False, is_last=False, time_padding=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            ConvBlock(16, 16, (5,1), stride=(5,1), padding=(2,0), groups=2, use_deconv=False, is_last=False, time_padding=True),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False),
            GTConvBlock(16, 16, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False)
        ])  # padding issue 32->31

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
            ConvBlock(16, 16, (5,1), stride=(5,1), padding=(2,0), groups=2, use_deconv=True, is_last=False, time_padding=True),
            ConvBlock(16, 16, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16, 16, (5,1), stride=(5,1), padding=(2,0), groups=2, use_deconv=True, is_last=False, time_padding=True),
            ConvBlock(16, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            last_out = en_outs[N_layers-1-i]
            x = x[:, :, :last_out.shape[2], :last_out.shape[3]]
            x = self.de_convs[i](x + last_out)
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
        
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        
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
        
        feat = self.dpgrnn1(feat) # (B,16,T,33)
        feat = self.dpgrnn2(feat) # (B,16,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output


if __name__ == "__main__":
    model = GTCRN().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                            print_per_layer_stat=True, verbose=True, backend='aten')
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
