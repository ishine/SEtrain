"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params
Refactored version of gtcrn_dynamic_repconv4.py using shared modules.
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function

from .modules.layers import ERB, SFE, ConvBlock, Mask
from .modules.conv import RepConvBlock
from .modules.rnn import DPRNN

CHANNELS = 16
OVER_PARAM_FACTOR = 5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, CHANNELS, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True)
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
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(5*2,1), dilation=(5,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(1*2,1), dilation=(1,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR, use_1d_kernel=True),
            ConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(CHANNELS, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
        return x
    

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
        
        self.dpgrnn1 = DPRNN(CHANNELS, 33, CHANNELS)
        self.dpgrnn2 = DPRNN(CHANNELS, 33, CHANNELS)
        
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
        
        feat = self.dpgrnn1(feat) # (B,CHANNELS,T,33)
        feat = self.dpgrnn2(feat) # (B,CHANNELS,T,33)

        m_feat = self.decoder(feat, en_outs)
        
        m = self.erb.bs(m_feat)

        spec_enh = self.mask(m, spec) # (B,2,T,F)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        
        return output
