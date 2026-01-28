"""
GTCRN V2: Intermediate version between Original and V1
Scale:
1. CHANNELS: 16 (Orig) -> 24 (V2) -> 32 (V1)
2. RepConvBlocks: 3 (Orig) -> 4 (V2) -> 4 (V1)
3. DPRNN layers: 2 (Orig) -> 2 (V2) -> 3 (V1)
4. Subbands: 129 (Orig) -> 145 (V2) -> 161 (V1)
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function

try:
    from .modules import conv
    from .modules.layers import ERB, SFE, ConvBlock, Mask
    from .modules.conv import RepConvBlock
    from .modules.rnn import DPRNN
except ImportError:
    from modules import conv
    from modules.layers import ERB, SFE, ConvBlock, Mask
    from modules.conv import RepConvBlock
    from modules.rnn import DPRNN

CHANNELS = 24
OVER_PARAM_FACTOR = 5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            ConvBlock(3*3, CHANNELS, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False),
            ConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(9,1), use_deconv=False, over_param_factor=OVER_PARAM_FACTOR)
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
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(9*2,1), dilation=(9,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(5*2,1), dilation=(5,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR),
            RepConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(1*2,1), dilation=(1,1), use_deconv=True, over_param_factor=OVER_PARAM_FACTOR),
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
        
        subbands = 145
        self.erb = ERB(65, subbands - 65)
        self.sfe = SFE(3, 1)

        self.encoder = Encoder()
        
        width = subbands
        width = (width - 1) // 2 + 1
        width = (width - 1) // 2 + 1
        
        self.dpgrnn1 = DPRNN(CHANNELS, width, CHANNELS)
        self.dpgrnn2 = DPRNN(CHANNELS, width, CHANNELS)
        
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
        
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)

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
    conv.CALCULATE_MACS_MODE = True

    """complexity count"""
    try:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                                print_per_layer_stat=True, verbose=True, backend="aten")
        params = 0
        for p in model.parameters():
            params += p.numel()
        print(flops, params/1e3)
    except ImportError:
        print("ptflops not installed, skipping complexity count")

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

    conv.CALCULATE_MACS_MODE = False