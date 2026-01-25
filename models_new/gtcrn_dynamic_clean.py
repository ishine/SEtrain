"""
Refactored GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Using Dynamic TRA and Conv2dAttention
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.profiler import record_function

from .modules import ERB, SFE, DynamicTRA, ConvBlock, Mask, DPGRNN, Conv2dAttention, set_calculate_macs_mode

CHANNELS = 32

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
        
        # Original: self.tra = TRA(in_channels//2, kernel_choices, 3) 
        # Here 3 corresponds to kat_heads=3 (used for point1, depth, point2)
        self.tra = DynamicTRA(in_channels//2, kernel_choices, 3)

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
        
        self.dpgrnn1 = DPGRNN(CHANNELS, 33, CHANNELS)
        self.dpgrnn2 = DPGRNN(CHANNELS, 33, CHANNELS)
        
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

if __name__ == "__main__":
    set_calculate_macs_mode(True)
    model = GTCRN().eval()

    """complexity count"""
    try:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                                print_per_layer_stat=True, verbose=False, backend='aten')
        params = 0
        for p in model.parameters():
            params += p.numel()
        print(flops, params/1e3)
    except ImportError:
        print("ptflops not installed, skipping complexity count")

    """causality check"""
    # Important: Reset MACS mode if needed, or keeping it True matches original script intent for this check?
    # Original script keeps it True for complexity count, and then...
    # CAUSALITY CHECK:
    # In original: it defines `calculate_macs_mode = False` at top.
    # __main__ sets `calculate_macs_mode = True` and then logs complexity.
    # It does NOT reset it to False before causality check.
    # Wait, causality check uses `model(x1)`.
    # If `calculate_macs_mode` is True, `Conv2dAttention` uses `self.conv` (the grouped kernel construction).
    # If `calculate_macs_mode` is False, it uses `self.conv_and_sum`.
    # Original code set it to True. So it tests the `conv` path.
    # I should check if `conv` path preserves causality. `pad_left` is used. Unfold uses padding.
    # So I will leave it as True to match original script behavior.

    a = torch.randn(1, 16000)
    b = torch.randn(1, 16000)
    c = torch.randn(1, 16000)
    x1 = torch.cat([a, b], dim=1)
    x2 = torch.cat([a, c], dim=1)

    y1 = model(x1)[0]
    y2 = model(x2)[0]

    print((y1[:16000-256*2] - y2[:16000-256*2]).abs().max())
    print((y1[16000:] - y2[16000:]).abs().max())
