import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parameter import Parameter

# ==========================================
# From dpr_layer.py
# ==========================================

class CustomLayerNorm(nn.Module):
    def __init__(self, input_dims, stat_dims=(1,), num_dims=4, eps=1e-5):
        super().__init__()
        assert isinstance(input_dims, tuple) and isinstance(stat_dims, tuple)
        assert len(input_dims) == len(stat_dims)
        param_size = [1] * num_dims
        for input_dim, stat_dim in zip(input_dims, stat_dims):
            param_size[stat_dim] = input_dim
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))
        init.ones_(self.gamma)
        init.zeros_(self.beta)
        self.eps = eps
        self.stat_dims = stat_dims
        self.num_dims = num_dims

    def forward(self, x):
        assert x.ndim == self.num_dims, print(
            "Expect x to have {} dimensions, but got {}".format(self.num_dims, x.ndim))

        mu_ = x.mean(dim=self.stat_dims, keepdim=True)  # [B,1,T,F]
        std_ = torch.sqrt(
            x.var(dim=self.stat_dims, unbiased=False, keepdim=True) + self.eps
        )  # [B,1,T,F]
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta
        return x_hat


class RNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            dropout_p=0.1,
            bidirectional=False,
    ):
        super().__init__()
        self.rnn = nn.GRU(emb_dim, hidden_dim, 1, batch_first=True, bidirectional=bidirectional)
        if bidirectional:
            self.dense = nn.Linear(hidden_dim * 2, emb_dim)
        else:
            self.dense = nn.Linear(hidden_dim, emb_dim)
    
    def forward(self, x):
        # x:(b,t,d)
        x,_ = self.rnn(x)
        x = self.dense(x)
        return x


class DualPathRNN(nn.Module):
    def __init__(
            self,
            emb_dim,
            hidden_dim,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.intra_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.intra_rnn_attn = RNN(emb_dim, hidden_dim // 2, dropout_p, bidirectional=True)

        self.inter_norm = nn.LayerNorm((n_freqs, emb_dim))
        self.inter_rnn_attn = RNN(emb_dim, hidden_dim, dropout_p, bidirectional=False)


    def forward(self, x):
        # x:(b,d,t,f)
        B, D, T, F = x.size()
        x = x.permute(0, 2, 3, 1)  # (b,t,f,d)

        x_res = x
        x = self.intra_norm(x)
        x = x.reshape(B * T, F, D)  # (b*t,f,d)
        x = self.intra_rnn_attn(x)
        x = x.reshape(B, T, F, D)
        x = x + x_res

        x_res = x
        x = self.inter_norm(x)
        x = x.permute(0, 2, 1, 3)  # (b,f,t,d)
        x = x.reshape(B * F, T, D)
        x = self.inter_rnn_attn(x)
        x = x.reshape(B, F, T, D).permute(0, 2, 1, 3) # (b,t,f,d)
        x = x + x_res

        x = x.permute(0, 3, 1, 2)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, emb_dim, n_freqs=32, expansion_factor=2, dropout_p=0.1):
        super().__init__()
        hidden_dim = int(emb_dim * expansion_factor)
        self.norm = CustomLayerNorm((emb_dim, n_freqs), stat_dims=(1, 3))
        self.fc1 = nn.Conv2d(emb_dim, hidden_dim * 2, 1)
        self.dwconv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 2, 0), value=0.0),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, groups=hidden_dim),
        )
        self.act = nn.Mish()
        self.fc2 = nn.Conv2d(hidden_dim, emb_dim, 1)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # x:(b,d,t,f)
        res = x
        x = self.norm(x)
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.dropout(x)
        x = self.fc2(x)
        x = x + res
        return x


class DPR(nn.Module):
    def __init__(
            self,
            emb_dim=16,
            hidden_dim=24,
            n_freqs=32,
            dropout_p=0.1,
    ):
        super().__init__()
        self.dp_rnn_attn = DualPathRNN(emb_dim, hidden_dim, n_freqs, dropout_p)
        self.conv_glu = ConvolutionalGLU(emb_dim, n_freqs=n_freqs, expansion_factor=2, dropout_p=dropout_p)

    def forward(self, x):
        x = self.dp_rnn_attn(x)
        x = self.conv_glu(x)
        return x

# ==========================================
# From generator.py
# ==========================================

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1, 1))
        self.slope.requires_grad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 4
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 3)),
        )
        self.high_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(2, 5), stride=(1, 3)),
        )
        self.norm = CustomLayerNorm((1, n_freqs // 2), stat_dims=(1, 3))
        self.act = nn.PReLU(out_channels)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]
        
        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)

        x = torch.cat([x_low, x_high], dim=-1)
        x = self.norm(x)
        x = self.act(x)
        return x


class USConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_freqs):
        super().__init__()
        self.low_freqs = n_freqs // 2
        self.low_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 0), value=0.0),
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3)),
        )
        self.high_conv = SPConvTranspose2d(in_channels, out_channels, kernel_size=(1, 3), r=3)
    
    def forward(self, x):
        # (b,d,t,f)
        x_low = x[..., :self.low_freqs]
        x_high = x[..., self.low_freqs:]

        x_low = self.low_conv(x_low)
        x_high = self.high_conv(x_high)
        x = torch.cat([x_low, x_high], dim=-1)
        return x


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad = nn.ConstantPad2d((kernel_size[1]//2, kernel_size[1]//2, kernel_size[0]-1, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1)
        )
        self.r = r

    def forward(self, x):
        x = self.pad(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, num_channels=16):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, num_channels//4, (1, 1), (1, 1)),
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(num_channels // 4),
        )
        
        self.conv_2 = DSConv(num_channels//4, num_channels//2, n_freqs=257)
        self.conv_3 = DSConv(num_channels//2, num_channels//4*3, n_freqs=128)
        self.conv_4 = DSConv(num_channels//4*3, num_channels, n_freqs=64)


    def forward(self, x):
        out_list = []
        x = self.conv_1(x)
        x = self.conv_2(x)
        out_list.append(x)  # 128
        x = self.conv_3(x)
        out_list.append(x)  # 64
        x = self.conv_4(x)
        out_list.append(x)  # 32
        return out_list


class MaskDecoder(nn.Module):
    def __init__(self, num_features, num_channels=64, out_channel=2, beta=1):
        super(MaskDecoder, self).__init__()
        self.up1 = USConv(num_channels * 2, num_channels // 4 * 3, n_freqs=32)
        self.up2 = USConv(num_channels // 4 * 3 * 2, num_channels // 2, n_freqs=64)  # 128
        self.up3 = USConv(num_channels // 2 * 2, num_channels // 4, n_freqs=128)  # 256
        self.mask_conv = nn.Sequential(
            nn.ConstantPad2d((1, 1, 1, 0), value=0.0),
            nn.Conv2d(num_channels // 4, out_channel, (2, 2)), # 257
            CustomLayerNorm((1, 257), stat_dims=(1, 3)),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1)),
        )
        self.lsigmoid = LearnableSigmoid2d(num_features, beta=beta)

    def forward(self, x, encoder_out_list):
        x = self.up1(torch.cat([x, encoder_out_list.pop()], dim=1))  # 64
        x = self.up2(torch.cat([x, encoder_out_list.pop()], dim=1))  # 128
        x = self.up3(torch.cat([x, encoder_out_list.pop()], dim=1))  # 256
        x = self.mask_conv(x)  # (B,out_channel,T,F)
        x = x.permute(0, 3, 2, 1)  # (B,F,T,out_channel)
        x = self.lsigmoid(x).permute(0, 3, 2, 1)
        return x


class LiSenNet(nn.Module):
    def __init__(self, n_fft=512, hop_len=256, win_len=512, num_channels=16, n_blocks=2, compress_factor=0.3):
        super(LiSenNet, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_len
        self.win_len = win_len
        self.n_freqs = n_fft // 2 + 1
        self.compress_factor = compress_factor

        self.encoder = Encoder(in_channels=3, num_channels=num_channels)

        self.blocks = nn.Sequential(
            *[DPR(
                emb_dim=num_channels,
                hidden_dim=num_channels // 2 * 3,
                n_freqs=self.n_freqs // (2 ** 3),
                dropout_p=0.1,
            ) for _ in range(n_blocks)]
        )

        self.decoder = MaskDecoder(self.n_freqs, num_channels=num_channels, out_channel=2, beta=1)

    def apply_stft(self, x, return_complex=True):
        # x:(B,T)
        assert x.ndim == 2
        win_length = self.win_len if self.win_len else self.n_fft
        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            onesided=True,
            return_complex=return_complex,
        ).transpose(1, 2)  # (B,T,F)
        return spec

    def apply_istft(self, x, length=None):
        # x:(B,T,F)
        assert x.ndim == 3
        x = x.transpose(1, 2)  # (B,F,T)
        win_length = self.win_len if self.win_len else self.n_fft
        audio = torch.istft(
            x,
            self.n_fft,
            self.hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(x.device),
            onesided=True,
            length=length,
            return_complex=False
        )  # (B,T)
        return audio

    def power_compress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** self.compress_factor
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))

    def power_uncompress(self, x):
        # x:(B,T,F)
        mag = torch.abs(x) ** (1.0 / self.compress_factor)
        phase = torch.angle(x)
        return torch.complex(mag * torch.cos(phase), mag * torch.sin(phase))
    
    @staticmethod
    def cal_gd(x):
        # x: (B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_gd = torch.diff(x, dim=2, prepend=torch.zeros(b, t, 1, device=x.device))  # (-2pi, 2pi]
        return torch.atan2(x_gd.sin(), x_gd.cos())

    def cal_ifd(self, x):
        # x:(B,T,F), return (-pi, pi]
        b, t, f = x.size()
        x_if = torch.diff(x, dim=1, prepend=torch.zeros(b, 1, f, device=x.device))  # (-2pi, 2pi]
        x_ifd = x_if - 2 * torch.pi * (self.hop_length / self.n_fft) * torch.arange(f, device=x.device)[None, None, :]
        return torch.atan2(x_ifd.sin(), x_ifd.cos())

    def griffinlim(self, mag, pha=None, length=None, n_iter=2, momentum=0.99):
        mag = mag.detach()
        mag = mag ** (1.0 / self.compress_factor) # uncompress
        assert 0 <= momentum < 1
        momentum = momentum / (1 + momentum)
        if pha is None:
            pha = torch.rand(mag.size(), dtype=mag.dtype, device=mag.device)

        tprev = torch.tensor(0.0, dtype=mag.dtype, device=mag.device)
        for _ in range(n_iter):
            inverse = self.apply_istft(torch.complex(mag * pha.cos(), mag * pha.sin()), length=length)
            rebuilt = self.apply_stft(inverse)
            pha = rebuilt
            pha = pha - tprev.mul_(momentum)
            pha = pha.angle()
            tprev = rebuilt

        return pha

    def forward(self, x):
        """
        x: (B, L)
        """
        n_samples = x.shape[1]
        
        src_spec = self.power_compress(self.apply_stft(x))  # (B,T,F)
        src_mag = src_spec.abs()
        src_pha = src_spec.angle()
        src_gd = self.cal_gd(src_pha)
        src_ifd = self.cal_ifd(src_pha)

        inp = torch.stack([src_mag, src_gd / torch.pi, src_ifd / torch.pi], dim=1)  # (B,3,T,F)

        encoder_out_list = self.encoder(inp)
        out = self.blocks(encoder_out_list[-1])
        out = self.decoder(out, encoder_out_list)  # (B,2,T,F)

        est_mag = (out[:, 0] + 1e-8) * src_mag + (out[:, 1] + 1e-8) * src_mag

        est_pha = self.griffinlim(est_mag.detach(), src_pha, n_samples)
        est_spec = torch.complex(est_mag * est_pha.cos(), est_mag * est_pha.sin())
        est = self.apply_istft(self.power_uncompress(est_spec), length=n_samples)
        
        return est

if __name__ == "__main__":
    model = LiSenNet().eval()

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                            print_per_layer_stat=True, verbose=True, backend="aten")
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

    y1 = model(x1)
    y2 = model(x2)

    print((y1[:, :16000-256*4] - y2[:, :16000-256*4]).abs().max())
    print((y1[:, :16000] - y2[:, :16000]).abs().max())
