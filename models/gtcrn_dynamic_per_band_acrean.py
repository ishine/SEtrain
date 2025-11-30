"""
GTCRN: ShuffleNetV2 + SFE + TRA + 2 DPGRNN
Ultra tiny, 33.0 MMACs, 23.67 K params -> 93.28 MMac 89.429 K params ?
VERY IMPORTANT: based on gtcrn_dynamic_per_band_cr.py
                change implementation of BConvBlock, 
                change implementation of Conv2dAttention, implementations now consistent,
                                                          but MACs calculation has problems
"""
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import math
from torch.profiler import record_function
from typing import Tuple, Optional

calculate_macs_mode = False
CHANNELS = 16


def partial_conv2d_w_functional(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    out_w_start: int,
    out_w_end: int
) -> torch.Tensor:
    """
    以函数式的方式，仅计算二维卷积在W维度上特定输出范围的结果。
    此版本通过手动padding解决了所有边界条件下的尺寸问题。
    """
    if out_w_start > out_w_end:
        raise ValueError("out_w_start cannot be greater than out_w_end.")

    W_in = input_tensor.shape[3]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    
    # 1. 计算理论上需要的输入窗口的全局索引
    in_w_start = out_w_start * stride_w - padding_w
    in_w_end = (out_w_end * stride_w - padding_w) + (kernel_w - 1) * dilation_w

    # 2. 从输入张量中切出实际存在的部分
    slice_w_start = max(0, in_w_start)
    slice_w_end = min(W_in, in_w_end + 1)
    
    # 如果所需范围完全在输入张量之外，则提前返回空张量
    if slice_w_start >= slice_w_end and (in_w_start >= W_in or in_w_end < 0):
         H_out = math.floor((input_tensor.shape[2] + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
         C_out = weight.shape[0]
         return torch.empty(
             input_tensor.shape[0], C_out, H_out, 0,
             device=input_tensor.device, dtype=input_tensor.dtype
         )

    input_slice = input_tensor[:, :, :, slice_w_start:slice_w_end]
    
    # 3. 计算需要在切片左右两侧手动补充的 padding
    left_pad = slice_w_start - in_w_start
    right_pad = (in_w_end + 1) - slice_w_end
    
    # 4. 使用 F.pad 进行精确的手动填充
    # F.pad 的填充顺序是 (左, 右, 上, 下)
    padded_input = F.pad(input_slice, (left_pad, right_pad, 0, 0))
    
    # 5. 使用 F.conv2d 进行计算，此时水平padding必须为0，垂直padding保持不变
    # 因为水平方向的padding已经手动完成了
    output_slice = F.conv2d(
        padded_input, weight, bias, stride, padding=(padding_h, 0), 
        dilation=dilation, groups=groups
    )
    
    return output_slice


def partial_conv_transpose2d_w_functional(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
    output_padding: Tuple[int, int],
    dilation: Tuple[int, int],
    groups: int,
    out_w_start: int,
    out_w_end: int
) -> torch.Tensor:
    """
    以函数式的方式，仅计算二维转置卷积在W维度上特定输出范围的结果。
    """
    if out_w_start > out_w_end:
        raise ValueError("out_w_start cannot be greater than out_w_end.")

    W_in = input_tensor.shape[3]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    out_padding_h, out_padding_w = output_padding
    
    in_w_end = math.floor((out_w_end + padding_w) / stride_w)
    numerator = out_w_start + padding_w - (kernel_w - 1) * dilation_w
    in_w_start = math.ceil(numerator / stride_w)
    
    in_w_start, in_w_end = int(in_w_start), int(in_w_end)

    slice_w_start = max(0, in_w_start)
    slice_w_end = min(W_in, in_w_end + 1)

    if slice_w_start >= slice_w_end:
        H_out = (input_tensor.shape[2] - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + out_padding_h + 1
        C_out = weight.shape[1] * groups # In conv_transpose, out_channels is at index 1
        return torch.empty(
            input_tensor.shape[0], C_out, H_out, 0,
            device=input_tensor.device, dtype=input_tensor.dtype
        )
        
    input_slice = input_tensor[:, :, :, slice_w_start:slice_w_end]

    full_slice_output = F.conv_transpose2d(
        input_slice, weight, bias, stride, padding,
        output_padding, groups, dilation
    )

    output_origin_offset = slice_w_start * stride_w
    
    relative_start = out_w_start - output_origin_offset
    relative_end = out_w_end - output_origin_offset

    final_start = max(0, relative_start)
    final_end = min(full_slice_output.shape[3], relative_end + 1)
    
    return full_slice_output[:, :, :, final_start:final_end]

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
        self.padding = padding
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False, bands=1, f_out=None):
        """
        Band-aware conv/deconv that pre-splits output frequency bins into equal bands.

        Args:
            in_channels: input channels
            out_channels: output channels
            kernel_size: (kT, kF) or int
            stride: (sT, sF) or int
            padding: (pT, pF) or int
            groups: groups for depthwise/group conv
            use_deconv: if True, use conv_transpose2d
            is_last: if True, use Tanh activation instead of PReLU
            bands: number of equal bands along output F to mix from separate candidate kernels
            f_out: expected output frequency bins for this layer. Required to precompute band splits.
        """
        super().__init__()
        assert out_channels % groups == 0 and in_channels % groups == 0
        # Store conv hyperparams
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.use_deconv = use_deconv
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bands = max(1, bands)
        self.f_out = int(f_out)

        #   candidates: (K, I, O, kT, kF) with I already divided by groups
        #   candidates_bias: (K, O)
        self.candidates_bias = nn.Parameter(torch.empty((self.num_bands, out_channels), requires_grad=True))
        cand_in, cand_out = in_channels, out_channels
        if use_deconv:
            cand_in, cand_out = out_channels, in_channels
        self.candidates = nn.Parameter(
            torch.empty((self.num_bands, cand_in // groups, cand_out, kernel_size[0], kernel_size[1])),
            requires_grad=True,
        )

        nn.init.kaiming_normal_(self.candidates)
        nn.init.zeros_(self.candidates_bias)

        att, self.f_slices = self._make_equal_band_attention(self.num_bands, self.f_out)  # (K, F_out)
        self.register_buffer("att", att, persistent=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    @staticmethod
    def _make_equal_band_attention(num_bands: int, f_out: int):
        sizes = [f_out // num_bands] * num_bands
        rem = f_out - sum(sizes)
        for i in range(rem):
            sizes[i] += 1
        att = torch.zeros(num_bands, f_out, dtype=torch.float32)
        start = 0
        f_slices = []
        for b, sz in enumerate(sizes):
            end = start + sz
            if sz > 0:
                att[b, start:end] = 1.0
            f_slices.append((start, end-1))
            start = end
        return att, f_slices  # att: (K, F_out)

    def _conv_and_select(self, x):
        # weights: (K, I, O, kT, kF) -> (O*K, I, kT, kF)
        weight = rearrange(self.candidates, "k i o p q -> (o k) i p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out_all = F.conv2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups)
        B, OK, T, Fout = out_all.shape
        if Fout != self.f_out:
            raise RuntimeError(f"BConvBlock: expected output F dimension {self.f_out}, but got {Fout}. Check stride/padding or f_out.")
        O = self.out_channels
        K = self.num_bands
        out_all = out_all.view(B, O, K, T, Fout)
        att = self.att.to(out_all.dtype)  # (K, Fout)
        out = torch.einsum("b o k t f, k f -> b o t f", out_all, att)
        return out

    def _conv(self, x):
        out_slices = []
        for idx, f_slice in enumerate(self.f_slices):
            out_slice = partial_conv2d_w_functional(
                x,
                self.candidates[idx].transpose(0,1),
                self.candidates_bias[idx],
                self.kernel_size,
                self.stride,
                self.padding,
                (1, 1),
                self.groups,
                *f_slice
            )
            out_slices.append(out_slice)
        out = torch.cat(out_slices, dim=-1)
        return out

    def _deconv_and_select(self, x):
        # candidates: (K, I, O, kT, kF) but for deconv we had (K, O, I, ...) logically
        # Rearrange to match conv_transpose2d expected shape: (I, O*K, kT, kF) with groups
        weight = rearrange(self.candidates, "k o i p q -> i (o k) p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out_all = F.conv_transpose2d(x, weight, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups)
        B, OK, T, Fout = out_all.shape
        if Fout != self.f_out:
            raise RuntimeError(f"BConvBlock (deconv): expected output F dimension {self.f_out}, but got {Fout}. Check stride/padding or f_out.")
        O = self.out_channels
        K = self.num_bands
        out_all = out_all.view(B, O, K, T, Fout)
        att = self.att.to(out_all.dtype)  # (K, Fout)
        out = torch.einsum("b o k t f, k f -> b o t f", out_all, att)
        return out
    
    def _deconv(self, x):
        out_slices = []
        for idx, f_slice in enumerate(self.f_slices):
            out_slice = partial_conv_transpose2d_w_functional(
                x,
                self.candidates[idx].transpose(0,1),
                self.candidates_bias[idx],
                self.kernel_size,
                self.stride,
                self.padding,
                (0, 0),
                (1, 1),
                self.groups,
                *f_slice
            )
            out_slices.append(out_slice)
        out = torch.cat(out_slices, dim=-1)
        return out

    def forward(self, x):
        if calculate_macs_mode:
            if self.use_deconv:
                out = self._deconv(x)
            else:
                out = self._conv(x)
            return self.act(self.bn(out))
        if self.use_deconv:
            out = self._deconv_and_select(x)
        else:
            out = self._conv_and_select(x)
        return self.act(self.bn(out))
    

class BConv2dAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1), kernel_choices=1, use_deconv=False, groups=1, pad_left=0, f_band=2, f_out=33):
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

        self.f_band = f_band
        f_att, self.f_slices = self._make_equal_band_attention(self.f_band, f_out)  # (K, F_out)
        self.register_buffer("f_att", f_att, persistent=False)

    @staticmethod
    def _conv(x, candidates, bias, stride, padding, groups, dilation, f_band):
        out_slices = []
        for idx, f_slice in enumerate(f_band):
            out_slice = partial_conv2d_w_functional(
                x,
                candidates[idx],
                bias[idx],
                candidates.shape[-2:],
                stride,
                padding,
                dilation,
                groups,
                *f_slice
            )
            out_slices.append(out_slice)
        out = torch.cat(out_slices, dim=-1)
        return out
    
    @staticmethod
    def _deconv(x, candidates, bias, stride, padding, groups, dilation, f_band):
        out_slices = []
        for idx, f_slice in enumerate(f_band):
            out_slice = partial_conv_transpose2d_w_functional(
                x,
                candidates[idx],
                bias[idx],
                candidates.shape[-2:],
                stride,
                padding,
                (0, 0),
                dilation,
                groups,
                *f_slice
            )
            out_slices.append(out_slice)
        out = torch.cat(out_slices, dim=-1)
        return out

    @staticmethod
    def _make_equal_band_attention(num_bands: int, f_out: int):
        sizes = [f_out // num_bands] * num_bands
        rem = f_out - sum(sizes)
        for i in range(rem):
            sizes[i] += 1
        att = torch.zeros(num_bands, f_out, dtype=torch.float32)
        start = 0
        f_slices = []
        for b, sz in enumerate(sizes):
            end = start + sz
            if sz > 0:
                att[b, start:end] = 1.0
            f_slices.append((start, end-1))
            start = end
        return att, f_slices  # att: (K, F_out)
    
    def conv_and_sum(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        candidates = rearrange(self.candidates, "k i o p q -> (o k) i p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out = F.conv2d(x_pad, candidates, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        out = rearrange(out, "b (o k) t f -> b o k t f", o=self.out_channels, k=self.candidates_bias.shape[0])
        tf_attention = torch.einsum("b k t, l f -> b k l t f", attention, self.f_att)
        tf_attention = rearrange(tf_attention, "b k l t f -> b (k l) t f")
        out = torch.einsum("b o k t f, b k t f -> b o t f", out, tf_attention)
        return out
    
    def deconv_and_sum(self, out, attention):
        out = F.pad(out, [0, 0, self.pad_left, 0])  # pad left for causality
        # attention = F.pad(attention, [self.pad_left, 0], "replicate")  # pad left for causality
        candidates = rearrange(self.candidates, "k o i p q -> i (o k) p q")
        bias = rearrange(self.candidates_bias, "k o -> (o k)")
        out = F.conv_transpose2d(out, candidates, bias=bias, stride=self.stride, padding=self.padding, groups=self.groups, dilation=self.dilation)
        out = rearrange(out, "b (o k) t f -> b o k t f", o=self.out_channels, k=self.candidates_bias.shape[0])
        tf_attention = torch.einsum("b k t, l f -> b k l t f", attention, self.f_att)
        tf_attention = rearrange(tf_attention, "b k l t f -> b (k l) t f")
        out = torch.einsum("b o k t f, b k t f -> b o t f", out, tf_attention)
        return out
    
    def conv(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        unfolded = F.unfold(x_pad, (self.kernel_size[0], 1), dilation=(self.dilation[0], 1), padding=(self.padding[0], 0), stride=(self.stride[0], 1))
        unfolded2 = rearrange(unfolded, "b (c p) (t f) -> 1 (b t c) p f", c=x.shape[1], p=self.kernel_size[0], t=x.shape[2], f=x.shape[3])
        # x: (B, C, T, F)
        # unfolded: (B, C * K_T, T * F) -> (1, B*T*C, K_T, F)
        # attention: (B, K, T)
        candidates = rearrange(self.candidates, "(k l) i o p q -> l k i o p q", l=self.f_band)
        grouped_kernels = torch.einsum("lkiopq, bkt -> lbtiopq", candidates, attention)
        grouped_kernels = rearrange(grouped_kernels, "l b t i o p q -> l (b t o) i p q")
        candidates_bias = rearrange(self.candidates_bias, "(k l) o -> l k o", l=self.f_band)
        grouped_bias = torch.einsum("lko, bkt -> lbto", candidates_bias, attention)
        grouped_bias = rearrange(grouped_bias, "l b t o -> l (b t o)")
        out = BConv2dAttention._conv(unfolded2, grouped_kernels, bias=grouped_bias, stride=self.stride, padding=(0, self.padding[1]), groups=x.shape[0]*x.shape[2]*self.groups,
                                     dilation=(1, 1), f_band=self.f_slices)
        out = rearrange(out, "1 (b t o) 1 f -> b o t f", b=x.shape[0], t=x.shape[2], o=self.out_channels)
        return out
     
    def deconv(self, x, attention):
        x_pad = F.pad(x, [0, 0, self.pad_left, 0])  # pad left for causality
        unfolded = F.unfold(x_pad, (self.kernel_size[0] * self.dilation[0] - self.dilation[0] + 1, 1), dilation=(1, 1), padding=(0, 0), stride=(self.stride[0], 1))
        unfolded2 = rearrange(unfolded, "b (c p) (t f) -> 1 (b t c) p f", c=x.shape[1], p=self.kernel_size[0] * self.dilation[0] - self.dilation[0] + 1, t=x.shape[2], f=x.shape[3])
        # x: (B, C, T, F)
        # unfolded: (B, C * K_T, T * F) -> (1, B*T*C, K_T, F)
        # attention: (B, K, T)
        candidates = rearrange(self.candidates, "(k l) i o p q -> l k i o p q", l=self.f_band)
        grouped_kernels = torch.einsum("lkiopq, bkt -> lbtiopq", candidates, attention)
        grouped_kernels = rearrange(grouped_kernels, "l b t i o p q -> l (b t o) i p q")
        candidates_bias = rearrange(self.candidates_bias, "(k l) o -> l k o", l=self.f_band)
        grouped_bias = torch.einsum("lko, bkt -> lbto", candidates_bias, attention)
        grouped_bias = rearrange(grouped_bias, "l b t o -> l (b t o)")
        out = BConv2dAttention._deconv(unfolded2, grouped_kernels, bias=grouped_bias, stride=self.stride, padding=self.padding, groups=x.shape[0]*x.shape[2]*self.groups, dilation=self.dilation, 
                                       f_band=self.f_slices)
        out = rearrange(out, "1 (b t o) 1 f -> b o t f", b=x.shape[0], t=x.shape[2], o=self.out_channels)
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
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, kernel_choices=8, use_deconv=False, f_band=2, f_out=33):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        # conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
    
        self.sfe = SFE(kernel_size=3, stride=1)

        self.ln = nn.LayerNorm((in_channels//2*3, 33))
        
        self.point_conv1 = BConv2dAttention(in_channels//2*3, hidden_channels, (1, 1), use_deconv=use_deconv, kernel_choices=kernel_choices, f_band=f_band, f_out=f_out)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = BConv2dAttention(hidden_channels, hidden_channels, kernel_size,
                                            stride=stride, padding=padding,
                                            dilation=dilation, groups=hidden_channels, use_deconv=use_deconv, pad_left=self.pad_size, 
                                            kernel_choices=kernel_choices, f_band=f_band, f_out=f_out)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = BConv2dAttention(hidden_channels, in_channels//2, (1, 1), use_deconv=use_deconv, kernel_choices=kernel_choices, f_band=f_band, f_out=f_out)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)
        
        self.tra = TRA(in_channels//2, kernel_choices // f_band, 3)

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


class BGRNN(nn.Module):
    def __init__(self, freq_bins, chunks, *args, **kwargs):
        super().__init__()
        self.freq_bins = freq_bins
        self.chunk_num = chunks
        self.rnns = nn.ModuleList([
            GRNN(*args, **kwargs) for _ in range(chunks)
        ])

    def forward(self, x):
        """x: (B, C, T, F)"""
        chunks = list(torch.chunk(x, self.chunk_num, dim=-1))
        for i, chunk in enumerate(chunks):
            reshaped = chunk.permute(0, 3, 2, 1).reshape(-1, chunk.shape[2], chunk.shape[1])  # (B*F,T,C)
            reshaped = self.rnns[i](reshaped)[0]
            reshaped = reshaped.reshape(chunk.shape[0], chunk.shape[3], -1, chunk.shape[1])  # (B,F,T,C)
            reshaped = reshaped.permute(0, 3, 2, 1)  # (B,C,T,F)
            chunks[i] = reshaped
        x = torch.cat(chunks, dim=-1)
        return x, None


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

        self.inter_rnn = BGRNN(33, 2, input_size=input_size, hidden_size=hidden_size, bidirectional=False)
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
        x = intra_out.permute(0, 3, 1, 2)  # (B,C,T,F)
        inter_x = self.inter_rnn(x)[0]  # (B,C,T,F)
        inter_x = inter_x.permute(0,2,3,1)  # (B,T,F,C)
        inter_x = self.inter_fc(inter_x)      # (B,T,F,C)
        inter_x = self.inter_ln(inter_x) 
        inter_x = inter_x.permute(0,3,1,2)   # (B,C,T,F)
        inter_out = torch.add(x, inter_x)

        return inter_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            # F: 129 -> 65
            BConvBlock(3*3, CHANNELS, (1,5), stride=(1,2), padding=(0,2), use_deconv=False, is_last=False, bands=2, f_out=65),
            # F: 65 -> 33
            BConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=False, is_last=False, bands=2, f_out=33),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(1,1), use_deconv=False, f_band=2, kernel_choices=16, f_out=33),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(2,1), use_deconv=False, f_band=2, kernel_choices=16, f_out=33),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(0,1), dilation=(5,1), use_deconv=False, f_band=2, kernel_choices=16, f_out=33)
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
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(5*2,1), dilation=(5,1), use_deconv=True, f_band=2, kernel_choices=16, f_out=33),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(2*2,1), dilation=(2,1), use_deconv=True, f_band=2, kernel_choices=16, f_out=33),
            GTConvBlock(CHANNELS, CHANNELS, (3,3), stride=(1,1), padding=(1*2,1), dilation=(1,1), use_deconv=True, f_band=2, kernel_choices=16, f_out=33),
            # F: 33 -> 65
            BConvBlock(CHANNELS, CHANNELS, (1,5), stride=(1,2), padding=(0,2), groups=2, use_deconv=True, is_last=False, bands=2, f_out=65),
            # F: 65 -> 129
            BConvBlock(CHANNELS, 2, (1,5), stride=(1,2), padding=(0,2), use_deconv=True, is_last=True, bands=2, f_out=129)
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
    calculate_macs_mode = True
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
    except Exception as e:
        print("Skipping FLOPs calculation...")

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

    """"implementation alignment test"""
    calculate_macs_mode = False
    y11 = model(x1)[0]
    print(((y1 - y11)/(y1 + 1e-12)).abs().max())
