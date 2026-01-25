import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Global flag to control re-parameterization mode (originally calculate_macs_mode)
# In the original code, this flag swithes between training mode (multi-branch) and fused inference mode.
CALCULATE_MACS_MODE = False

def set_calculate_macs_mode(mode: bool):
    global CALCULATE_MACS_MODE
    CALCULATE_MACS_MODE = mode

class RepConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        assert kernel_size == (3, 3)
        assert in_channels == out_channels
        assert stride == (1, 1) or stride == 1
        assert dilation[1] == 1
        assert padding[1] == 1 and padding[0] == (2*dilation[0] if use_deconv else 0)
        assert groups == 1
        self.use_deconv = use_deconv

        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv_functional = F.conv_transpose2d if use_deconv else F.conv2d
        self.conv = conv_module(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.point_conv = conv_module(in_channels, out_channels, (1,1), stride=1, padding=0, groups=groups)
        self.point_bn = nn.BatchNorm2d(out_channels)
        self.identity_bn = nn.BatchNorm2d(in_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()
        self.infer_initailized = False
        self.infer_conv_kernel = None
        self.infer_offset = None

    def forward(self, x):
        x_pad = F.pad(x, [0, 0, self.pad_size, 0])
        
        # Use module-level flag, or can be overridden by instance attribute if we added one (but keeping signature close to original)
        if not CALCULATE_MACS_MODE:
            out1 = self.bn(self.conv(x_pad))
            out2 = self.point_bn(self.point_conv(x))
            out3 = self.identity_bn(x)
            return self.act(out1 + out2 + out3)
            
        if not self.infer_initailized:
            if self.use_deconv:
                bn_reshape = lambda x: x.reshape(1, -1, 1, 1)
                identity_index = 0
            else:
                bn_reshape = lambda x: x.reshape(-1, 1, 1, 1)
                identity_index = 2
            self.infer_conv_kernel = self.conv.weight * bn_reshape(self.bn.weight) / bn_reshape((self.bn.running_var + self.bn.eps).sqrt())
            self.infer_conv_kernel[:, :, identity_index, 1] += (self.point_conv.weight * bn_reshape(self.point_bn.weight) / bn_reshape((self.point_bn.running_var + self.point_bn.eps).sqrt())).squeeze(-1).squeeze(-1)
            self.infer_conv_kernel[:, :, identity_index, 1] += torch.diag(self.identity_bn.weight / (self.identity_bn.running_var + self.identity_bn.eps).sqrt())
            self.infer_offset = (self.conv.bias - self.bn.running_mean) * self.bn.weight / (self.bn.running_var + self.bn.eps).sqrt() + self.bn.bias
            self.infer_offset += (self.point_conv.bias - self.point_bn.running_mean) * self.point_bn.weight / (self.point_bn.running_var + self.point_bn.eps).sqrt() + self.point_bn.bias
            self.infer_offset += ( - self.identity_bn.running_mean) * self.identity_bn.weight / (self.identity_bn.running_var + self.identity_bn.eps).sqrt() + self.identity_bn.bias
            self.infer_initailized = True
        
        out = self.conv_functional(x_pad, self.infer_conv_kernel, bias=self.infer_offset, stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        return self.act(out)

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
        self.candidates = nn.Parameter(torch.empty((kernel_choices, in_channels//groups, out_channels, *kernel_size), requires_grad=True))

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
