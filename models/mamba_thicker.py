import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.profiler import record_function
import math

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

CALCULATE_MACS_MODE = False

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
        if CALCULATE_MACS_MODE:
            scan_macs = int(seqlen * self.d_inner * self.d_state * 3) # (1, K) @ (K, 1) = K MACs
            if scan_macs > 0:
                dummy_v = torch.zeros(1, scan_macs, device=device)
                dummy_w = torch.zeros(scan_macs, 1, device=device)
                torch.mm(dummy_v, dummy_w)
        
        out = self.out_proj(y)
        return out

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
        # x_low: 0~65, x_high: 65~end
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        # Output dim = 65 + 64 = 129
        return torch.cat([x_low, x_high], dim=-1)
    
    def bs(self, x_erb):
        """x: (B,C,T,F_erb=129)"""
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
        """
        Input: (B, 3, T, 129)
        Output: (B, 9, T, 129)
        """
        xs = self.unfold(x).reshape(x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs

class Mask(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:,0] * mask[:,0] - spec[:,1] * mask[:,1]
        s_imag = spec[:,1] * mask[:,0] + spec[:,0] * mask[:,1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s
    

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
        n_layers=4
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.win_len = win_len
        
        self.erb = ERB(65, 64) 
        self.sfe = SFE(3, 1)   
        
        # (B, 9, T, 129) -> (B, T, d_model)
        self.input_conv = nn.Conv2d(9, 5, kernel_size=1) 
        self.input_fc = nn.Linear(129 * 5, d_model) 
        self.input_act = nn.PReLU()

        # Mamba
        self.backbone = nn.Sequential(*[
            PureMambaBlock(d_model) for _ in range(n_layers)
        ])

        # (B, T, d_model) -> (B, 2, T, 129)
        self.output_fc = nn.Linear(d_model, 129 * 2)
        
        self.output_act = nn.Tanh() 

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
        spec = torch.view_as_real(spec) # (B, F=257, T, 2)

        spec_real = spec[..., 0].permute(0,2,1)
        spec_imag = spec[..., 1].permute(0,2,1)
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=1)  # (B,3,T,257)
        
        spec = spec.permute(0,3,2,1)  # (B,2,T,F=257)

        feat = self.erb.bm(feat)  # (B,3,T,129)
        feat = self.sfe(feat)     # (B,9,T,129)

        # (B, 9, T, 129) -> (B, 3, T, 129)
        x_in = self.input_conv(feat) 
        # (B, 3, T, 129) -> (B, T, 387)
        x_in = x_in.permute(0, 2, 1, 3).reshape(x_in.shape[0], x_in.shape[2], -1)
        # (B, T, 129) -> (B, T, d_model)
        x_in = self.input_act(self.input_fc(x_in))

        # (B, T, d_model)
        x_mem = self.backbone(x_in)

        # (B, T, d_model) -> (B, T, 258)
        x_out = self.output_act(self.output_fc(x_mem))
        
        # Reshape to (B, 2, T, 129)
        m_feat = x_out.view(x_out.shape[0], x_out.shape[1], 2, 129).permute(0, 2, 1, 3)

        # ERB Decompression (129 -> 257)
        m = self.erb.bs(m_feat) 

        spec_enh = self.mask(m, spec) # (B,2,T,F=257)
        spec_enh = spec_enh.permute(0,3,2,1)  # (B,F,T,2)
        
        spec_enh = torch.complex(spec_enh[...,0], spec_enh[...,1])
        output = torch.istft(spec_enh, **stft_kwargs)
        
        if output.shape[1] < n_samples:
            output = torch.nn.functional.pad(output, (0, n_samples-output.shape[1]))
        else:
            output = output[:, :n_samples]
        
        return output


if __name__ == "__main__":
    
    model = MambaEnhancer().eval().cuda()

    CALCULATE_MACS_MODE = True
    print(f"\n--- Model Statistics (Simulated Mode: {CALCULATE_MACS_MODE}) ---")
    
    try:
        from ptflops import get_model_complexity_info
        flops, params = get_model_complexity_info(model, (16000,), as_strings=True,
                                                print_per_layer_stat=True, verbose=False, backend="aten")
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Total Params: {num_params} (Limit: ~48200)")
        print(f"KMac/s: {flops}") 
    except Exception as e:
        print(f"Flops counting failed: {e}")

    print("\n--- Equivalence Test (Simulated vs Real) ---")
    x_test = torch.randn(1, 16000).cuda()
    
    CALCULATE_MACS_MODE = True
    with torch.no_grad():
        y_sim = model(x_test)
        
    CALCULATE_MACS_MODE = False
    with torch.no_grad():
        y_real = model(x_test)
        
    diff = (y_real - y_sim).abs().max()
    print(f"Max Difference between Real Mamba and Simulated Mamba: {diff.item():.2e}")
    
    if diff < 1e-4:
        print(">>> Equivalence Test PASSED.")
    else:
        print(">>> Equivalence Test FAILED.")

    print("\n--- Causality Check (Real Model) ---")
    CALCULATE_MACS_MODE = False
    x = torch.randn(1, 16000).cuda()
    split = 8000
    
    x1 = x.clone()
    x2 = x.clone()
    x2[:, split:] = torch.randn(1, 16000 - split).cuda()
    
    with torch.no_grad():
        y1 = model(x1)
        y2 = model(x2)
    
    safe_point = split - 512
    diff = (y1[:, :safe_point] - y2[:, :safe_point]).abs().max()
    print(f"Max Diff in Past (t < {safe_point}): {diff.item():.2e}")

    if diff < 1e-4:
        print(">>> Causality PASSED.")
    else:
        print(">>> Causality FAILED.")

    y = model(x)
    print(f"\nIn: {x.shape}, Out: {y.shape}")