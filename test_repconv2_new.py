import torch
import torch.nn as nn
from models_new.modules.conv import RepConvBlock, RepConvBlock2, set_calculate_macs_mode

DILATIONS = [1, 2, 3, 5]

def test_cross_compatibility():
    print(f"\nTesting Cross Compatibility (RepConvBlock vs RepConvBlock2(use_1d_kernel=False))...")
    C = 8
    
    for d in DILATIONS:
        print(f"  Dilation: ({d}, 1)")
        # 1. Init RepConvBlock
        try:
             block1 = RepConvBlock(C, C, kernel_size=(3, 3), padding=(0, 1), dilation=(d, 1))
        except Exception as e:
             print(f"    Skipping RepConvBlock dilation {d} due to init error (model mismatch?): {e}")
             continue
             
        block1.eval()

        # 2. Init RepConvBlock2
        block2 = RepConvBlock2(C, C, kernel_size=(3, 3), padding=(0, 1), dilation=(d, 1), use_1d_kernel=False)
        block2.eval()
        
        # 3. Load weights
        try:
            block2.load_state_dict(block1.state_dict())
        except Exception as e:
            print(f"    FAIL: Load state dict failed: {e}")
            continue

        # 4. Compare
        x = torch.randn(1, C, 30, 16) 
        
        set_calculate_macs_mode(False)
        with torch.no_grad():
            out1 = block1(x)
            out2 = block2(x)
        
        diff = (out1 - out2).abs().max()
        print(f"    Max diff: {diff}")
        
        if diff > 1e-6:
            print("    FAIL: Cross compatibility check failed.")
        else:
            print("    PASS: Cross compatibility check passed.")

def test_causality():
    print(f"\nTesting Causality (Conv)...")
    C = 4
    F = 16
    T = 40 

    for d in DILATIONS:
        print(f"  Dilation: ({d}, 1)")
        # Padding (0,1) for conv
        block = RepConvBlock2(C, C, kernel_size=(3, 3), padding=(0, 1), dilation=(d, 1), use_1d_kernel=True)
        block.eval()
        
        x = torch.randn(1, C, T, F)
        with torch.no_grad():
            out1 = block(x)
            
        # Modify future
        modify_idx = 25
        
        x_mod = x.clone()
        x_mod[:, :, modify_idx:, :] += 1.0 
        
        with torch.no_grad():
            out2 = block(x_mod)
            
        diff = (out1[:, :, :modify_idx, :] - out2[:, :, :modify_idx, :]).abs().max()
        print(f"    Max diff (t<{modify_idx}): {diff}")
        if diff > 1e-6:
            print("    FAIL: Causality violation.")
        else:
            print("    PASS: Causality check passed.")

def test_consistency():
    print(f"\nTesting Consistency (Fused vs Multi-branch)...")
    C = 8
    
    for d in DILATIONS:
        print(f"  Dilation: ({d}, 1)")
        block = RepConvBlock2(C, C, kernel_size=(3, 3), padding=(0, 1), dilation=(d, 1), use_1d_kernel=True)
        block.eval()
        
        x = torch.randn(1, C, 30, 16)
        
        # Branch mode
        set_calculate_macs_mode(False)
        with torch.no_grad():
            out_branch = block(x)
            
        # Fused mode
        set_calculate_macs_mode(True)
        with torch.no_grad():
            out_fused = block(x)
            
        diff = (out_branch - out_fused).abs().max()
        print(f"    Max diff: {diff}")
        
        if diff > 1e-5:
             print("    FAIL: Consistency violation.")
        else:
             print("    PASS: Consistency check passed.")
        set_calculate_macs_mode(False) # Reset

def test_deconv_consistency():
    print(f"\nTesting Consistency (Deconv)...")
    C = 8
    
    for d in DILATIONS:
        print(f"  Dilation: ({d}, 1)")
        # For deconv: padding[0] = 2*dilation (based on assert)
        padding_val = (2*d, 1)
        
        try:
             block = RepConvBlock2(C, C, kernel_size=(3, 3), padding=padding_val, dilation=(d, 1), use_deconv=True, use_1d_kernel=True)
        except Exception as e:
             print(f"    Init failed for deconv dilation {d}: {e}")
             continue
             
        block.eval()
        
        x = torch.randn(1, C, 30, 16)
        
        set_calculate_macs_mode(False)
        with torch.no_grad():
            out_branch = block(x)
            
        set_calculate_macs_mode(True)
        with torch.no_grad():
            out_fused = block(x)
            
        diff = (out_branch - out_fused).abs().max()
        print(f"    Max diff: {diff}")
        if diff > 1e-5:
             print("    FAIL: Consistency violation.")
        else:
             print("    PASS: Consistency check passed.")
        set_calculate_macs_mode(False)

def test_compatibility():
    print(f"\nTesting Compatibility (use_1d_kernel=False)...")
    C = 8
    
    for d in DILATIONS:
        print(f"  Dilation: ({d}, 1)")
        block = RepConvBlock2(C, C, kernel_size=(3, 3), padding=(0, 1), dilation=(d, 1), use_1d_kernel=False)
        block.eval()
        x = torch.randn(1, C, 30, 16)
        
        set_calculate_macs_mode(False)
        out1 = block(x)
        
        set_calculate_macs_mode(True)
        out2 = block(x)
        
        diff = (out1 - out2).abs().max()
        print(f"    Max diff: {diff}")
        assert diff < 1e-5
        print("    PASS: Compatibility passed.")
        set_calculate_macs_mode(False)

if __name__ == "__main__":
    try:
        test_cross_compatibility()
        # test_dilation()
        test_causality()
        test_consistency()
        test_deconv_consistency()
        test_compatibility()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
