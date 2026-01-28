import torch
import torch.nn as nn
import sys
import os

# Add the current directory to sys.path to ensure we can import the module correctly
sys.path.append(os.getcwd())

try:
    from models_new.modules.conv import RepPointConv, set_calculate_macs_mode
except ImportError as e:
    print(f"Import failed: {e}")
    print("Please ensure you are running this script from the project root directory.")
    sys.exit(1)

def test_reppointconv_consistency():
    print("Testing RepPointConv Consistency...")
    
    # Configuration
    in_channels = 16
    out_channels = 32
    stride = 1 # Test stride 1 first
    
    # 1. Initialize the module (Default: No Activation)
    print(f"Initializing RepPointConv(in={in_channels}, out={out_channels}, stride={stride}, use_act=False)")
    model = RepPointConv(in_channels, out_channels, stride=stride, use_act=False)
    
    # IMPORTANT: Use eval mode so that BatchNorms uses running statistics
    model.eval() 
    
    # Create input
    x = torch.randn(2, in_channels, 32, 32)
    
    # 2. Forward in training mode (multi-branch)
    set_calculate_macs_mode(False)
    model.eval() 
    
    with torch.no_grad():
        out_train = model(x)
    
    print("Forward pass (calculating macs mode = False) done.")
        
    # 3. Forward in inference mode (fused)
    set_calculate_macs_mode(True)
    
    with torch.no_grad():
        out_infer = model(x)
        
    print("Forward pass (calculating macs mode = True, fused) done.")
        
    # 4. Compare
    diff = (out_train - out_infer).abs().max()
    print(f"Max difference: {diff.item()}")
    
    if diff < 1e-4:
        print("✅ Test Passed: Consistency verified for stride=1 (No Act).")
    else:
        print("❌ Test Failed: Outputs are not consistent.")
        print("out_train sample:", out_train[0, 0, :2, :2])
        print("out_infer sample:", out_infer[0, 0, :2, :2])


    # Test Case 2: Stride > 1 and Identity (in=out) WITH ACTIVATION
    print("\n-------------------------------------------")
    print("Testing with stride=2 and identity (in=out) WITH ACTIVATION...")
    in_channels = 32
    out_channels = 32
    stride = 2
    
    model = RepPointConv(in_channels, out_channels, stride=stride, use_act=True)
    model.eval()
    
    x = torch.randn(2, in_channels, 32, 32)
    
    set_calculate_macs_mode(False)
    with torch.no_grad():
        out_train = model(x)
        
    set_calculate_macs_mode(True)
    with torch.no_grad():
        out_infer = model(x)
        
    diff = (out_train - out_infer).abs().max()
    print(f"Max difference: {diff.item()}")
    
    if diff < 1e-4:
        print("✅ Test Passed: Consistency verified for stride=2 (With Act).")
    else:
        print("❌ Test Failed: Outputs are not consistent.")

if __name__ == "__main__":
    test_reppointconv_consistency()
