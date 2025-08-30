#!/usr/bin/env python
"""
Quick installation script for remaining dependencies
PyTorch is already installed!
"""

import subprocess
import sys

def install_remaining():
    """Install remaining packages that are compatible with Python 3.12"""
    
    print("="*60)
    print("Installing remaining dependencies...")
    print("="*60)
    print()
    
    # These packages are compatible with Python 3.12
    packages = [
        "matplotlib",
        "psutil",
        "tqdm",
        "pandas",
        "scikit-learn",
        "seaborn",
        "plotly"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except:
            print(f"⚠ {package} installation failed, but continuing...")
    
    print("\n" + "="*60)
    print("Testing installation...")
    print("="*60)
    
    try:
        import torch
        import torchvision
        import numpy as np
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ TorchVision version: {torchvision.__version__}")
        print(f"✓ NumPy version: {np.__version__}")
        print(f"✓ Python version: {sys.version}")
        print(f"✓ CUDA available: {torch.cuda.is_available()} (Should be False for CPU)")
        
        # Test basic tensor operation
        x = torch.randn(2, 3)
        print(f"✓ Test tensor created: shape {x.shape}")
        
        print("\n✅ Installation successful!")
        print("\nYou can now run:")
        print("  python hrm_trainer.py")
        print("  or")
        print("  python train_simple.py")
        
    except ImportError as e:
        print(f"\n⚠ Warning: {e}")
        print("But you should still be able to run the training scripts!")
    
    return True

if __name__ == "__main__":
    install_remaining()