#!/usr/bin/env python
"""
Installation script for HRM Laptop Training on Windows
Run with: python install.py
"""

import subprocess
import sys
import os
from pathlib import Path

def install_packages():
    """Install required packages for Windows CPU training"""
    
    print("="*60)
    print("HRM Laptop Training - Installation Script")
    print("="*60)
    print()
    
    # First, upgrade pip
    print("Upgrading pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Install PyTorch CPU version
    print("\nInstalling PyTorch (CPU version)...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "torch", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # Install other requirements
    print("\nInstalling other dependencies...")
    packages = [
        "numpy==1.24.3",
        "matplotlib==3.7.1",
        "psutil==5.9.5",
        "tqdm==4.65.0",
        "pandas==2.0.2",
        "scikit-learn==1.2.2",
        "Pillow==9.5.0",
        "seaborn==0.12.2",
        "plotly==5.14.1"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Create necessary directories
    print("\nCreating project directories...")
    dirs = ["data", "checkpoints", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"Created {dir_name}/ directory")
    
    # Test installation
    print("\nTesting installation...")
    try:
        import torch
        import torchvision
        import numpy
        import matplotlib
        
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ NumPy version: {numpy.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()} (Should be False for CPU)")
        print(f"✓ CPU threads: {torch.get_num_threads()}")
        
        print("\n✅ Installation successful!")
        
    except ImportError as e:
        print(f"\n❌ Installation test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nYou can now run training with:")
    print("  python hrm_trainer.py")
    print("\nOr for simple training:")
    print("  python train_simple.py")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = install_packages()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\nError during installation: {e}")
        print("Please try manual installation:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("  pip install -r requirements.txt")
        sys.exit(1)