# HRMlaptop
A laptop version of Hierarchical Reasoning Model
# HRM Training on Windows Laptop (CPU-Only)

Train a Hierarchical Reasoning Model on your Windows laptop using only CPU - no GPU required!

## Quick Start

### Prerequisites
- **Python 3.12** (via Anaconda or python.org)
- **Windows 10/11** (64-bit)
- **8GB RAM minimum** (16GB recommended)

### Installation

1. **Clone or download this project**
```bash
git clone https://github.com/alessoh/HRMlaptop
cd HRMlaptop
```

2. **Create conda environment (if using Anaconda)**
```bash
conda create -n HRMlaptop python=3.12
conda activate HRMlaptop
```

3. **Install PyTorch CPU version**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

4. **Install remaining dependencies**
```bash
python quick_install.py
```

Or manually:
```bash
pip install matplotlib psutil tqdm pandas scikit-learn seaborn plotly
```

### Running the Training

**Simplest option - for beginners:**
```bash
python train_simple.py
```

**Full training with all features:**
```bash
python hrm_trainer.py
```

**Monitor system resources (in separate terminal):**
```bash
python monitor.py
```

## What's Included

### Core Training Scripts
- `hrm_trainer.py` - Main training script with full features
- `train_simple.py` - Simplified training for beginners (recommended for first run)
- `test_model.py` - Test your trained model
- `monitor.py` - Monitor CPU and memory usage during training
- `visualize.py` - Create graphs and visualizations of results

### Setup Files
- `quick_install.py` - Install remaining dependencies after PyTorch
- `requirements.txt` - Package list (reference only)
- `.env` - Configuration settings
- `.gitignore` - Git ignore patterns

### Installation Helpers
- `install.py` - Full installation script
- `setup_windows.bat` - Windows batch installer

## Expected Performance

### Training Time (10 epochs on MNIST)
| CPU Type | Training Time | Accuracy |
|----------|--------------|----------|
| Intel i5 (8th gen) | 90-120 min | ~97.5% |
| Intel i7 (10th gen) | 45-60 min | ~97.8% |
| AMD Ryzen 7 | 40-55 min | ~97.8% |

### Resource Usage
- **RAM**: 1.5-2.5 GB
- **CPU**: 60-80% utilization
- **Disk**: ~500 MB (including dataset)
- **Model Size**: ~350 KB

## Python 3.12 Compatibility Notes

This project is optimized for Python 3.12 with:
- PyTorch 2.8.0+cpu (latest CPU version)
- NumPy 2.1.2 (included with PyTorch)
- All dependencies tested with Python 3.12

## Training Output Example

```
============================================================
HRM Training on Windows CPU
============================================================

System Information:
  CPU: 6 cores, 12 threads
  RAM: 16.0 GB
  Python: 3.12.0
  PyTorch: 2.8.0+cpu
============================================================

Starting training for 10 epochs
Epoch 1/10 [0%] Loss: 0.4521
Epoch 1/10 [20%] Loss: 0.3421
Epoch 1/10 [40%] Loss: 0.2341
...
Epoch 10/10 completed in 324.5s
Train Loss: 0.0234, Train Acc: 98.45%
Val Loss: 0.0412, Val Acc: 97.82%

TRAINING COMPLETE!
Final Validation Accuracy: 97.82%
Total Training Time: 54.3 minutes
```

## Troubleshooting

### Issue: "No module named torch"
**Solution**: Install PyTorch CPU version:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Issue: NumPy compatibility error
**Solution**: Use the NumPy that comes with PyTorch (2.1.2). Don't install numpy==1.24.3

### Issue: Training is very slow
**Solutions**:
- Close Chrome and other heavy applications
- Set Windows to High Performance mode
- Ensure good laptop cooling
- Reduce batch size in `.env` file

### Issue: ImportError with Python 3.12
**Solution**: Use the provided `quick_install.py` script which installs compatible versions

### Issue: High CPU temperature
**Solutions**:
- Elevate laptop for better airflow
- Use a cooling pad
- Reduce batch size to 16 or 8
- Take breaks between training epochs

## Configuration Options

Edit `.env` file to customize:

```bash
# Training parameters
BATCH_SIZE=32          # Reduce to 16 if low on RAM
LEARNING_RATE=0.001    
NUM_EPOCHS=10          # Increase for better accuracy
HIDDEN_DIM=64          # Model size (32 for faster, 128 for better)
NUM_LAYERS=2           # Depth of reasoning layers

# CPU optimization
NUM_THREADS=4          # Set to your physical core count
```

## Project Structure

```
HRMlaptop/
├── hrm_trainer.py       # Main training script
├── train_simple.py      # Beginner-friendly trainer
├── test_model.py        # Model testing
├── monitor.py           # System monitoring
├── visualize.py         # Results visualization
├── quick_install.py     # Dependency installer
├── README.md           # This file
├── .env                # Configuration
├── requirements.txt    # Package reference
├── data/              # MNIST dataset (auto-created)
├── checkpoints/       # Saved models (auto-created)
└── logs/              # Training logs (auto-created)
```

## Tips for Faster Training

1. **Windows Performance Mode**
   - Settings → System → Power & battery → Best performance

2. **Close Background Apps**
   - Especially browsers (Chrome uses lots of RAM)
   - Disable Windows updates during training

3. **Optimal Settings for Speed**
   ```bash
   BATCH_SIZE=64      # If you have 16GB+ RAM
   HIDDEN_DIM=32      # Smaller model
   NUM_LAYERS=1       # Fewer layers
   ```

4. **Optimal Settings for Accuracy**
   ```bash
   BATCH_SIZE=32      # Default
   HIDDEN_DIM=64      # Default
   NUM_LAYERS=2       # Default
   NUM_EPOCHS=20      # More training
   ```

## After Training

1. **Test your model**
   ```bash
   python test_model.py
   ```

2. **Visualize results**
   ```bash
   python visualize.py
   ```

3. **View training metrics**
   - Check `training_results.json` for detailed metrics
   - View `training.log` for full training history
   - Open generated `.png` files for graphs

## Understanding the Model

The HRM (Hierarchical Reasoning Model) is optimized for CPU training:
- **Small architecture**: 64 hidden units (vs 128+ for GPU)
- **Efficient layers**: 2 reasoning layers with residual connections
- **CPU optimizations**: Batch normalization, dropout for stability
- **Total parameters**: ~85K (very lightweight)

## Citation

If you use this code for learning or research:
```
H. Peter Alesso (2025). Hierarchical Reasoning Model: HRM on Your Laptop. 
Chapter 12: Training on Your Windows Laptop.
```

## Support

- **Issues**: Create an issue on GitHub
- **Logs**: Check `training.log` for detailed information
- **System Info**: See `system_info.json` for your configuration

## License

MIT License - Free for educational and research use

---

**Ready to start?** Run `python train_simple.py` and watch your laptop learn to recognize handwritten digits!