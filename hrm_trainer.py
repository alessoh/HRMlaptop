# hrm_trainer.py - Main HRM training script for Windows CPU
"""
Hierarchical Reasoning Model training optimized for Windows laptops without GPU.
Run with: python hrm_trainer.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import json
import os
import psutil
import platform
from pathlib import Path
from datetime import datetime
import logging
from torchvision import datasets, transforms

# Configure for CPU-only training on Windows
torch.set_num_threads(psutil.cpu_count(logical=False))  # Use physical cores
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Windows-specific fix

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WindowsHRM(nn.Module):
    """CPU-optimized HRM for Windows laptops"""
    
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Smaller architecture for CPU efficiency
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Lightweight reasoning layers
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights for CPU
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # Encode
        h = self.encoder(x)
        
        # Apply reasoning layers with residual connections
        for layer in self.reasoning_layers:
            h = h + layer(h)  # Residual connection
        
        # Output
        return self.output(h)

class CPUOptimizedTrainer:
    """Trainer optimized for Windows CPU training"""
    
    def __init__(self, model, save_dir="checkpoints"):
        self.model = model
        self.device = torch.device("cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Get system info
        self.system_info = self._get_system_info()
        logger.info(f"System: {self.system_info}")
        
        # Save system info
        with open('system_info.json', 'w') as f:
            json.dump(self.system_info, f, indent=2)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': [],
            'learning_rate': []
        }
        
    def _get_system_info(self):
        """Get Windows system information"""
        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_cores': psutil.cpu_count(logical=False),
            'cpu_threads': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__,
            'timestamp': datetime.now().isoformat()
        }
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001):
        """Train model on CPU"""
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Batch size: {train_loader.batch_size}")
        logger.info(f"Learning rate: {lr}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
                # Progress update
                if batch_idx % 100 == 0:
                    progress = 100. * batch_idx / len(train_loader)
                    logger.info(f'Epoch: {epoch+1} [{progress:.0f}%] '
                              f'Loss: {loss.item():.4f}')
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Update scheduler
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(avg_train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_time'].append(epoch_time)
            self.history['learning_rate'].append(current_lr)
            
            # Log results
            logger.info(f'Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s')
            logger.info(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            logger.info('-' * 50)
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, optimizer, val_acc)
        
        total_time = time.time() - start_time
        logger.info(f'Training completed in {total_time/60:.1f} minutes')
        logger.info(f'Average time per epoch: {total_time/epochs:.1f} seconds')
        
        # Save final model and results
        self.save_final_model()
        self.save_training_results()
        
        return self.history
    
    def evaluate(self, data_loader, criterion):
        """Evaluate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                output = self.model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch, optimizer, val_acc):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
            'system_info': self.system_info
        }
        
        path = self.save_dir / f'checkpoint_epoch_{epoch+1}.pt'
        torch.save(checkpoint, path)
        logger.info(f'Checkpoint saved: {path}')
    
    def save_final_model(self):
        """Save final model"""
        model_path = self.save_dir / 'hrm_final.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim
            },
            'history': self.history,
            'system_info': self.system_info,
            'timestamp': datetime.now().isoformat()
        }, model_path)
        logger.info(f'Final model saved: {model_path}')
    
    def save_training_results(self):
        """Save training results to JSON"""
        with open('training_results.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info('Training results saved to training_results.json')

def get_data_loaders(batch_size=32):
    """Get MNIST data loaders optimized for CPU"""
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    logger.info("Loading MNIST dataset...")
    
    # Download MNIST
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # CPU-optimized data loaders (num_workers=0 for Windows)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Important: use 0 for Windows
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size*2,  # Larger batch for validation
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader

def print_banner():
    """Print welcome banner"""
    print("="*60)
    print(" " * 10 + "HRM Training on Windows CPU")
    print(" " * 15 + "Chapter 12 Implementation")
    print("="*60)
    print()
    print("System Information:")
    print(f"  CPU: {psutil.cpu_count(logical=False)} cores, {psutil.cpu_count(logical=True)} threads")
    print(f"  RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  Python: {platform.python_version()}")
    print(f"  PyTorch: {torch.__version__}")
    print("="*60)
    print()

def main():
    """Main training function"""
    # Print banner
    print_banner()
    
    # Load environment variables if .env exists
    if os.path.exists('.env'):
        logger.info("Loading settings from .env file")
        with open('.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    
    # Get settings from environment or use defaults
    batch_size = int(os.environ.get('BATCH_SIZE', 32))
    learning_rate = float(os.environ.get('LEARNING_RATE', 0.001))
    num_epochs = int(os.environ.get('NUM_EPOCHS', 10))
    hidden_dim = int(os.environ.get('HIDDEN_DIM', 64))
    num_layers = int(os.environ.get('NUM_LAYERS', 2))
    
    logger.info("Training Configuration:")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Epochs: {num_epochs}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Number of layers: {num_layers}")
    
    # Create model
    logger.info("\nCreating HRM model...")
    model = WindowsHRM(
        input_dim=784,
        hidden_dim=hidden_dim,
        output_dim=10,
        num_layers=num_layers
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} (all trainable)")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size=batch_size)
    
    # Create trainer
    trainer = CPUOptimizedTrainer(model)
    
    # Train model
    logger.info("\nStarting training...")
    logger.info("This will take approximately 45-90 minutes on a typical Windows laptop")
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=num_epochs,
        lr=learning_rate
    )
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Validation Accuracy: {history['val_acc'][-1]:.2f}%")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Total Training Time: {sum(history['epoch_time'])/60:.1f} minutes")
    print(f"Model saved to: checkpoints/hrm_final.pt")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Test your model: python test_model.py")
    print("2. Visualize results: python visualize.py")
    print("3. Monitor system usage: python monitor.py")
    
    return model, history

if __name__ == "__main__":
    try:
        model, history = main()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise