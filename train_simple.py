# train_simple.py - Simplified training script for beginners
"""
Simple HRM training script for Windows CPU.
Just run: python train_simple.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import os

# Set CPU threads for Windows
torch.set_num_threads(4)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("=" * 50)
print("HRM Training - Simple Version")
print("=" * 50)

# Simple HRM Model
class SimpleHRM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.layer3(x)
        return x

# Load MNIST data
print("\nDownloading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Create model
print("Creating model...")
model = SimpleHRM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
print("\nStarting training (10 epochs)...")
print("This will take about 45-60 minutes on a typical laptop")
print("-" * 50)

start_time = time.time()

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Progress indicator
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch+1}: [{batch_idx*32}/{len(train_dataset)}] "
                  f"Loss: {loss.item():.4f}")
    
    # Epoch summary
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Testing
print("\nTesting model...")
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

test_accuracy = 100. * correct / total

# Results
elapsed_time = (time.time() - start_time) / 60
print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Training Time: {elapsed_time:.1f} minutes")
print("=" * 50)

# Save model
torch.save(model.state_dict(), 'simple_hrm_model.pt')
print("\nModel saved as 'simple_hrm_model.pt'")
print("You can now run 'python test_model.py' to test your model!")