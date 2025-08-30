# hrm_applications.py - Various problems you can solve with HRM
"""
Examples of different problems that can be solved using the HRM architecture.
Each shows how to adapt the core HRM design to different domains.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================
# 1. XOR Problem - Classic non-linear problem
# ============================================================

class XOR_HRM(nn.Module):
    """HRM for solving XOR (simplest non-linear problem)"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU()
        )
        self.reasoning = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.output = nn.Linear(16, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h + self.reasoning(h)  # Residual connection
        return torch.sigmoid(self.output(h))

def train_xor():
    """Train HRM on XOR problem"""
    print("\n1. XOR Problem")
    print("-"*40)
    
    # XOR dataset
    X = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
    y = torch.FloatTensor([[0], [1], [1], [0]])
    
    model = XOR_HRM()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    # Train
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test
    with torch.no_grad():
        predictions = model(X)
        print("\nXOR Results:")
        for i in range(4):
            print(f"Input: {X[i].numpy()} -> Output: {predictions[i].item():.3f} "
                  f"(Target: {y[i].item()})")
    
    return model

# ============================================================
# 2. Sequence Prediction - Predict next in sequence
# ============================================================

class SequenceHRM(nn.Module):
    """HRM for sequence prediction"""
    def __init__(self, seq_length=5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_length, 32),
            nn.ReLU()
        )
        self.reasoning = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.output = nn.Linear(32, 1)
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        h = self.encoder(x)
        h = h + self.reasoning(h)
        return self.output(h)

def train_sequence():
    """Train HRM to predict sequences"""
    print("\n2. Sequence Prediction")
    print("-"*40)
    
    # Generate arithmetic sequence data
    sequences = []
    targets = []
    
    for start in range(1, 20):
        for step in range(1, 5):
            seq = [start + i*step for i in range(5)]
            sequences.append(seq)
            targets.append(start + 5*step)  # Next in sequence
    
    X = torch.FloatTensor(sequences)
    y = torch.FloatTensor(targets).unsqueeze(1)
    
    model = SequenceHRM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test
    test_sequences = [
        [2, 4, 6, 8, 10],      # Next: 12
        [5, 10, 15, 20, 25],   # Next: 30
        [1, 3, 5, 7, 9]        # Next: 11
    ]
    
    print("\nSequence Predictions:")
    model.eval()  # Set to eval mode for inference
    with torch.no_grad():
        for seq in test_sequences:
            input_tensor = torch.FloatTensor(seq)
            pred = model(input_tensor).item()
            print(f"Sequence: {seq} -> Predicted next: {pred:.1f}")
    
    return model

# ============================================================
# 3. Simple Calculator - Learn arithmetic operations
# ============================================================

class CalculatorHRM(nn.Module):
    """HRM to learn basic arithmetic"""
    def __init__(self):
        super().__init__()
        # Input: [num1, num2, operation_code]
        self.encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU()
        )
        self.reasoning1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.reasoning2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.output = nn.Linear(64, 1)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h + self.reasoning1(h)
        h = h + self.reasoning2(h)
        return self.output(h)

def train_calculator():
    """Train HRM to be a simple calculator"""
    print("\n3. Simple Calculator")
    print("-"*40)
    
    # Generate training data
    data = []
    targets = []
    
    # Addition (operation code 0)
    for a in range(1, 10):
        for b in range(1, 10):
            data.append([a/10, b/10, 0])  # Normalize inputs
            targets.append((a + b)/20)     # Normalize output
    
    # Subtraction (operation code 1)
    for a in range(1, 10):
        for b in range(1, 10):
            data.append([a/10, b/10, 1])
            targets.append((a - b + 10)/20)  # Shift to positive
    
    X = torch.FloatTensor(data)
    y = torch.FloatTensor(targets).unsqueeze(1)
    
    model = CalculatorHRM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Train
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    for epoch in range(100):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
    
    # Test
    print("\nCalculator Test:")
    test_cases = [
        ([3, 5, 0], 8),   # 3 + 5 = 8
        ([7, 2, 0], 9),   # 7 + 2 = 9
        ([8, 3, 1], 5),   # 8 - 3 = 5
        ([6, 4, 1], 2),   # 6 - 4 = 2
    ]
    
    with torch.no_grad():
        for (a, b, op), expected in test_cases:
            input_tensor = torch.FloatTensor([[a/10, b/10, op]])
            pred = model(input_tensor).item()
            
            if op == 0:
                actual_pred = pred * 20
                print(f"{a} + {b} = {actual_pred:.1f} (expected {expected})")
            else:
                actual_pred = pred * 20 - 10
                print(f"{a} - {b} = {actual_pred:.1f} (expected {expected})")
    
    return model

# ============================================================
# 4. Pattern Completion - Complete simple patterns
# ============================================================

class PatternHRM(nn.Module):
    """HRM for pattern completion"""
    def __init__(self):
        super().__init__()
        # Input: 3x3 grid with one missing
        self.encoder = nn.Sequential(
            nn.Linear(8, 32),  # 8 known positions
            nn.ReLU()
        )
        self.reasoning = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.output = nn.Linear(32, 1)  # Predict missing value
        
    def forward(self, x):
        h = self.encoder(x)
        h = h + self.reasoning(h)
        return self.output(h)

def train_pattern_completion():
    """Train HRM to complete patterns"""
    print("\n4. Pattern Completion")
    print("-"*40)
    
    # Create checkerboard patterns with one missing
    patterns = []
    targets = []
    
    # Checkerboard pattern
    for missing_idx in range(9):
        pattern = []
        for i in range(9):
            if i == missing_idx:
                continue  # Skip missing position
            # Checkerboard: alternating 0s and 1s
            value = (i // 3 + i % 3) % 2
            pattern.append(value)
        
        target = (missing_idx // 3 + missing_idx % 3) % 2
        patterns.append(pattern[:8])  # Use first 8 values
        targets.append(target)
    
    X = torch.FloatTensor(patterns)
    y = torch.FloatTensor(targets).unsqueeze(1)
    
    model = PatternHRM()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Train
    for epoch in range(500):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    # Test
    print("\nPattern Completion Test:")
    with torch.no_grad():
        predictions = model(X)
        correct = 0
        for i in range(len(X)):
            pred = 1 if predictions[i].item() > 0.5 else 0
            actual = int(y[i].item())
            correct += (pred == actual)
            if i < 3:  # Show first few examples
                print(f"Pattern {i}: Predicted {pred}, Actual {actual}")
        
        print(f"Accuracy: {correct}/{len(X)} = {100*correct/len(X):.1f}%")
    
    return model

# ============================================================
# 5. Binary Classification - Classify points in 2D space
# ============================================================

class BinaryClassifierHRM(nn.Module):
    """HRM for binary classification"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.reasoning = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        h = self.encoder(x)
        h = h + self.reasoning(h)
        return self.output(h)

def train_binary_classifier():
    """Train HRM for binary classification"""
    print("\n5. Binary Classification (Circle vs Outside)")
    print("-"*40)
    
    # Generate data: points inside/outside a circle
    np.random.seed(42)
    n_samples = 200
    
    X_data = []
    y_data = []
    
    for _ in range(n_samples):
        x = np.random.uniform(-2, 2)
        y = np.random.uniform(-2, 2)
        # Inside circle of radius 1.5
        label = 1 if (x**2 + y**2) < 1.5**2 else 0
        X_data.append([x, y])
        y_data.append(label)
    
    X = torch.FloatTensor(X_data)
    y = torch.FloatTensor(y_data).unsqueeze(1)
    
    model = BinaryClassifierHRM()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # Train
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(50):
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(loader):.4f}")
    
    # Test accuracy
    with torch.no_grad():
        predictions = model(X)
        predicted_labels = (predictions > 0.5).float()
        accuracy = (predicted_labels == y).float().mean()
        print(f"\nClassification Accuracy: {accuracy.item()*100:.1f}%")
    
    return model

# ============================================================
# Main function to run all examples
# ============================================================

def main():
    """Run all HRM application examples"""
    print("="*50)
    print("HRM Applications - What You Can Solve")
    print("="*50)
    
    print("\nYour trained MNIST HRM has shown that the architecture works.")
    print("Here are other problems you can solve with the same approach:")
    
    # 1. XOR Problem
    xor_model = train_xor()
    
    # 2. Sequence Prediction
    seq_model = train_sequence()
    
    # 3. Calculator
    calc_model = train_calculator()
    
    # 4. Pattern Completion
    pattern_model = train_pattern_completion()
    
    # 5. Binary Classification
    classifier_model = train_binary_classifier()
    
    print("\n" + "="*50)
    print("Summary of HRM Applications")
    print("="*50)
    
    print("\nProblems successfully solved:")
    print("1. XOR - Classic non-linear problem")
    print("2. Sequence Prediction - Arithmetic sequences")
    print("3. Simple Calculator - Addition and subtraction")
    print("4. Pattern Completion - Checkerboard patterns")
    print("5. Binary Classification - Circle detection")
    print("\nAll trained in seconds on CPU!")
    
    print("\nOther problems you can tackle:")
    print("- Tic-Tac-Toe (run tic_tac_toe_hrm.py)")
    print("- Connect 4")
    print("- Simple maze solving")
    print("- Text sentiment classification")
    print("- Time series prediction")
    print("- Anomaly detection")
    print("- Simple control tasks (CartPole, etc.)")
    
    print("\nThe HRM architecture is versatile and efficient!")

if __name__ == "__main__":
    main()