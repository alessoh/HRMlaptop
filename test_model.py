# test_model.py - Test your trained HRM model
"""
Test the trained HRM model on MNIST test set and individual samples.
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import json

# Recreate model architecture
class WindowsHRM(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10, num_layers=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h = self.encoder(x)
        for layer in self.reasoning_layers:
            h = h + layer(h)
        return self.output(h)

def load_model(model_path='checkpoints/hrm_final.pt'):
    """Load trained model"""
    if not os.path.exists(model_path):
        # Try alternative paths
        alt_paths = ['hrm_model.pt', 'simple_hrm_model.pt']
        for path in alt_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            print(f"Error: No model file found!")
            print("Please train a model first using: python hrm_trainer.py")
            return None
    
    print(f"Loading model from {model_path}...")
    model = WindowsHRM()
    
    # Load state dict - use weights_only=False for compatibility with saved checkpoint
    # This is safe because we created the checkpoint ourselves
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model

def test_accuracy(model):
    """Test model on full test set"""
    print("\nTesting on MNIST test set...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    correct = 0
    total = 0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            c = (predicted == target).squeeze()
            for i in range(target.size(0)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Overall accuracy
    overall_accuracy = 100 * correct / total
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"Correctly classified: {correct}/{total}")
    
    # Per-class accuracy
    print("\nPer-digit accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"Digit {i}: {acc:.1f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    return overall_accuracy

def visualize_predictions(model, num_samples=10):
    """Visualize model predictions on random samples"""
    print(f"\nVisualizing {num_samples} random predictions...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    # Random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        image, true_label = test_dataset[sample_idx]
        
        # Predict
        with torch.no_grad():
            output = model(image.unsqueeze(0))
            probabilities = torch.nn.functional.softmax(output, dim=1)
            predicted_label = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_label].item()
        
        # Display
        img_display = image.squeeze().numpy()
        axes[idx].imshow(img_display, cmap='gray')
        
        # Color code: green if correct, red if wrong
        color = 'green' if predicted_label == true_label else 'red'
        axes[idx].set_title(f'True: {true_label}, Pred: {predicted_label}\n'
                           f'Conf: {confidence:.2%}', color=color)
        axes[idx].axis('off')
    
    plt.suptitle('Model Predictions on Random Test Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()
    print("Predictions saved to predictions.png")

def test_single_digit(model):
    """Interactive single digit testing"""
    print("\nInteractive digit testing (press 'q' to quit)")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    
    while True:
        user_input = input("\nEnter a test image index (0-9999) or 'q' to quit: ")
        
        if user_input.lower() == 'q':
            break
            
        try:
            idx = int(user_input)
            if idx < 0 or idx >= len(test_dataset):
                print(f"Please enter a number between 0 and {len(test_dataset)-1}")
                continue
                
            image, true_label = test_dataset[idx]
            
            # Predict
            with torch.no_grad():
                output = model(image.unsqueeze(0))
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted_label = output.argmax(dim=1).item()
            
            # Show probabilities for all classes
            print(f"\nTrue label: {true_label}")
            print(f"Predicted: {predicted_label}")
            print("\nProbabilities for each digit:")
            for i in range(10):
                prob = probabilities[0][i].item()
                bar = '█' * int(prob * 20)
                print(f"  {i}: {bar:<20} {prob:.2%}")
            
            # Visualize
            plt.figure(figsize=(6, 6))
            plt.imshow(image.squeeze().numpy(), cmap='gray')
            plt.title(f'True: {true_label}, Predicted: {predicted_label}')
            plt.axis('off')
            plt.show()
            
        except ValueError:
            print("Please enter a valid number or 'q'")
        except Exception as e:
            print(f"Error: {e}")

def analyze_errors(model):
    """Analyze model errors"""
    print("\nAnalyzing model errors...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    confusion_matrix = np.zeros((10, 10), dtype=int)
    errors = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(target.size(0)):
                true_label = target[i].item()
                pred_label = predicted[i].item()
                confusion_matrix[true_label][pred_label] += 1
                
                if true_label != pred_label:
                    errors.append({
                        'index': batch_idx * 100 + i,
                        'true': true_label,
                        'predicted': pred_label
                    })
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    print("     ", end="")
    for i in range(10):
        print(f"{i:4}", end="")
    print("\n" + "-" * 45)
    
    for i in range(10):
        print(f"{i:2} | ", end="")
        for j in range(10):
            if i == j:
                print(f"\033[92m{confusion_matrix[i][j]:4}\033[0m", end="")  # Green for correct
            elif confusion_matrix[i][j] > 0:
                print(f"\033[91m{confusion_matrix[i][j]:4}\033[0m", end="")  # Red for errors
            else:
                print(f"{confusion_matrix[i][j]:4}", end="")
        print()
    
    # Most common errors
    print(f"\nTotal errors: {len(errors)}")
    if errors:
        error_pairs = {}
        for error in errors:
            pair = (error['true'], error['predicted'])
            error_pairs[pair] = error_pairs.get(pair, 0) + 1
        
        sorted_errors = sorted(error_pairs.items(), key=lambda x: x[1], reverse=True)
        print("\nMost common misclassifications:")
        for (true, pred), count in sorted_errors[:5]:
            print(f"  {true} → {pred}: {count} times")
    
    return confusion_matrix, errors

def main():
    """Main testing function"""
    print("=" * 50)
    print("HRM Model Testing")
    print("=" * 50)
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # Test accuracy
    accuracy = test_accuracy(model)
    
    # Analyze errors
    confusion_matrix, errors = analyze_errors(model)
    
    # Visualize predictions
    visualize_predictions(model)
    
    # Save test results
    results = {
        'test_accuracy': accuracy,
        'total_parameters': total_params,
        'num_errors': len(errors),
        'confusion_matrix': confusion_matrix.tolist()
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to test_results.json")
    
    # Interactive testing
    user_input = input("\nDo you want to test individual digits? (y/n): ")
    if user_input.lower() == 'y':
        test_single_digit(model)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()