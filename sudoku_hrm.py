# sudoku_hrm.py - Sudoku solver using HRM architecture
"""
Train an HRM to solve Sudoku puzzles using constraint satisfaction
and hierarchical reasoning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import random
from typing import List, Tuple, Optional

class SudokuHRM(nn.Module):
    """
    Hierarchical Reasoning Model for Sudoku solving.
    Takes a 9x9 grid (81 inputs) and outputs probability distributions
    for each empty cell.
    """
    
    def __init__(self, hidden_dim=128, num_reasoning_layers=4):
        super().__init__()
        
        # Input: 81 cells (9x9 grid), values 0-9 (0 = empty)
        # We'll use 10 channels per cell for one-hot encoding
        self.input_dim = 81 * 10  # One-hot encoded
        self.hidden_dim = hidden_dim
        
        # Encoder: Process the board state
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # Hierarchical reasoning layers for constraint propagation
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_reasoning_layers)
        ])
        
        # Output: 81 * 9 probabilities (which number for each cell)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 81 * 9)
        )
        
    def forward(self, x):
        """
        Forward pass through the HRM.
        x: Batch of Sudoku boards (batch_size, 81) with values 0-9
        Returns: Probabilities for each cell and number (batch_size, 81, 9)
        """
        batch_size = x.shape[0]
        
        # One-hot encode the input
        x_encoded = self._one_hot_encode(x)
        
        # Encode the board state
        h = self.encoder(x_encoded)
        
        # Apply hierarchical reasoning with residual connections
        for layer in self.reasoning_layers:
            h_new = layer(h)
            h = h + h_new  # Residual connection
        
        # Generate output probabilities
        output = self.output(h)
        output = output.view(batch_size, 81, 9)
        
        return torch.softmax(output, dim=-1)
    
    def _one_hot_encode(self, x):
        """Convert board values to one-hot encoding"""
        batch_size = x.shape[0]
        x_flat = x.view(-1).long()
        one_hot = torch.zeros(x_flat.size(0), 10)
        one_hot.scatter_(1, x_flat.unsqueeze(1), 1)
        return one_hot.view(batch_size, -1)

class SudokuEnvironment:
    """Environment for generating and validating Sudoku puzzles"""
    
    @staticmethod
    def is_valid_move(board, row, col, num):
        """Check if placing num at (row, col) is valid"""
        # Check row
        if num in board[row]:
            return False
        
        # Check column
        if num in board[:, col]:
            return False
        
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if num in board[box_row:box_row+3, box_col:box_col+3]:
            return False
        
        return True
    
    @staticmethod
    def get_valid_moves(board):
        """Get all valid moves for the current board state"""
        valid_moves = []
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    for num in range(1, 10):
                        if SudokuEnvironment.is_valid_move(board, i, j, num):
                            valid_moves.append((i, j, num))
        return valid_moves
    
    @staticmethod
    def is_complete(board):
        """Check if the board is completely filled"""
        return not (board == 0).any()
    
    @staticmethod
    def generate_puzzle(difficulty='easy'):
        """Generate a Sudoku puzzle with a unique solution"""
        # Start with a completed valid Sudoku
        board = np.zeros((9, 9), dtype=int)
        
        # Fill diagonal 3x3 boxes first (they don't affect each other)
        for box in range(3):
            nums = list(range(1, 10))
            random.shuffle(nums)
            for i in range(3):
                for j in range(3):
                    board[box*3 + i, box*3 + j] = nums[i*3 + j]
        
        # Solve the rest using backtracking
        SudokuEnvironment._solve_board(board)
        
        # Remove numbers based on difficulty
        cells_to_remove = {
            'easy': 30,
            'medium': 40,
            'hard': 50,
            'expert': 60
        }.get(difficulty, 30)
        
        # Create puzzle by removing numbers
        puzzle = board.copy()
        positions = [(i, j) for i in range(9) for j in range(9)]
        random.shuffle(positions)
        
        for i, j in positions[:cells_to_remove]:
            puzzle[i, j] = 0
        
        return puzzle, board  # Return puzzle and solution
    
    @staticmethod
    def _solve_board(board):
        """Solve Sudoku using backtracking"""
        empty = SudokuEnvironment._find_empty(board)
        if not empty:
            return True
        
        row, col = empty
        for num in range(1, 10):
            if SudokuEnvironment.is_valid_move(board, row, col, num):
                board[row, col] = num
                
                if SudokuEnvironment._solve_board(board):
                    return True
                
                board[row, col] = 0
        
        return False
    
    @staticmethod
    def _find_empty(board):
        """Find an empty cell"""
        for i in range(9):
            for j in range(9):
                if board[i, j] == 0:
                    return (i, j)
        return None

class SudokuTrainer:
    """Trainer for the Sudoku HRM"""
    
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_supervised(self, num_puzzles=1000, difficulty='easy'):
        """Train the model using supervised learning on generated puzzles"""
        print(f"Training HRM on {num_puzzles} Sudoku puzzles ({difficulty} difficulty)...")
        print("="*60)
        
        losses = []
        accuracies = []
        
        for episode in range(num_puzzles):
            # Generate a puzzle and its solution
            puzzle, solution = SudokuEnvironment.generate_puzzle(difficulty)
            
            # Find cells to predict (empty cells in puzzle)
            empty_cells = (puzzle == 0)
            
            # Convert to tensors
            puzzle_tensor = torch.FloatTensor(puzzle).flatten().unsqueeze(0)
            solution_tensor = torch.LongTensor(solution).flatten()
            
            # Forward pass
            self.model.train()
            output = self.model(puzzle_tensor)  # Shape: (1, 81, 9)
            output = output.squeeze(0)  # Shape: (81, 9)
            
            # Calculate loss only for empty cells
            loss = 0
            correct = 0
            total = 0
            
            for i in range(81):
                if empty_cells.flatten()[i]:
                    # Target is solution[i] - 1 (0-indexed)
                    target = solution_tensor[i] - 1
                    cell_loss = self.criterion(output[i].unsqueeze(0), target.unsqueeze(0))
                    loss = loss + cell_loss
                    
                    # Check accuracy
                    pred = output[i].argmax().item()
                    correct += (pred == target.item())
                    total += 1
            
            if total > 0:
                loss = loss / total
                accuracy = correct / total
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                losses.append(loss.item())
                accuracies.append(accuracy)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                avg_acc = np.mean(accuracies[-100:])
                print(f"Episode {episode+1}: Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2%}")
        
        return losses, accuracies
    
    def solve_puzzle(self, puzzle):
        """Solve a Sudoku puzzle using the trained model"""
        self.model.eval()
        
        board = puzzle.copy()
        attempts = 0
        max_attempts = 100
        
        with torch.no_grad():
            while not SudokuEnvironment.is_complete(board) and attempts < max_attempts:
                # Get model predictions
                board_tensor = torch.FloatTensor(board).flatten().unsqueeze(0)
                output = self.model(board_tensor).squeeze(0)  # Shape: (81, 9)
                
                # Find empty cells and their probabilities
                best_move = None
                best_prob = -1
                
                for i in range(81):
                    row, col = i // 9, i % 9
                    if board[row, col] == 0:
                        # Get probabilities for this cell
                        probs = output[i]
                        
                        # Try numbers in order of probability
                        sorted_nums = torch.argsort(probs, descending=True)
                        for num_idx in sorted_nums:
                            num = num_idx.item() + 1
                            if SudokuEnvironment.is_valid_move(board, row, col, num):
                                prob = probs[num_idx].item()
                                if prob > best_prob:
                                    best_move = (row, col, num)
                                    best_prob = prob
                                break
                
                # Make the best move
                if best_move:
                    row, col, num = best_move
                    board[row, col] = num
                else:
                    # No valid move found
                    break
                
                attempts += 1
        
        return board, attempts

def print_board(board, title="Sudoku Board"):
    """Pretty print a Sudoku board"""
    print(f"\n{title}:")
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")
        
        row_str = "| "
        for j in range(9):
            if j % 3 == 0 and j != 0:
                row_str += "| "
            
            if board[i, j] == 0:
                row_str += ". "
            else:
                row_str += str(int(board[i, j])) + " "
        
        row_str += "|"
        print(row_str)
    print("+" + "-"*7 + "+" + "-"*7 + "+" + "-"*7 + "+")

def test_model(model, num_tests=10, difficulty='easy'):
    """Test the trained model on new puzzles"""
    print(f"\nTesting model on {num_tests} new puzzles...")
    print("="*60)
    
    trainer = SudokuTrainer(model)
    solved_count = 0
    total_attempts = 0
    
    for i in range(num_tests):
        # Generate a new puzzle
        puzzle, solution = SudokuEnvironment.generate_puzzle(difficulty)
        
        # Try to solve it
        solved, attempts = trainer.solve_puzzle(puzzle)
        
        # Check if solved correctly
        is_solved = SudokuEnvironment.is_complete(solved) and \
                   np.array_equal(solved, solution)
        
        if is_solved:
            solved_count += 1
            total_attempts += attempts
            print(f"Puzzle {i+1}: SOLVED in {attempts} steps")
        else:
            print(f"Puzzle {i+1}: FAILED")
        
        # Show first puzzle as example
        if i == 0:
            print_board(puzzle, "Original Puzzle")
            print_board(solved, "Model's Solution")
            print_board(solution, "Correct Solution")
    
    success_rate = solved_count / num_tests * 100
    avg_attempts = total_attempts / solved_count if solved_count > 0 else 0
    
    print(f"\nResults: {solved_count}/{num_tests} solved ({success_rate:.1f}%)")
    if solved_count > 0:
        print(f"Average steps to solve: {avg_attempts:.1f}")
    
    return success_rate

def main():
    """Main training and testing loop"""
    print("="*60)
    print("Sudoku Solver using Hierarchical Reasoning Model")
    print("="*60)
    
    # Create model
    model = SudokuHRM(hidden_dim=128, num_reasoning_layers=4)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel created with {total_params:,} parameters")
    
    # Create trainer
    trainer = SudokuTrainer(model)
    
    # Train on easy puzzles first
    print("\nPhase 1: Training on easy puzzles...")
    start_time = time.time()
    losses_easy, acc_easy = trainer.train_supervised(num_puzzles=500, difficulty='easy')
    
    # Train on medium puzzles
    print("\nPhase 2: Training on medium puzzles...")
    losses_medium, acc_medium = trainer.train_supervised(num_puzzles=300, difficulty='medium')
    
    # Fine-tune on hard puzzles
    print("\nPhase 3: Fine-tuning on hard puzzles...")
    losses_hard, acc_hard = trainer.train_supervised(num_puzzles=200, difficulty='hard')
    
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.1f} minutes")
    
    # Test the model
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    # Test on different difficulties
    easy_success = test_model(model, num_tests=10, difficulty='easy')
    medium_success = test_model(model, num_tests=10, difficulty='medium')
    hard_success = test_model(model, num_tests=5, difficulty='hard')
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Training time: {training_time/60:.1f} minutes")
    print(f"Model parameters: {total_params:,}")
    print(f"Success rates:")
    print(f"  Easy:   {easy_success:.1f}%")
    print(f"  Medium: {medium_success:.1f}%")
    print(f"  Hard:   {hard_success:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), 'sudoku_hrm.pt')
    print(f"\nModel saved to sudoku_hrm.pt")
    
    # Interactive solving
    print("\n" + "="*60)
    print("Would you like to see the model solve a puzzle interactively?")
    user_input = input("Enter 'y' for yes: ")
    
    if user_input.lower() == 'y':
        puzzle, solution = SudokuEnvironment.generate_puzzle('medium')
        print_board(puzzle, "Puzzle to solve")
        
        print("\nSolving...")
        solved, steps = trainer.solve_puzzle(puzzle)
        
        print_board(solved, f"Solution (took {steps} steps)")
        
        if np.array_equal(solved, solution):
            print("✓ Correct solution!")
        else:
            print("✗ Incorrect solution")
            print_board(solution, "Correct solution")

if __name__ == "__main__":
    main()