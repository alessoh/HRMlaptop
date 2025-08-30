# tic_tac_toe_hrm.py - Train HRM to play Tic-Tac-Toe
"""
Apply the HRM architecture to learn Tic-Tac-Toe strategy.
The model learns to predict the best move given a board state.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import time
import json

class TicTacToeHRM(nn.Module):
    """HRM adapted for Tic-Tac-Toe (9 inputs, 9 outputs)"""
    
    def __init__(self, hidden_dim=32, num_layers=2):
        super().__init__()
        
        # Input: 9 positions (flattened 3x3 board)
        # Output: 9 positions (move probabilities)
        # Note: No BatchNorm for single-sample inference
        self.encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Hierarchical reasoning layers without BatchNorm
        self.reasoning_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Output layer with sigmoid for move probabilities
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, 9),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Ensure batch dimension
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        h = self.encoder(x)
        
        # Apply reasoning with residual connections
        for layer in self.reasoning_layers:
            h = h + layer(h)
            
        return self.output(h)

class TicTacToeGame:
    """Tic-Tac-Toe game environment"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset the board"""
        self.board = np.zeros(9, dtype=np.float32)  # 0=empty, 1=X, -1=O
        self.current_player = 1  # X starts
        return self.board.copy()
    
    def get_valid_moves(self):
        """Return list of valid move indices"""
        return [i for i in range(9) if self.board[i] == 0]
    
    def make_move(self, position):
        """Make a move, return (new_board, reward, done)"""
        if self.board[position] != 0:
            return self.board.copy(), -10, True  # Invalid move penalty
        
        self.board[position] = self.current_player
        
        # Check for win
        if self.check_winner() == self.current_player:
            return self.board.copy(), 10, True  # Win reward
        
        # Check for draw
        if len(self.get_valid_moves()) == 0:
            return self.board.copy(), 0, True  # Draw
        
        # Game continues
        self.current_player *= -1  # Switch player
        return self.board.copy(), 0, False
    
    def check_winner(self):
        """Check if there's a winner"""
        # Win patterns (indices)
        wins = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        
        for pattern in wins:
            values = [self.board[i] for i in pattern]
            if values[0] != 0 and values[0] == values[1] == values[2]:
                return values[0]
        
        return 0  # No winner
    
    def print_board(self):
        """Print the board in a nice format"""
        symbols = {0: '.', 1: 'X', -1: 'O'}
        for i in range(3):
            row = [symbols[self.board[i*3 + j]] for j in range(3)]
            print(' '.join(row))
        print()

class TicTacToeTrainer:
    """Train HRM to play Tic-Tac-Toe using self-play"""
    
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)  # Experience replay
        self.epsilon = 0.3  # Exploration rate
        
    def get_move(self, board, valid_moves, training=True):
        """Get move from model with epsilon-greedy exploration"""
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves)
        
        # Set model to eval mode for single sample inference
        self.model.eval()
        with torch.no_grad():
            board_tensor = torch.FloatTensor(board)
            move_probs = self.model(board_tensor).squeeze()
            
            # Mask invalid moves
            for i in range(9):
                if i not in valid_moves:
                    move_probs[i] = -float('inf')
            
            move = move_probs.argmax().item()
        
        # Return to training mode
        if training:
            self.model.train()
        
        return move
    
    def train_self_play(self, episodes=1000):
        """Train through self-play"""
        game = TicTacToeGame()
        history = {'wins': 0, 'losses': 0, 'draws': 0}
        
        print("Training HRM on Tic-Tac-Toe through self-play...")
        print("="*50)
        
        for episode in range(episodes):
            game.reset()
            states = []
            actions = []
            
            # Play one game
            done = False
            while not done:
                state = game.board.copy()
                valid_moves = game.get_valid_moves()
                
                # Get move from model
                if game.current_player == 1:  # Model plays X
                    action = self.get_move(state, valid_moves)
                    states.append(state)
                    actions.append(action)
                else:  # Random opponent plays O
                    action = random.choice(valid_moves)
                
                _, reward, done = game.make_move(action)
            
            # Store experience
            winner = game.check_winner()
            if winner == 1:
                history['wins'] += 1
                final_reward = 1.0
            elif winner == -1:
                history['losses'] += 1
                final_reward = -1.0
            else:
                history['draws'] += 1
                final_reward = 0.5
            
            # Add to memory with rewards
            for state, action in zip(states, actions):
                target = torch.zeros(9)
                target[action] = final_reward
                self.memory.append((state, target))
            
            # Train on batch from memory
            if len(self.memory) >= 32:
                self.train_batch()
            
            # Decay exploration
            if episode % 100 == 0:
                self.epsilon *= 0.95
                self.epsilon = max(self.epsilon, 0.1)
            
            # Print progress
            if (episode + 1) % 100 == 0:
                win_rate = history['wins'] / (episode + 1) * 100
                print(f"Episode {episode+1}: Win rate: {win_rate:.1f}% "
                      f"(W:{history['wins']} L:{history['losses']} D:{history['draws']})")
        
        return history
    
    def train_batch(self, batch_size=32):
        """Train on a batch from memory"""
        if len(self.memory) < batch_size:
            return
        
        # Set model to train mode for batch training
        self.model.train()
        
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([s for s, _ in batch])
        targets = torch.stack([t for _, t in batch])
        
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()

def play_against_human(model):
    """Play a game against a human"""
    game = TicTacToeGame()
    model.eval()
    
    print("\nPlay Tic-Tac-Toe against the HRM!")
    print("You are O, HRM is X")
    print("Enter moves as 0-8 (row by row):")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print("="*30)
    
    game.reset()
    game.print_board()
    
    while True:
        if game.current_player == 1:  # HRM's turn (X)
            print("HRM is thinking...")
            board_tensor = torch.FloatTensor(game.board)
            with torch.no_grad():
                move_probs = model(board_tensor).squeeze()
                
                # Mask invalid moves
                valid_moves = game.get_valid_moves()
                for i in range(9):
                    if i not in valid_moves:
                        move_probs[i] = -float('inf')
                
                move = move_probs.argmax().item()
            
            print(f"HRM plays position {move}")
        else:  # Human's turn (O)
            valid_moves = game.get_valid_moves()
            while True:
                try:
                    move = int(input(f"Your move (valid: {valid_moves}): "))
                    if move in valid_moves:
                        break
                    print("Invalid move! Try again.")
                except:
                    print("Enter a number 0-8")
        
        _, _, done = game.make_move(move)
        game.print_board()
        
        if done:
            winner = game.check_winner()
            if winner == 1:
                print("HRM wins!")
            elif winner == -1:
                print("You win!")
            else:
                print("It's a draw!")
            break
    
    play_again = input("\nPlay again? (y/n): ")
    if play_again.lower() == 'y':
        play_against_human(model)

def main():
    """Main training and playing loop"""
    print("HRM for Tic-Tac-Toe")
    print("="*50)
    
    # Create model
    model = TicTacToeHRM(hidden_dim=32, num_layers=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    trainer = TicTacToeTrainer(model)
    start_time = time.time()
    
    history = trainer.train_self_play(episodes=2000)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final win rate: {history['wins']/20:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), 'tic_tac_toe_hrm.pt')
    print("Model saved to tic_tac_toe_hrm.pt")
    
    # Play against human
    play_against_human(model)

if __name__ == "__main__":
    main()