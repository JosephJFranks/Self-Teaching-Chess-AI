import chess
import torch
import torch.optim as optim
from model import ChessCNN
from utils import *

class ChessAI:
    def __init__(self, model, debug=False):
        self.model = model
        self.optimiser = optim.Adam(model.parameters(), lr=0.001)

    def choose_move(self, board, debug=False):
        # Convert the board to the input tensor
        input_tensor = convert_board_to_tensor(board)
        input_tensor = input_tensor.unsqueeze(0)

        if debug:
            print("Input tensor shape:", input_tensor.shape)

        with torch.no_grad():
            move_probs = self.model(input_tensor)  # Shape: (4096,)
            if debug:
                print("Model output shape:", move_probs.shape)

        # Generate the mask for legal moves
        legal_moves_mask = self.get_legal_moves_mask(board)
        if debug:
            print("Legal moves mask shape:", legal_moves_mask.shape)
            print("List of actual legal moves: ", sorted([move.uci() for move in list(board.legal_moves)]))
            print("List of mask's legal moves: ", sorted([move.uci() for move in mask_to_legal_moves(legal_moves_mask)]))

        # Apply the mask to move_probs (setting probabilities of illegal moves to 0)
        masked_move_probs = move_probs * legal_moves_mask
        
        if debug:
            print("Masked move probs shape:", masked_move_probs.shape)
        

        # Sample a move using the masked probabilities
        move = self.sample_move(masked_move_probs, board, legal_moves_mask)
        if debug:
            print("Final Selected move: ", move)
            print("----------------")

        return move

    def get_legal_moves_mask(self, board):
        # Initialize a mask of zeros for all 4096 possible moves
        mask = torch.zeros(4096)
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)  # Legal moves as a list of Move objects
        
        for move in legal_moves:
            # Convert the move to an index corresponding to the 64x64 flattened tensor
            move_index = convert_move_to_index(move)
            
            # Set the corresponding index in the mask to 1 (legal move)
            mask[move_index] = 1
        
        return mask

    def sample_move(self, move_probs, board, mask, debug=False):
        if debug:
            print("Move probs shape:", move_probs.shape)
            print_non_zero_probabilities(move_probs)

        # Sample based on probabilities
        move_index = torch.multinomial(move_probs, 1).item()
        
        # Convert move_index back to a chess.Move
        move = convert_index_to_move(move_index)
        
        if move in board.legal_moves:
            print("Valid move made")
        else:
            print(f"Invalid move sampled: {move}")

        return move


    def self_play(self):
        board = chess.Board()  # Start a new game

        white_states = []
        white_actions = []
        black_states = []
        black_actions = []

        while not board.is_game_over():
            state = board.copy()

            if board.turn == chess.WHITE:
                move = self.choose_move(board)  # White's turn
                white_states.append(state)  # Store White's state
                white_actions.append(move)  # Store White's action
            else:
                move = self.choose_move(board)  # Black's turn
                black_states.append(state)  # Store Black's state
                black_actions.append(move)  # Store Black's action

            board.push(move)  # Apply the move

        # Assign rewards based on game outcome
        outcome = board.result()  # '1-0', '0-1', '1/2-1/2'
        print(check_winner(board))

        if outcome == '1-0':
            white_rewards = [1] * len(white_states)  # White wins
            black_rewards = [-1] * len(black_states)  # Black loses
        elif outcome == '0-1':
            white_rewards = [-1] * len(white_states)  # White loses
            black_rewards = [1] * len(black_states)  # Black wins
        else:
            white_rewards = [0] * len(white_states)  # Draw
            black_rewards = [0] * len(black_states)  # Draw

        white_data = [white_states,white_actions,white_rewards]
        black_data = [black_states,black_actions,black_rewards]

        return [white_data, black_data]

    def train_model(self, data, model, optimiser, debug=False):
        states = data[0]
        actions = data[1]
        rewards = data[2]
        
        if debug:
            print(states)
            print()
            print()
            print(actions)
            print()
            print()
            print(rewards)

        model.train()

        optimiser.zero_grad()
        for i in range(len(states)):
            input_tensor = convert_board_to_tensor(states[i])
            input_tensor = input_tensor.unsqueeze(0)
            logits = model(input_tensor)
            action_index = torch.tensor([convert_move_to_index(actions[i])])
            loss = self.compute_loss(logits, action_index, rewards[i])
            loss.backward()
        optimiser.step()

    def compute_loss(self, logits, actions, rewards):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(logits, actions) * rewards

    def calculate_discounted_rewards(self, rewards, gamma=0.99):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        return discounted_rewards


