import chess
import chess.svg
import torch
import torch.optim as optim
from model import ChessCNN
from utils import *

class ChessAI:
    def __init__(self, model, debug=False):
        self.model = model
        self.optimiser = optim.Adam(model.parameters(), lr=0.001)

    def self_play(self, debug=False):
        board = chess.Board()  # Start a new game

        white_states = []
        white_actions = []
        black_states = []
        black_actions = []

        while not board.is_game_over():
            state = board.copy()

            if board.turn == chess.WHITE:
                move, valid = self.choose_move(board)  # White's turn
                white_states.append(state)  # Store White's state
                white_actions.append(move)  # Store White's action
            else:
                move, valid = self.choose_move(board)  # Black's turn
                black_states.append(state)  # Store Black's state
                black_actions.append(move)  # Store Black's action
            
            #Handling pawn promotion
            if not valid:
                if board.piece_at(move.from_square).piece_type == chess.PAWN and chess.square_rank(move.to_square) in [0, 7]:
                    move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
                    if debug:
                        print('Invalidity dealt with')
                    elif debug:
                        print('There is another error')
                else:
                    print('Problem move chosen, ', move)
                    board_svg = chess.svg.board(board)
                    with open("chess_board.svg", "w") as f:
                        f.write(board_svg)
                    print(state)

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
            white_rewards = [-0.1] * len(white_states)  # Draw
            black_rewards = [0.1] * len(black_states)  # Draw

        white_data = [white_states,white_actions,white_rewards]
        black_data = [black_states,black_actions,black_rewards]

        return [white_data, black_data]

    def choose_move(self, board, debug=False):
        # Convert the board to the input tensor. Flips it round if its black's turn
        input_tensor = convert_board_to_tensor(board)
        input_tensor = input_tensor.unsqueeze(0)

        if debug:
            print("Input tensor shape:", input_tensor.shape)

        with torch.no_grad():
            move_probs = self.model(input_tensor)  # For black will once trained, give moves that need to be flipped back

            if debug:
                print("Model output shape:", move_probs.shape)

        # Generate the mask for legal moves, feeding in true state so function must flip
        legal_moves_mask = self.get_legal_moves_mask(board)
        if debug:
            print("Legal moves mask shape:", legal_moves_mask.shape)
            print("List of actual legal moves: ", sorted([move.uci() for move in list(board.legal_moves)]))
            print("List of mask's legal moves: ", sorted([move.uci() for move in mask_to_legal_moves(legal_moves_mask)]))

        # Apply the mask to move_probs (setting probabilities of illegal moves to 0)
        masked_move_probs = move_probs * legal_moves_mask
        
        if debug:
            print("Masked move probs shape:", masked_move_probs.shape)
        

        # Sample a move using the masked probabilities, will output a move that doesn't need to be flipped
        move, valid = self.sample_move(masked_move_probs, board, legal_moves_mask)
        if debug:
            print("Final Selected move: ", move)
            print("----------------")

        return move, valid

    def get_legal_moves_mask(self, board):
        # Initialize a mask of zeros for all 4096 possible moves
        mask = torch.zeros(4096)
        
        # Get all legal moves
        legal_moves = list(board.legal_moves)  # Legal moves as a list of Move objects
        
        for move in legal_moves:
            # Convert the move to an index corresponding to the 64x64 flattened tensor, this will handle perspectives
            move_index = convert_move_to_index(move, board.turn)
            
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
        move = convert_index_to_move(move_index, board)
        
        if move in board.legal_moves:
            valid = True
            if debug:
                print("Valid move made")
        else:
            valid = False
            if debug:
                print(f"Invalid move sampled: {move}")

        return move, valid

    def train_model(self, data, model, optimiser, factor = 1, debug = False):
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
        
        total_loss = 0

        optimiser.zero_grad()
        for i in range(len(states)):
            input_tensor = convert_board_to_tensor(states[i], training = True)
            if factor == -1:
                input_tensor = input_tensor.flip(1).flip(2)  # Rotate 180 degrees
                input_tensor = input_tensor * -1
            logits_input_tensor = input_tensor.unsqueeze(0)
            logits = model(logits_input_tensor)
            if factor == -1:
                board_state = False
            else:
                board_state = True
            action_index = torch.tensor([convert_move_to_index(actions[i], whites_turn = board_state)])
            loss = self.compute_loss(logits, action_index, rewards[i])
            loss.backward()

            total_loss += loss.item()
        optimiser.step()

        return total_loss

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


