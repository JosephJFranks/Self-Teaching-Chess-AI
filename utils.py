import chess
import torch
import os

def convert_board_to_tensor(board,training = False):
    # Initialize a tensor of shape (6, 8, 8) filled with zeros
    tensor = torch.zeros((6, 8, 8), dtype=torch.float32)

    # Define the piece values (1 for white pieces, -1 for black pieces)
    piece_map = {
        chess.PAWN: 1,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 1,
        chess.QUEEN: 1,
        chess.KING: 1,
    }

    # Fill the tensor with piece information
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 1 if piece.color == chess.WHITE else -1
            tensor[piece.piece_type - 1, square // 8, square % 8] = color

    # Check whose turn it is and adjust the tensor if it's Black's turn
    if not board.turn and not training:  # It's Black's turn
        tensor = tensor.flip(1).flip(2)  # Rotate 180 degrees
        tensor = tensor * -1  # Multiply by -1

    return tensor

def convert_move_to_index(move, whites_turn = True):
        # Each move is represented by the from_square and to_square
        from_square = move.from_square  # Integer from 0 to 63
        to_square = move.to_square      # Integer from 0 to 63

        if not whites_turn:
            from_square = 63 - from_square
            to_square = 63 - to_square
        
        # Convert the move into a 64x64 flattened index
        move_index = from_square * 64 + to_square
        
        return move_index

def convert_index_to_move(index, board):
        # From the index, recover the from_square and to_square
        from_square = index // 64
        to_square = index % 64
        
        if not board.turn:
            from_square = 63 - from_square
            to_square = 63 - to_square
        
        # Create and return a chess.Move object
        move = chess.Move(from_square, to_square)
        
        return move

def mask_to_legal_moves(mask):
    legal_moves = []

    # Iterate through the mask
    for index, value in enumerate(mask):
        if value.item() == 1:  # If the move is legal (1 in the mask)
            move = convert_index_to_move(index)  # Convert index back to Move
            legal_moves.append(move)

    return legal_moves

def check_winner(board):
    if board.is_checkmate():
        if board.turn:  # True if it's white's turn
            return "Black wins!"
        else:
            return "White wins!"
    elif board.is_stalemate():
        return "It's a stalemate!"
    elif board.is_insufficient_material():
        return "It's a draw due to insufficient material!"
    elif board.is_seventyfive_moves():
        return "It's a draw due to the 75-move rule!"
    elif board.is_fivefold_repetition():
        return "It's a draw due to fivefold repetition!"
    else:
        return "The game is still ongoing."

def print_non_zero_probabilities(move_probs):
    # Get the move probabilities as a numpy array
    useable = move_probs.numpy()
    print(useable[useable!=0])

    listed_move_probs = move_probs.detach().cpu().numpy().flatten()

    print("Moves with non-zero probabilities:")
    for index, prob in enumerate(listed_move_probs):
        if prob != 0:  # Check if probability is non-zero
            print(f"Move: {convert_index_to_move(index)}, Probability: {prob:.4f}")

def list_saved_models(directory):
    try:
        # List all files in the directory
        files = os.getcwd()
        
        # Filter for model files (you can adjust the extension based on your framework)
        model_files = [f for f in os.listdir(directory) if f.endswith('.pt') or f.endswith('.pth')]
        
        if model_files:
            print("Saved Models:")
            for model in model_files:
                print(model)
        else:
            print("No model files found.")
    
    except FileNotFoundError:
        print("The specified directory does not exist.")

import chess.pgn

def save_game_to_pgn(game_moves, file_name="game.pgn"):
    # Create a PGN game
    game = chess.pgn.Game()

    # Set headers for PGN (optional, you can add player names, etc.)
    game.headers["Event"] = "Self-Play"
    game.headers["Site"] = "Local"
    game.headers["Date"] = "2024.10.24"
    game.headers["Round"] = "1"
    game.headers["White"] = "Chess AI (White)"
    game.headers["Black"] = "Chess AI (Black)"

    # Create a board
    board = chess.Board()

    # Add all the moves to the game
    node = game
    for move in game_moves:
        if isinstance(move, str):  # If the move is in UCI format, convert to chess.Move
            move = chess.Move.from_uci(move)
        node = node.add_variation(move)
        board.push(move)

    # Write the PGN to a file
    with open(file_name, "w") as pgn_file:
        exporter = chess.pgn.FileExporter(pgn_file)
        game.accept(exporter)

    print(f"Game saved to {file_name}")