import chess
import torch

def convert_board_to_tensor(board):
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
    if not board.turn:  # It's Black's turn
        tensor = tensor.flip(1).flip(2)  # Rotate 180 degrees
        tensor = tensor * -1  # Multiply by -1

    return tensor

def save_game_data(states,actions,outcomes):

