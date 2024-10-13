import torch
from model import ChessCNN
from agent import ChessAI
from torch.utils.tensorboard import SummaryWriter
import datetime

if __name__ == "__main__":
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = "chess_ai"
    model_name = "model1"
    log_dir = f'logs/{project_name}/{model_name}/runs/{timestamp}'

    writer = SummaryWriter(log_dir)
    
    number_of_legal_moves = 4096  # 64**2 to represent the starting and ending states
    
    # Initialize the model and the AI agent
    model = ChessCNN(number_of_legal_moves)
    chess_ai = ChessAI(model)

    # Train the model via self-play
    num_episodes = 1000
    for episode in range(num_episodes):
        game_data, outcome = chess_ai.self_play()
        # Training logic here
    
    writer.close()
