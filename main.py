import torch
from model import ChessCNN
from agent import ChessAI
from torch.utils.tensorboard import SummaryWriter
import datetime
import os

def intro_ui():
    while True:
        choice = input("Choose to train a new model [\"NEW\"], load and train a current model [\"LOAD\"] or play against a model [\"PLAY\"]: ")
        if choice.lower() == "new":
            model_name = input("Please input a model name: ")
                    
            # Initialize the model and the AI agent
            model = ChessCNN(number_of_legal_moves)
            chess_ai = ChessAI(model)

            return model_name, chess_ai

        if choice.lower() == "load":
            
            while True:
                list_saved_models('')
                model_choice = input("Pick one of the models to load")

                try: 
                    checkpoint = torch.load(f'{model_choice}.pth')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    chess_ai = ChessAI(model)

                    return model_choice, chess_ai
                except:
                    print('That seemed to not have work')
                    print('---------')
        
        if choice.lower() == 'play':
            print('Not currently running, sorry')


if __name__ == "__main__":

    number_of_legal_moves = 4096  # 64**2 to represent the starting and ending states

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = "chess_ai"
    model_name, chess_ai = intro_ui()
    log_dir = f'logs/{project_name}/{model_name}/runs/{timestamp}'

    writer = SummaryWriter(log_dir)

    # Train the model via self-play
    num_episodes = 1000
    for episode in range(num_episodes):
        game_data = chess_ai.self_play()

        # Train both White and Black with their respective data
        loss_white = chess_ai.train_model(data=game_data[0], model=chess_ai.model, optimiser=chess_ai.optimiser, factor = 1)
        loss_black =chess_ai.train_model(data=game_data[1], model=chess_ai.model, optimiser=chess_ai.optimiser, factor = -1)

        avg_loss = (loss_white + loss_black) / 2

        print('Epoch:',  episode, '; Loss:', avg_loss)

        if episode % 10 == 0:
            torch.save({
                'epoch': episode,
                'model_state_dict': chess_ai.model.state_dict(),
                'optimizer_state_dict': chess_ai.optimiser.state_dict(),
                'loss': avg_loss,
            }, f'{model_name}_epoch_{episode}.pth')

            print('Model saved')

    writer.close()
