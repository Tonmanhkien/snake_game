import torch
from game import SnakeGameAI
from model import Linear_QNet
from agent import Agent

def load_model(input_size, hidden_size, output_size, model_path):
    model = Linear_QNet(input_size, hidden_size, output_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def play_game(model,agent):
    agent = Agent()
    game = SnakeGameAI()
    while True:
        state = agent.get_state(game)
        state_tensor = torch.tensor(state, dtype=torch.float)
        
        with torch.no_grad():
            prediction = model(state_tensor)
            action = torch.argmax(prediction).numpy()
        
        reward, game_over, score = game.play_step(action)
        
        if game_over:
            break

        game._update_ui()

if __name__ == '__main__':
    agent = Agent()
    model = load_model(11, 256, 3, '/Users/macbook/snake_game_club/model_best_till_now/model.pth')
    play_game(model,agent)

