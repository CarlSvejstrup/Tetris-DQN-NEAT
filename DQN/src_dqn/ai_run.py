from tetris_engine import Tetris
from agent_dqn import Agent, QNetwork
import time
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter


pygame.init()

env = Tetris(10, 20)
agent = Agent(env.state_size)

model_path = "DQN/models/model2.pt"

model = QNetwork(env.state_size)
model.load_state_dict(torch.load(model_path))
model.eval()

max_episodes = 1
episodes = []
rewards = []
current_max = 0
log_evaluation = True
framerate = 1

for name, param in model.named_parameters():
    print(f"Parameter name: {name}")
    print(f"Parameter shape: {param.shape}")
    print(f"Parameter values: {param}")
    print("\n")

if log_evaluation:
    log_folder = "run1"
    log_dir = "evaluation/" + log_folder
    writer = SummaryWriter(log_dir=log_dir)

def logging():
    writer.add_scalar("Total Reward", total_reward, episode)


for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    total_reward = 0

    print("Running episode " + str(episode))

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    quit()

        env.render(total_reward, framerate)

        next_states = env.get_next_states()

        # If the dictionary is empty, meaning the game is over
        if not next_states:
            break

        # Tell the agent to choose the best possible state
        best_state = agent.act(
            states=list(next_states.values()), model=model, use_epsilon=False
        )

        # Grab the best tetromino position and its rotation chosen by the agent
        best_action = None
        for action, state in next_states.items():
            if (best_state == state).all():
                best_action = action
                break

        reward, done = env.step(best_action)
        total_reward += reward

        current_state = next_states[best_action]

        logging()

    episodes.append(episode)
    rewards.append(total_reward)

    print("Total reward: " + str(total_reward))

writer.close()
