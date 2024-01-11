import sys
import os

# Get the parent directory (one level up)
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(main_directory)

from tetris_engine import Tetris
from agent_dqn import Agent, QNetwork
import time
import numpy as np
import pygame
import torch
from torch.utils.tensorboard import SummaryWriter

pygame.init()

seed = 50
env = Tetris(10, 20, seed)
agent = Agent(env.state_size, seed=seed)

width, height = 250, 625
screen = pygame.display.set_mode((width, height))

model_name = 'hold_test1'
model_path = f"DQN/models/{model_name}.pt"

model = QNetwork(env.state_size)
model.load_state_dict(torch.load(model_path))
model.eval()

max_episodes = 100
episodes = []
rewards = []
current_max = 0
interval_reward = []
highscore = 0
exit_program = False

log_evaluation = True
log_name = "server_test1"
framerate = 20
run_hold = True
print_interval = 10


if log_evaluation:
    log_dir = "evaluation/" + log_name
    writer = SummaryWriter(log_dir=log_dir)


def logging():
    writer.add_scalar("Total Reward", total_reward, episode)


for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    total_reward = 0

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.toggle_render()  # Toggle render state with 'r'
                if event.key == pygame.K_q:
                    exit_program = True
                if event.type == pygame.QUIT:
                    exit_program = True

        if exit_program:
            break

        env.render(total_reward, framerate)

        if run_hold:
            next_states = env.merge_next_states()
        else:
            next_states = env.get_next_states(env.shape, env.anchor, False)

        # If the dictionary is empty, meaning the game is over
        if not next_states:
            break

        states = list(next_states.values())
        # Tell the agent to choose the best possible state
        best_state = agent.act(
            states=states, model=model, use_epsilon=False
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

    if exit_program:
        break

    if log_evaluation:
        logging()

    episodes.append(episode)
    rewards.append(total_reward)

    if len(interval_reward) <= print_interval:
        interval_reward.append(total_reward)
    else:
        interval_reward = []

    if total_reward > highscore:
        highscore = total_reward

    if episode % print_interval == 0:
        print(f"Running episode {str(episode)}")
        print(f"Mean reward:  {str(np.mean(interval_reward))}")
        print(f"Round Highscore: {str(max(interval_reward))}")
        print(f"Training Highscore: {str(highscore)}")


pygame.quit()

if log_evaluation:
    writer.close()

print(highscore)
