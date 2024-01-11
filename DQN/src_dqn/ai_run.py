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

seed = 44
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
tetris_clear_list = []
current_max = 0
interval_reward = []
highscore = 0
exit_program = False

log_evaluation = True
log_name = "model1"
framerate = 10
run_hold = True
print_interval = 1


if log_evaluation:
    log_dir = "./DQN/evaluation/" + log_name
    writer = SummaryWriter(log_dir=log_dir)


def logging():
    writer.add_scalar("Total Reward", total_reward, episode)


for episode in range(max_episodes):
    current_state = env.reset()
    done = False
    total_reward = 0
    env.tetris_amount = 0
    start_time = time.time()

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

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time
    if elapsed_time_seconds < 60:
        seconds = round(elapsed_time_seconds, 2)
        minutes = 0
    else:
        minutes = int(elapsed_time_seconds // 60)
        seconds = int(elapsed_time_seconds % 60)

    if exit_program:
        break

    if log_evaluation:
        logging()

    episodes.append(episode)
    rewards.append(total_reward)
    tetris_clear_list.append(env.tetris_amount)

    if total_reward > highscore:
        highscore = total_reward

    # Print training data
    if episode % print_interval == 0:
        print("-" * 30)
        print(f"Running episode {str(episode + 1)}")
        print(f"Mean reward:  {str(np.mean(rewards[-print_interval:]))}")
        print(f"Round Highscore: {str(max(rewards[-print_interval:]))}")
        print(f"Training Highscore: {str(highscore)}")
        print(f"Round 'tetris-clear' highscore:{str(max(tetris_clear_list[-print_interval:]))}")
        print(f"'tetris-clear' highscore:{str(max(tetris_clear_list))}")
        print(f'episodetime: {minutes} minutes, {seconds} seconds')


pygame.quit()

if log_evaluation:
    writer.close()

print(highscore)
