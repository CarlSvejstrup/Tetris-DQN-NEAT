import sys
import os

# Get the parent directory (one level up)
main_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(main_directory)

from tetris_engine import Tetris
from agent_dqn import Agent
import time
import pygame
from torch.utils.tensorboard import SummaryWriter
import numpy as np


## TODO
# Seed setup
# Server setup
# Hyperprameters

pygame.init()

# Initializing pygame window
width, height = 250, 625
screen = pygame.display.set_mode((width, height))

# set seed
seed = 12

# Initialize tetris environment
env = Tetris(10, 20, seed)

# Initialize training variables
max_episode = 4000
max_steps = 250000
max_reward = 250000


# Log parameters
print_interval = 10
interval_reward = []

framerate = 2
save_log = False
log_name = "server_test1"
save_model = False
model_name = "hold_test1"
exit_program = False
run_hold = True

# Reward system
env.reward_system = 1

""""
env.reward_system = 1
Oldschool tetris reward system

env.reward_system = 2
Reward = cleared_lines**2 * self.width + self.soft_count

env.reward_system = 3
Reward = cleared_lines**2 * self.width + 1
"""


# Initializing agent
agent = Agent(
    env.state_size,
    memory_size=30000,
    discount=0.98,
    epsilon_min=0.001,
    epsilon_end_episode=3000,
    batch_size=512,
    episodes_per_update=1,
    replay_start=3000,
    learning_rate=0.001,
    seed=seed,
)

episodes = []
rewards = []
tetris_clear_list = []
current_max = 0
highscore = 0

# Creating log writer
if save_log:
    log_dir = "./DQN/training_logs/" + log_name
    writer = SummaryWriter(log_dir=log_dir)

    # Logging text
    writer.add_text("Max episode", str(max_episode))
    writer.add_text("Learningrate", str(agent.learning_rate))
    writer.add_text("Replaystart", str(agent.replay_start))
    writer.add_text("Batchsize", str(agent.batch_size))
    writer.add_text("Discount value", str(agent.discount))
    writer.add_text("Replay buffer size", str(agent.memory_size))
    writer.add_text("Hold", str(run_hold))
    writer.add_text("Rewardsystem", str(env.reward_system))


# logging reward, loss and epsilon to tensorboard
def logging():
    writer.add_scalar("Total Reward", total_reward, episode)
    if agent.losses:
        writer.add_scalar("Loss", agent.losses[-1], episode)
    writer.add_scalar("Number of Tetris Clears", env.tetris_amount, episode)
    writer.add_scalar("Epsilon", agent.epsilon_list[-1], episode)


for episode in range(max_episode):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    render_enabled = True
    env.held_shape = None
    env.tetris_amount = 0

    while not done and steps < max_steps:
        # Key controls for the training session
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

        # Render game
        if env.render_enabled:
            env.render(total_reward, framerate=framerate)

        if run_hold:
            next_states = env.merge_next_states()
        else:
            next_states = env.get_next_states(env.shape, env.anchor, False)

        # If the dictionary is empty, meaning the game is over
        if not next_states:
            break

        # Extract values from next_states
        states = list(next_states.values())

        best_state = agent.act(states=states, model=agent.model, use_epsilon=True)

        # Grab the best tetromino position and its rotation chosen by the agent
        best_action = None
        for action, state in next_states.items():
            if (best_state == state).all():
                best_action = action
                break

        reward, done = env.step(best_action)
        total_reward += reward

        # Add to memory for replay
        agent.add_to_memory(current_state, next_states[best_action], reward, done)

        # Set the current new state
        current_state = next_states[best_action]

        steps += 1

    if exit_program:
        break

    # logs data to tensorboard
    if save_log:
        logging()

    # Monitor reward and episodes
    episodes.append(episode)
    rewards.append(total_reward)
    tetris_clear_list.append(env.tetris_amount)

    # Save the model if it achieves a higher total reward than the current maximum
    if total_reward > max_reward and total_reward > highscore:
        print("model_save")
        agent.model_save(path=f"./DQN/models/{model_name}.pt")

    # Check if episode was a highscore
    if total_reward > highscore:
        highscore = total_reward

    # Train model
    agent.replay(episode=episode)

    # Epsilon decay
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay

    # Print training data
    if episode % print_interval == 0:
        print("-" * 30)
        print(f"Running episode {str(episode)}")
        print(f"Epsilon:  {str(agent.epsilon)}")
        print(f"Mean reward:  {str(np.mean(rewards[-print_interval:]))}")
        print(f"Round Highscore: {str(max(rewards[-print_interval:]))}")
        print(f"Training Highscore: {str(highscore)}")
        print(
            f"Round 'retris-clear' highscore:{str(max(tetris_clear_list[-print_interval:]))}"
        )
        print(f"'retris-clear' highscore:{str(max(tetris_clear_list))}")

# Close pygame
pygame.quit()

# Close tensorboard
if save_log:
    writer.close()

print(highscore)
