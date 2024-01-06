from tetris_engine import Tetris
from agent_dqn import Agent
import time
import matplotlib.pyplot as plt
import keyboard
import pygame
import csv
import os


## TODO
# Pull training data to CSV for visualization and debug high loss/ high reward problem
#


pygame.init()

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 100
max_steps = 25000

agent = Agent(env.state_size)

episodes = []
rewards = []

current_max = 0

for episode in range(max_episode):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    render_enabled = True
    print("Running episode " + str(episode))

    while not done and steps < max_steps:
        # Rendering game. press r to toggle render 'on' and 'off'
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.toggle_render()  # Toggle render state

        if env.render_enabled:
            env.render(total_reward)

        next_states = env.get_next_states()

        # If the dictionary is empty, meaning the game is over
        if not next_states:
            break

        # Tell the agent to choose the best possible state
        best_state = agent.act(list(next_states.values()))

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

    print("Total reward: " + str(total_reward))

    if agent.losses:
        print(f"loss: {agent.losses[-1]}")
    print(f"epsilon: {agent.epsilon_list[-1]}")

    episodes.append(episode)
    rewards.append(total_reward)

    agent.replay()

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay


data = list(zip(episodes, rewards, agent.epsilon_list, agent.losses))

csv_file_path = os.path.join(
    os.path.dirname(__file__), "..", "training_stats", "training_1.csv"
)

with open(csv_file_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "reward", "epsilon", "loss"])
    csv_writer.writerow(data)

plt.plot(episodes, rewards)
plt.title("Learning Curve")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
