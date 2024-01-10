from tetris_engine import Tetris
from agent_dqn import Agent
import time
import pygame
from torch.utils.tensorboard import SummaryWriter


## TODO
# Pull training data to CSV for visualization and debug high loss/ high reward problem
#
pygame.init()

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 4000
max_steps = 25000

agent = Agent(env.state_size)

episodes = []
rewards = []

current_max = 0

log_folder = "run3"
log_dir = "training_logs/" + log_folder
writer = SummaryWriter(log_dir=log_dir)


def logging():
    writer.add_scalar("Total Reward", total_reward, episode)
    writer.add_scalar("Epsilon", agent.epsilon_list[-1], episode)

    if agent.losses:
        writer.add_scalar("Loss", agent.losses[-1], episode)
    # Log metrics to TensorBoard


writer.add_text("Max episode", str(max_episode))
writer.add_text("Learningrate", str(agent.learning_rate))
writer.add_text("Replaystart", str(agent.replay_start))
writer.add_text("Batchsize", str(agent.batch_size))
writer.add_text("Discount value", str(agent.discount))
writer.add_text("Replay buffer size", str(agent.memory_size))

for episode in range(max_episode):
    current_state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    render_enabled = True
    print("Running episode " + str(episode))

    while not done and steps < max_steps:
        # Key controls for the training session
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.toggle_render()  # Toggle render state with 'r'
                if event.key == pygame.K_q:
                    quit()  # quit game with 'q'

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

        logging()

    episodes.append(episode)
    rewards.append(total_reward)

    agent.replay()

    # if total_reward < 25000:
    #     agent.replay()
    # else:
    #     pass

    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay

    """print("Total reward: " + str(total_reward))

    if agent.losses:
        print(f"loss: {agent.losses[-1]}")
    print(f"epsilon: {agent.epsilon_list[-1]}")"""

writer.close()
