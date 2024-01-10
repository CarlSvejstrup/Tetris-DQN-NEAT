from tetris_engine import Tetris
from agent_dqn import Agent
import time
import pygame
from torch.utils.tensorboard import SummaryWriter


## BUG
# Placement of new piece with hold
## Mixes up game board


## TODO
# Implement hold function
# Mulitprossesing next_states
# Server setup
# Hyperprameters

pygame.init()

# Initialize tetris environment
env = Tetris(10, 20)

# Initialize training variables
max_episode = 100000
max_steps = 250000
max_reward = 500000

print_interval = 10

# Initializing agent
agent = Agent(
    env.state_size,
    memory_size=100000,
    discount=0.99,
    epsilon_min=0.1,
    epsilon_end_episode=3000,
    batch_size=516,
    episodes_per_update=2,
    replay_start=3000,
    learning_rate=0.0001,
)

episodes = []
rewards = []
current_max = 0
highscore = 0

# Creating log writer
log_folder = "run7"
log_dir = "./DQN/training_logs/" + log_folder
writer = SummaryWriter(log_dir=log_dir)


# logging reward, loss and epsilon to tensorboard
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

    while not done and steps < max_steps:
        # Key controls for the training session
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    env.toggle_render()  # Toggle render state with 'r'
                if event.key == pygame.K_q:
                    agent.model_save(path=f"DQN/models/{str(highscore)}.pt")
                    quit()  # quit game with 'q'
        # Render game
        if env.render_enabled:
            env.render(total_reward, framerate=2000)

        # Calcutate next states
        if steps == 0 or env.held_shape == None:
            env.hold_shape()

        next_states = env.get_next_states()
        print(next_states)
        
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

        if best_action[2]:
            print("held action")
            env.shape = env.held_shape
            env.anchor = env.held_anchor
            env.held_shape = None

        reward, done = env.step(best_action)
        total_reward += reward

        # Add to memory for replay
        agent.add_to_memory(current_state, next_states[best_action], reward, done)

        # Set the current new state
        current_state = next_states[best_action]

        steps += 1

    # logs data to tensorboard
    logging()

    # Monitor reward and episodes
    episodes.append(episode)
    rewards.append(total_reward)

    # Check if episode was a highscore
    if total_reward > highscore:
        highscore = total_reward

    # Save the model if it achieves a higher total reward than the current maximum
    if total_reward > max_reward and total_reward > highscore:
        agent.model_save(path=f"DQN/models/{str(highscore)}.pt")

    # Train model
    agent.replay(episode=episode)

    # Epsilon decay
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon -= agent.epsilon_decay

    # Print training data
    if episode % print_interval == 0:
        print("Running episode " + str(episode))
        print("Total reward: " + str(total_reward))

# Close tensorboard
writer.close()

print(highscore)
