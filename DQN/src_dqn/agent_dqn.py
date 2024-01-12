import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from collections import deque

seed = 42

# Set seed for random library
random.seed(seed)

# Set seed for numpy
np.random.seed(seed)

# Set seed for PyTorch
torch.manual_seed(seed)


class QNetwork(nn.Module):
    def __init__(self, state_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(
        self,
        state_size: int,
        seed: int,
        memory_size=100000,
        discount=0.99,
        epsilon_min=0.1,
        epsilon_end_episode=3000,
        batch_size=516,
        episodes_per_update=1,
        replay_start=3000,
        learning_rate=0.0001,
    ):
        self.state_size = state_size
        self.losses = []
        self.epsilon_list = []
        self.memory_size = memory_size
        self.memory = deque(maxlen=self.memory_size)
        self.discount = discount
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_end_episode = epsilon_end_episode
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min
        ) / self.epsilon_end_episode
        self.batch_size = batch_size
        self.episodes_per_update = episodes_per_update
        self.replay_start = replay_start
        self.learning_rate = learning_rate
        self.seed = random.seed(seed)

        self.model = QNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append([current_state, next_state, reward, done])

    def act(self, states, model, use_epsilon=True):
        max_value = -sys.maxsize - 1
        best = None

        self.epsilon_list.append(self.epsilon)

        if use_epsilon and random.random() <= self.epsilon:
            return random.choice(list(states))

        else:
            for state in states:
                state_tensor = torch.tensor(
                    np.reshape(state, (1, self.state_size)), dtype=torch.float32
                )
                value = model(state_tensor).item()
                if value > max_value:
                    max_value = value
                    best = state

        return best

    def replay(self, episode):
        if (
            len(self.memory) > self.replay_start
            and episode % self.episodes_per_update == 0
        ):
            batch = random.sample(self.memory, self.batch_size)

            next_states = torch.tensor(
                np.array([s[1] for s in batch]), dtype=torch.float32
            )

            next_qvalue = self.model(next_states).detach().numpy()

            x = []
            y = []

            for i in range(self.batch_size):
                state, _, reward, done = batch[i][0], None, batch[i][2], batch[i][3]
                if not done:
                    new_q = reward + self.discount * next_qvalue[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(np.array([new_q]))

            x = torch.tensor(np.array(x), dtype=torch.float32)
            y = torch.tensor(
                np.vstack(y), dtype=torch.float32
            )  # Use np.vstack to stack arrays vertically

            self.optimizer.zero_grad()
            output = self.model(x)
            loss = nn.MSELoss()(output, y)
            self.losses.append(loss)
            loss.backward()
            self.optimizer.step()

    def model_save(self, path):
        torch.save(self.model.state_dict(), path)
