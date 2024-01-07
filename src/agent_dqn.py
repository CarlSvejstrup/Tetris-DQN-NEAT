import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import sys
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x


class Agent:
    def __init__(self, state_size):
        self.state_size = state_size
        self.losses = []
        self.epsilon_list = []
        self.memory_size = 30000
        self.memory = deque(maxlen=self.memory_size)
        self.discount = 0.98
        self.epsilon = 1.0
        self.epsilon_min = 0.001
        self.epsilon_end_episode = 3000
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min
        ) / self.epsilon_end_episode

        self.batch_size = 500
        self.replay_start = 3000
        self.learning_rate = 0.001

        self.model = QNetwork(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def add_to_memory(self, current_state, next_state, reward, done):
        self.memory.append([current_state, next_state, reward, done])

    def act(self, states):
        max_value = -sys.maxsize - 1
        best = None

        self.epsilon_list.append(self.epsilon)

        if random.random() <= self.epsilon:
            return random.choice(list(states))
        else:
            for state in states:
                state_tensor = torch.tensor(
                    np.reshape(state, (1, self.state_size)), dtype=torch.float32
                )
                value = self.model(state_tensor).item()
                if value > max_value:
                    max_value = value
                    best = state

        return best

    def replay(self):
        print(len(self.memory))
        if len(self.memory) > self.replay_start:
            batch = random.sample(self.memory, self.batch_size)

            next_states = torch.tensor([s[1] for s in batch], dtype=torch.float32)
            # print(f'next_states: {next_states}')
            next_qvalue = self.model(next_states).detach().numpy()
            # print(f'next q_values: {next_qvalue}')

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
