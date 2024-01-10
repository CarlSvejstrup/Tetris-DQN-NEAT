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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,padding=2)
        self.drop1 = nn.Dropout(p=0.1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.drop2 = nn.Dropout(p=0.1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        #self.maxp1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=1600, out_features=256)
        self.relu6 = nn.ReLU()
        self.drop4 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.relu7 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=128, out_features=1)
        self.relu8 = nn.ReLU()



    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.drop1(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.drop2(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.drop3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.drop4(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.fc3(x)
        x = self.relu8(x)
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
        self.epsilon_min = 0.1
        self.epsilon_end_episode = 3000
        self.epsilon_decay_rate = 0.995
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min
        ) / self.epsilon_end_episode

        self.batch_size = 500
        self.replay_start = 3000
        self.learning_rate = 0.005

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
            prev_state = None            
            for i in states:
                if prev_state is not None and i.all() != prev_state.all():
                    print("test")
                prev_state = i

            for state in states:
                state_tensor = torch.tensor(
                    np.reshape(state, (1, 1, 20, 10)), dtype=torch.float32
                )
                value = self.model(state_tensor).item()
                if value > max_value:
                    #print("Max:", value)
                    max_value = value
                    best = state

        return best

    def replay(self):
        #print(len(self.memory))
        if len(self.memory) > self.replay_start:
            batch = random.sample(self.memory, self.batch_size)

            next_states = torch.tensor([[s[1]] for s in batch], dtype=torch.float32)
            # print(f'next_states: {next_states}')
            next_qvalue = self.model(next_states).detach().numpy()
            # print(f'next q_values: {next_qvalue}')

            x = []
            y = []

            for i in range(self.batch_size):
                state, _, reward, done = [batch[i][0]], None, batch[i][2], batch[i][3]
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
            print(sum([abs(output_) for output_, y_ in zip(output, y)]), self.epsilon)
            loss = nn.MSELoss()(output, y)
            self.losses.append(loss)
            loss.backward()
            self.optimizer.step()
