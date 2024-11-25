import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Check if GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available.")


# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)

        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float)
        a_batch = torch.tensor(np.array(a_lst), dtype=torch.float)
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float)

        return s_batch, a_batch, r_batch, s_prime_batch

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        # self.fc_1 = nn.Linear(state_dim, 64)
        # self.fc_2 = nn.Linear(64, 32)
        # self.fc_out = nn.Linear(32, action_dim)
        # Interestingly, this is way better for my simple regression task
        self.fc_1 = nn.Linear(state_dim, 8)
        self.fc_2 = nn.Linear(8, 8)
        self.fc_out = nn.Linear(8, action_dim)

        # self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=q_lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    def __init__(self, state_dim=3, action_dim=9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.01
        self.gamma = 0.98
        self.tau = 0.01  # 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.94  # 0.98
        self.epsilon_min = 0.001
        self.buffer_size = 100000
        self.batch_size = 200
        self.memory = ReplayBuffer(self.buffer_size)

        self.min_output = 100
        self.max_output = 2000

        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)  # Q-Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)  # Target Network
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state, explore=False):
        random_number = np.random.rand()
        if explore or self.epsilon > random_number:  # Explore
            action = np.random.choice([n for n in range(self.action_dim)])
        else:  # Exploit
            with torch.no_grad():
                action = float(torch.argmax(self.Q(state)).numpy())

        return action

    def calc_target(self, mini_batch):
        s, a, r, s_prime = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * q_target
        return target

    # @utils.print_execution_time
    def train_agent(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch = mini_batch
        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        #### Q train ####
        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        self.Q.optimizer.step()
        #### Q train ####

        #### Q soft-update ####
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
