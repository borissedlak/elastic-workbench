import logging
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

import utils
from agent.LGBN_Env import LGBN_Env

logger = logging.getLogger("multiscale")
logging.getLogger("multiscale").setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {"GPU (CUDA)" if torch.cuda.is_available() else "CPU"} for training")


# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        s_batch = torch.tensor(s_lst, dtype=torch.float).to(device)
        a_batch = torch.tensor(np.array(a_lst), dtype=torch.float).to(device)
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        d_batch = torch.tensor(done_mask_lst).to(device)

        return s_batch, a_batch, r_batch, s_prime_batch, d_batch

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        # self.fc_1 = nn.Linear(state_dim, 64)
        # self.fc_2 = nn.Linear(64, 32)
        # self.fc_out = nn.Linear(32, action_dim)
        # Interestingly, this is way better for my simple regression task
        self.fc_1 = nn.Linear(state_dim, 8).to(device)
        self.fc_2 = nn.Linear(8, 8).to(device)
        self.fc_out = nn.Linear(8, action_dim).to(device)

        # self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=q_lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQN:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.01
        self.gamma = 0.98
        self.tau = 0.01  # 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.95  # 0.98
        self.epsilon_min = 0.001
        self.buffer_size = 100000
        self.batch_size = 200
        self.memory = ReplayBuffer(self.buffer_size)

        self.min_output = 100
        self.max_output = 2000

        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr).to(device)  # Q-Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr).to(device)  # Target Network
        self.Q_target.load_state_dict(self.Q.state_dict())

        self.training_time = None
        self.env = LGBN_Env()

    def choose_action(self, state):

        s_tensor = torch.FloatTensor(state).to(device)

        if self.epsilon > np.random.rand():  # Explore
            action = np.random.choice([n for n in range(self.action_dim)])
        else:  # Exploit
            with torch.no_grad():
                action = float(torch.argmax(self.Q(s_tensor)).cpu().numpy())  # Must bring tensor to CPU for Numpy

        return action

    def calc_target(self, mini_batch):
        s, a, r, s_prime, d = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * q_target
        return target

    @utils.print_execution_time
    def train_dqn_from_env(self):
        self.env.reload_lgbn_model()
        self.env.reset()
        self.reset_q_networks()  # TODO: Resume training from intermediary point

        score = 0.0
        score_list = []
        round_counter = 0

        while round_counter < 20 * 500:

            initial_state = self.env.state.copy()
            action = self.choose_action(np.array(self.env.state))
            next_state, reward, done, _, _ = self.env.step(action)
            # print(f"State transition {initial_state}, {action} --> {next_state}")

            self.memory.put((initial_state, action, reward, next_state, done))
            score += reward

            if self.memory.size() > self.batch_size:
                self.train_batch()

            round_counter += 1

            if round_counter % 500 == 0:
                self.env.reset()
                self.epsilon *= self.epsilon_decay

                # print("EP:{}, Abs_Score:{:.1f}, Epsilon:{:.3f}".format(round_counter, score, self.epsilon))
                score_list.append(score)
                score = 0.0

        print(f"Average Score for 5 last rounds: {np.average(score_list[-5:])}")
        # TODO: DO this through an animation or interactive plot
        plt.plot(score_list)
        plt.show()

    # @utils.print_execution_time
    def train_batch(self):
        mini_batch = self.memory.sample(self.batch_size)
        # TODO: Check Done and other implementations
        s_batch, a_batch, r_batch, s_prime_batch, d_batch = mini_batch
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

    def reset_q_networks(self):
        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)  # Q-Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)  # Target Network
        self.Q_target.load_state_dict(self.Q.state_dict())
