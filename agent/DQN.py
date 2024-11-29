import logging
import os
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError

import utils
from agent.LGBN_Env import LGBN_Env

logger = logging.getLogger("multiscale")
logging.getLogger("multiscale").setLevel(logging.INFO)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {"GPU (CUDA)" if torch.cuda.is_available() else "CPU"} for training")

if not torch.cuda.is_available():
    torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

NN_FOLDER = "../share/networks"


class DQN:
    def __init__(self, state_dim, action_dim, force_restart=False):
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
        self.training_rounds = 1.0

        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr).to(device)  # Q-Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr).to(device)  # Target Network

        if not force_restart and os.path.exists(NN_FOLDER + "/Q.pt"):
            self.Q.load_state_dict(torch.load(NN_FOLDER + "/Q.pt", weights_only=True))
            # self.Q_target.load_state_dict(torch.load(NN_FOLDER + "/Q_target.pt", weights_only=True))
            self.training_rounds = 0.5
            logger.info("Loaded existing Q network on startup")

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.last_time_trained = datetime(1970, 1, 1, 0, 0, 0)
        self.currently_training = False
        self.env = LGBN_Env()

    @torch.no_grad()  # We don't want to store gradient updates here at inference
    def choose_action(self, state: np.ndarray, rand=None):
        s_tensor = torch.FloatTensor(state).to(device)

        if rand is None:
            rand = self.epsilon

        if rand > np.random.rand():  # Explore
            action = np.random.choice([n for n in range(self.action_dim)])
        else:  # Exploit
            action = torch.argmax(self.Q(s_tensor)).cpu().numpy()  # Must bring tensor to CPU for Numpy

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
        try:
            self.env.reset()
            self.currently_training = True
        except LinAlgError as e:
            logger.warning(f"Could not initialize ENV due to {e.args[0]}, waiting for more samples")
            return

        episode_score = 0.0
        score_list = []
        round_counter = 0
        EPISODE_LENGTH = 100
        NO_EPISODE = 70

        self.epsilon = np.clip(self.epsilon, 0, self.training_rounds)
        # print(f"Episodes: {NO_EPISODE} * {self.training_rounds}; epsilon: {self.epsilon}")
        while round_counter < (NO_EPISODE * self.training_rounds) * EPISODE_LENGTH:

            initial_state = self.env.state
            action = self.choose_action(np.array(self.env.state.for_tensor()))
            next_state, reward, done, _, _ = self.env.step(action)
            # print(f"State transition {initial_state}, {action} --> {next_state}")
            # print(f"Reward {reward}")

            self.memory.put((initial_state.for_tensor(), action, reward, next_state.for_tensor(), done))
            episode_score += reward

            if self.memory.size() > self.batch_size:
                self.train_batch()

            round_counter += 1

            if round_counter % EPISODE_LENGTH == 0:
                self.env.reset()
                self.epsilon *= self.epsilon_decay
                score_list.append(episode_score)
                episode_score = 0.0

        if logger.level <= logging.INFO:
            logger.info(f"Average Score for 5 last rounds: {np.average(score_list[-5:])}")
            plt.plot(score_list)
            plt.show()

        self.store_dqn_as_file()
        self.last_time_trained = datetime.now()
        self.currently_training = False
        self.training_rounds = np.clip(self.training_rounds - 0.2, 0.15, 1.0)
        self.epsilon = 1.0

    # @utils.print_execution_time
    def train_batch(self):
        mini_batch = self.memory.sample(self.batch_size)
        s_batch, a_batch, r_batch, s_prime_batch, d_batch = mini_batch
        a_batch = a_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        #### Q train ####
        Q_a = self.Q(s_batch).gather(1, a_batch)
        q_loss = F.smooth_l1_loss(Q_a, td_target)
        self.Q.optimizer.zero_grad()
        q_loss.mean().backward()
        self.Q.optimizer.step()

        #### Q soft-update ####
        for param_target, param in zip(self.Q_target.parameters(), self.Q.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    # def reset_q_networks(self):
    #     self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)  # Q-Network
    #     self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)  # Target Network
    #     self.Q_target.load_state_dict(self.Q.state_dict())

    # @utils.print_execution_time
    def store_dqn_as_file(self):
        torch.save(self.Q.state_dict(), NN_FOLDER + "/Q.pt")
        # torch.save(self.Q_target.state_dict(), NN_FOLDER + "/Q_target.pt")


NO_NEURONS = 16
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        # TODO: Find optimal number of neurons
        self.fc_1 = nn.Linear(state_dim, NO_NEURONS).to(device)
        self.fc_2 = nn.Linear(NO_NEURONS, NO_NEURONS).to(device)
        self.fc_out = nn.Linear(NO_NEURONS, action_dim).to(device)

        # TODO: Read more about Adam
        self.optimizer = optim.Adam(self.parameters(), lr=q_lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


# ReplayBuffer from https://github.com/seungeunrho/minimalRL
class ReplayBuffer:
    def __init__(self, buffer_limit: int):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n: int):
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


if __name__ == '__main__':

    dqn = DQN(state_dim=7, action_dim=5, force_restart=True)
    dqn.train_dqn_from_env()
    dqn.train_dqn_from_env()
    dqn.train_dqn_from_env()
