import logging
import os
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy.linalg import LinAlgError

import utils
from agent.ES_Registry import ServiceType
from iwai.LGBN_Training_Env import LGBN_Training_Env

logger = logging.getLogger("multiscale")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using {"GPU (CUDA)" if torch.cuda.is_available() else "CPU"} for training")

if not torch.cuda.is_available():
    torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

STATE_DIM = 8
ACTION_DIM_QR = 5
ACTION_DIM_CV = 7


class DQN:
    def __init__(self, state_dim, action_dim, force_restart=False, neurons=16,
                 nn_folder="../share/networks", suffix=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = 0.01
        self.gamma = 0.98
        self.tau = 0.01  # 0.01
        self.epsilon_default = 1.0
        self.epsilon = self.epsilon_default
        self.epsilon_decay = 0.98  # 0.98
        self.epsilon_min = 0.001
        self.buffer_size = 100000
        self.batch_size = 200
        self.memory = ReplayBuffer(self.buffer_size)

        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr, neurons).to(device)  # Q-Network
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr, neurons).to(device)  # Target Network

        # if not force_restart and os.path.exists(nn_folder + f"/Q{"" + suffix if suffix else ""}.pt"):
        #     self.Q.load_state_dict(torch.load(nn_folder + f"/Q{"" + suffix if suffix else ""}.pt", weights_only=True))
        #     logger.info("Loaded existing Q network on startup")

        self.Q_target.load_state_dict(self.Q.state_dict())
        self.nn_folder = nn_folder

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
    def train_dqn_from_env(self, training_env: LGBN_Training_Env, suffix=None):

        # try:
        self.epsilon = self.epsilon_default
        training_env.reset()
            # self.currently_training = True
        # except LinAlgError as e:
        #     logger.warning(f"Could not initialize ENV due to {e.args[0]}, waiting for more samples")
        #     return

        episode_score = 0.0
        score_list = []
        episode_position = 0
        finished_episodes = 0
        EPISODE_LENGTH = 50
        NO_EPISODE = 1000

        # self.epsilon = np.clip(self.epsilon, 0, self.training_length_coeff)
        # print(f"Episodes: {NO_EPISODE} * {self.training_rounds}; epsilon: {self.epsilon}")
        while finished_episodes < NO_EPISODE:

            initial_state = training_env.state
            action = self.choose_action(np.array(training_env.state.for_tensor()))
            next_state, reward, done, _, _ = training_env.step(action)
            print(f"State transition {initial_state}, {action} --> {next_state}")
            print(f"Reward {reward}")

            self.memory.put((initial_state.for_tensor(), action, reward, next_state.for_tensor(), done))
            episode_score += reward

            if self.memory.size() > self.batch_size:
                self.train_batch()

            episode_position += 1

            if episode_position >= EPISODE_LENGTH or done:
                training_env.reset()
                self.epsilon *= self.epsilon_decay
                score_list.append(episode_score)

                episode_score = 0.0
                episode_position = 0
                finished_episodes += 1

        if logger.level <= logging.INFO:
            logger.info(f"Average Score for 5 last rounds: {np.average(score_list[-5:])}")
            plt.plot(score_list)
            plt.show()

        self.store_dqn_as_file(suffix=suffix)
        # self.last_time_trained = datetime.now()
        # self.currently_training = False
        # self.training_length_coeff = np.clip(self.training_length_coeff - 0.2, 0.15, 1.0)

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

    # @utils.print_execution_time
    def store_dqn_as_file(self, suffix=None):
        torch.save(self.Q.state_dict(), self.nn_folder + f"/Q{"_" + suffix if suffix else ""}.pt")


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr, neurons):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, neurons).to(device)
        self.fc_2 = nn.Linear(neurons, neurons).to(device)
        self.fc_out = nn.Linear(neurons, action_dim).to(device)
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
    logging.getLogger("multiscale").setLevel(logging.INFO)
    df_t = pd.read_csv("../share/metrics/metrics.csv")

    # qr_env_t = LGBN_Training_Env(ServiceType.QR, step_quality=100)
    # qr_env_t.reload_lgbn_model(df_t)
    # DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM, force_restart=True).train_dqn_from_env(qr_env_t, "QR")

    cv_env_t = LGBN_Training_Env(ServiceType.CV, step_quality=32)
    cv_env_t.reload_lgbn_model(df_t)
    DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV, force_restart=True).train_dqn_from_env(cv_env_t, "CV")
