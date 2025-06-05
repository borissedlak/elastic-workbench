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

import utils
from agent.es_registry import ServiceType
from iwai.lgbn_training_env import LGBNTrainingEnv
from proj_types import ESServiceAction

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("multiscale")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(
    f"Using {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'''} for training"
)

if not torch.cuda.is_available():
    torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

STATE_DIM = 8
ACTION_DIM_QR = 5
ACTION_DIM_CV = 7

CV_DATA_QUALITY_STEP = 32
QR_DATA_QUALITY_STEP = 100

EPISODE_LENGTH = 25
NO_EPISODES = 250


class DQN:
    def __init__(
        self, state_dim, action_dim, neurons=16, nn_folder=ROOT + "/../share/networks"
    ):
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

        self.q_network = QNetwork(self.state_dim, self.action_dim, self.lr, neurons).to(
            device
        )
        self.q_target_network = QNetwork(
            self.state_dim, self.action_dim, self.lr, neurons
        ).to(device)

        self.q_target_network.load_state_dict(self.q_network.state_dict())
        self.nn_folder = nn_folder

    @torch.no_grad()  # We don't want to store gradient updates here at inference
    def choose_action(self, state: np.ndarray, rand=None) -> int:
        s_tensor = torch.FloatTensor(state).to(device)

        if rand is None:
            rand = self.epsilon
        if rand > np.random.rand():  # Explore
            action = np.random.choice([n for n in range(self.action_dim)])
        else:  # Exploit
            action = (
                torch.argmax(self.q_network(s_tensor)).cpu().numpy()
            )  # Must bring tensor to CPU for Numpy

        return action

    def calc_target(self, mini_batch):
        s, a, r, s_prime, d = mini_batch
        with torch.no_grad():
            q_target = self.q_target_network(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * q_target
        return target

    @utils.print_execution_time
    def train_single_dqn_from_env(self, training_env: LGBNTrainingEnv, suffix=None):

        self.epsilon = self.epsilon_default
        training_env.reset()

        episode_score = 0.0
        score_list = []
        episode_position = 0
        finished_episodes = 0

        # self.epsilon = np.clip(self.epsilon, 0, self.training_length_coeff)
        # print(f"Episodes: {NO_EPISODE} * {self.training_rounds}; epsilon: {self.epsilon}")
        while finished_episodes < NO_EPISODES:

            initial_state = training_env.state
            action = self.choose_action(training_env.state.to_np_ndarray(True))
            next_state, reward, done, _, _ = training_env.step(ESServiceAction(action))
            # print(f"State transition {initial_state}, {action} --> {next_state}")
            # print(f"Reward {reward}")

            self.memory.put(
                (
                    initial_state.to_np_ndarray(True),
                    action,
                    reward,
                    next_state.to_np_ndarray(True),
                    done,
                )
            )
            episode_score += reward

            if self.memory.size() > self.batch_size:
                self.train_batch()

            episode_position += 1

            if episode_position >= EPISODE_LENGTH or done:
                logger.info(f"Final state for Env: {training_env.state}")
                logger.info(
                    f"[EP {finished_episodes + 1:3d}] Score: {episode_score:.2f} | Epsilon: {self.epsilon:.4f}"
                )

                training_env.reset()
                self.epsilon *= self.epsilon_decay
                score_list.append(episode_score)

                episode_score = 0.0
                episode_position = 0
                finished_episodes += 1

        if logger.level <= logging.INFO:
            logger.info(
                f"Average Score for 5 last rounds: {np.average(score_list[-5:])}"
            )
            plt.plot(score_list)
            plt.show()

        self.store_dqn_as_file(suffix=suffix)

    # @utils.print_execution_time
    def train_batch(self):
        mini_batch = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, state_prime_batch, done_batch = mini_batch
        action_batch = action_batch.type(torch.int64)

        td_target = self.calc_target(mini_batch)

        # Q train
        q_action = self.q_network(state_batch).gather(1, action_batch)
        q_loss = F.smooth_l1_loss(q_action, td_target)
        self.q_network.optimizer.zero_grad()
        q_loss.mean().backward()
        self.q_network.optimizer.step()

        # Q soft-update
        for param_target, param in zip(
            self.q_target_network.parameters(), self.q_network.parameters()
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.tau) + param.data * self.tau
            )

    # @utils.print_execution_time
    def store_dqn_as_file(self, suffix=None):
        file_name = self.nn_folder + f"/Q{'_' + suffix if suffix else ''}.pt"
        logger.info(f"Save DQN as {file_name}")
        torch.save(self.q_network.state_dict(), file_name)

    def load(self, file_name):
        self.q_network.load_state_dict(
            torch.load(self.nn_folder + "/" + file_name, weights_only=True)
        )


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, actions_dim: int, q_lr: float, latent_dim: int):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, latent_dim).to(device)
        self.fc_2 = nn.Linear(latent_dim, latent_dim).to(device)
        self.fc_out = nn.Linear(latent_dim, actions_dim).to(device)
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

        s_batch = torch.tensor(np.array(s_lst), dtype=torch.float).to(device)
        a_batch = torch.tensor(np.array(a_lst), dtype=torch.float).to(device)
        r_batch = torch.tensor(np.array(r_lst), dtype=torch.float).to(device)
        s_prime_batch = torch.tensor(np.array(s_prime_lst), dtype=torch.float).to(device)
        d_batch = torch.tensor(done_mask_lst).to(device)

        return s_batch, a_batch, r_batch, s_prime_batch, d_batch

    def size(self):
        return len(self.buffer)


if __name__ == "__main__":
    logging.getLogger("multiscale").setLevel(logging.INFO)
    df_t = pd.read_csv(ROOT + "/../share/metrics/LGBN.csv")

    qr_env_t = LGBNTrainingEnv(ServiceType.QR, step_data_quality=QR_DATA_QUALITY_STEP)
    qr_env_t.reload_lgbn_model(df_t)
    DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_QR).train_single_dqn_from_env(qr_env_t, "QR")

    # cv_env_t = LGBNTrainingEnv(ServiceType.CV, step_data_quality=CV_DATA_QUALITY_STEP)
    # cv_env_t.reload_lgbn_model(df_t)
    # DQN(state_dim=STATE_DIM, action_dim=ACTION_DIM_CV).train_single_dqn_from_env(cv_env_t, "CV")
