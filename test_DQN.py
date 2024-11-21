import os
import random
from collections import deque

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

        s_batch = torch.tensor(s_lst, dtype=torch.float)
        a_batch = torch.tensor(a_lst, dtype=torch.float)
        r_batch = torch.tensor(r_lst, dtype=torch.float)
        s_prime_batch = torch.tensor(s_prime_lst, dtype=torch.float)

        return s_batch, a_batch, r_batch, s_prime_batch

    def size(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, q_lr):
        super(QNetwork, self).__init__()

        self.fc_1 = nn.Linear(state_dim, 64)
        self.fc_2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, action_dim)

        self.lr = q_lr

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x):
        q = F.leaky_relu(self.fc_1(x))
        q = F.leaky_relu(self.fc_2(q))
        q = self.fc_out(q)
        return q


class DQNAgent:
    def __init__(self):
        self.state_dim = 2  # 3
        self.action_dim = 3
        self.lr = 0.01
        self.gamma = 0.98
        self.tau = 0.1  # 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.98
        self.epsilon_min = 0.001
        self.buffer_size = 100000
        self.batch_size = 200
        self.memory = ReplayBuffer(self.buffer_size)

        self.min_output = 100
        self.max_output = 2000

        self.Q = QNetwork(self.state_dim, self.action_dim, self.lr)  # Policy
        self.Q_target = QNetwork(self.state_dim, self.action_dim, self.lr)  # Critic
        self.Q_target.load_state_dict(self.Q.state_dict())

    def choose_action(self, state, explore=False):
        random_number = np.random.rand()
        if explore or self.epsilon > random_number:  # Explore
            action = np.random.choice([n for n in range(self.action_dim)])
            # real_action = action * 100
        else:  # Exploit
            with torch.no_grad():
                action = float(torch.argmax(self.Q(state)).numpy())
                # action = float(action.numpy())
                # real_action = (action -2) * 100
                # maxQ_action_count = 1

        scaled_action = (action - 1) * 100
        print(f"Scaled Action: {scaled_action}")
        return scaled_action

    def calc_target(self, mini_batch):
        s, a, r, s_prime = mini_batch
        with torch.no_grad():
            q_target = self.Q_target(s_prime).max(1)[0].unsqueeze(1)
            target = r + self.gamma * done * q_target
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


if __name__ == '__main__':

    ###### logging ######
    log_name = '0404'

    model_save_dir = 'saved_model/' + log_name
    if not os.path.isdir(model_save_dir): os.mkdir(model_save_dir)
    log_save_dir = 'log/' + log_name
    if not os.path.isdir(log_save_dir): os.mkdir(log_save_dir)
    ###### logging ######

    agent = DQNAgent()

    env = gym.make('Pendulum-v1')

    EPISODE = 50
    score_list = []  # [-2000]

    for EP in range(EPISODE):
        state, _ = env.reset()
        score, done = 0.0, False
        maxQ_action_count = 0

        while True:
            action, real_action, count = agent.choose_action(
                torch.FloatTensor(state))  # [ 0.99982554 -0.01867788  0.00215611]

            state_prime, reward, terminated, truncated, _ = env.step([real_action])

            agent.memory.put((state, action, reward, state_prime))

            score += reward
            maxQ_action_count += count

            state = state_prime

            if agent.memory.size() > 1000:
                agent.train_agent()

            if terminated or truncated:
                break

        if EP % 10 == 0:
            torch.save(agent.Q.state_dict(), model_save_dir + "/DQN_Q_EP" + str(EP) + ".pt")

        print("EP:{}, Avg_Score:{:.1f}, MaxQ_Action_Count:{}, Epsilon:{:.5f}".format(EP, score, maxQ_action_count,
                                                                                     agent.epsilon))
        score_list.append(score)

        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

    plt.plot(score_list)
    plt.show()

    np.savetxt(log_save_dir + '/pendulum_score.txt', score_list)
