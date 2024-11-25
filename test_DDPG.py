import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ScalingEnv import ScalingEnv

# Hyperparameters
BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
LR_ACTOR = 1e-3
LR_CRITIC = 1e-3
# NOISE_STD = 0.1
NOISE_DECAY = 0.99


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.float32),
            torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)


# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.max_action = max_action
        self.buffer = ReplayBuffer(BUFFER_SIZE)

    def select_action(self, state, std, noise=0.0):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action += noise * np.random.normal(0, std, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)

    def train(self):
        if self.buffer.size() < BATCH_SIZE:
            return

        # Sample a batch
        states, actions, rewards, next_states = self.buffer.sample(BATCH_SIZE)

        # Critic Loss
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + GAMMA * target_q
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Loss
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)


# Training Loop
# env = gym.make('Pendulum-v1')
env = ScalingEnv()

state_dim = env.observation_space.shape[0]  # = 3
action_dim = env.action_space.shape[0]  # = 1
max_action = env.action_space.high[0]  # = 2.0

agent = DDPGAgent(state_dim, action_dim, max_action)

episodes = 200
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    noise = max_action * (NOISE_DECAY ** episode)

    for t in range(200):
        action = agent.select_action(state, noise, max_action)
        next_state, reward, done, _, _ = env.step(action)
        # print(action, reward)
        agent.buffer.add(state, action, reward, next_state)

        agent.train()
        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode + 1}, Reward: {episode_reward:.2f}, Noise: {NOISE_DECAY ** episode:.2f}")

env.close()
