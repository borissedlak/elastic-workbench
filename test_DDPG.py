import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the Actor Network (Policy)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action_mean = self.fc3(x)
        return action_mean


# Define the Critic Network (Value Function)
class Critic(nn.Module):
    def __init__(self, state_dim, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


# Define the PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, epsilon=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.epsilon = epsilon

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        action_mean = self.actor(state)
        # In continuous action spaces, the output is the mean of the action distribution (e.g., Gaussian)
        action = action_mean + torch.randn_like(action_mean)  # Assuming no action noise for simplicity
        return action, action_mean

    def get_value(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        return self.critic(state)

    def update(self, states, actions, rewards, next_states, done, old_action_means, old_log_probs):
        # Ensure actions are stacked properly
        actions = torch.stack([torch.tensor(action, dtype=torch.float32) for action in actions])

        # Convert old_log_probs to a tensor
        old_log_probs = torch.stack(old_log_probs)  # If it's a list of tensors, this will stack them

        states = torch.tensor(states, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Calculate returns (bootstrap the final return with the value of the last state)
        target_values = self.critic(next_states)
        returns = rewards + self.gamma * target_values * (1 - done)

        # Compute advantage
        values = self.critic(states)
        advantages = returns - values.detach()

        # Compute the log probability of the current actions
        action_means = self.actor(states)
        dist = torch.distributions.Normal(action_means, torch.ones_like(action_means))  # Gaussian distribution
        log_probs = dist.log_prob(actions).sum(dim=-1)

        # Calculate the ratio (pi_theta / pi_theta_old)
        ratios = torch.exp(log_probs - old_log_probs)

        # Surrogate loss function
        surrogate_loss = torch.min(ratios * advantages,
                                   torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages)

        # Actor loss
        actor_loss = -surrogate_loss.mean()

        # Critic loss (mean squared error between value predictions and target values)
        critic_loss = nn.MSELoss()(values, returns)

        # Update the actor and critic networks
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


# Sample usage of the PPOAgent class
if __name__ == "__main__":
    state_dim = 2
    action_dim = 2
    agent = PPOAgent(state_dim, action_dim)

    # Example loop (simplified)
    for episode in range(100):
        state = np.random.rand(state_dim)  # Random initial state
        done = False
        episode_reward = 0

        states, actions, rewards, next_states, dones, old_action_means, old_log_probs = [], [], [], [], [], [], []

        while not done:
            action, action_mean = agent.get_action(state)
            value = agent.get_value(state)
            next_state = np.random.rand(state_dim)  # Dummy next state
            reward = np.random.rand()  # Dummy reward
            done = np.random.rand() > 0.95  # Random done condition

            # Save transition for later updates
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            old_action_means.append(action_mean)
            old_log_probs.append(
                torch.distributions.Normal(action_mean, torch.ones_like(action_mean)).log_prob(action).sum(dim=-1))

            state = next_state
            episode_reward += reward

        # After episode ends, perform the PPO update
        agent.update(states, actions, rewards, next_states, dones, old_action_means, old_log_probs)
        print(f"Episode {episode}, Reward: {episode_reward}")
