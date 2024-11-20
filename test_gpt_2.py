import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers


# Environment simulation
class PixelEnv:
    def __init__(self):
        self.state_dim = (64, 64, 1)  # Grayscale image size
        self.fps = 30  # Initial FPS
        self.max_fps = 60

    def reset(self):
        # Random grayscale image and FPS
        self.state = np.random.random(self.state_dim)
        self.fps = 30
        return self.state, self.fps

    def step(self, action):
        # Apply action (pixel adjustment)
        adjusted_state = np.clip(self.state + action, 0, 1)
        fps_change = np.random.uniform(-1, 1) + np.mean(action) * 10
        self.fps = np.clip(self.fps + fps_change, 0, self.max_fps)

        # Reward: FPS improvement minus adjustment penalty
        reward = (self.fps - 30) - 0.1 * np.sum(np.abs(action))

        # New state and FPS
        self.state = adjusted_state
        done = self.fps >= self.max_fps or self.fps <= 0  # Episode ends if out of bounds
        return adjusted_state, self.fps, reward, done


# Actor Network
def build_actor(state_dim, action_dim):
    inputs = layers.Input(shape=state_dim + (1,), name="state_fps")
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(action_dim, activation="tanh", name="actions")(x)
    return Model(inputs, outputs, name="Actor")


# Critic Network
def build_critic(state_dim, action_dim):
    state_input = layers.Input(shape=state_dim + (1,), name="state_fps")
    action_input = layers.Input(shape=(action_dim,), name="actions")
    x = layers.Conv2D(32, (3, 3), activation="relu")(state_input)
    x = layers.Flatten()(x)
    x = layers.Concatenate()([x, action_input])
    x = layers.Dense(64, activation="relu")(x)
    q_value = layers.Dense(1, name="q_value")(x)
    return Model([state_input, action_input], q_value, name="Critic")


# Hyperparameters
state_dim = (64, 64, 1)
action_dim = 1  # Single adjustment factor
gamma = 0.99  # Discount factor
tau = 0.005  # Target network update rate
lr_actor = 1e-4
lr_critic = 1e-3

# Initialize networks
actor = build_actor(state_dim, action_dim)
critic = build_critic(state_dim, action_dim)
target_actor = build_actor(state_dim, action_dim)
target_critic = build_critic(state_dim, action_dim)
target_actor.set_weights(actor.get_weights())
target_critic.set_weights(critic.get_weights())

# Optimizers
actor_optimizer = optimizers.Adam(learning_rate=lr_actor)
critic_optimizer = optimizers.Adam(learning_rate=lr_critic)


# Replay buffer
class ReplayBuffer:
    def __init__(self, buffer_size=10000):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in indices]


buffer = ReplayBuffer()


# Update target networks
def update_target_networks(target_model, model, tau):
    target_weights = target_model.get_weights()
    model_weights = model.get_weights()
    new_weights = [
        tau * mw + (1 - tau) * tw
        for mw, tw in zip(model_weights, target_weights)
    ]
    target_model.set_weights(new_weights)


# Training loop
env = PixelEnv()
episodes = 500
batch_size = 64

for episode in range(episodes):
    state, fps = env.reset()
    state_fps = np.expand_dims(np.concatenate([state, np.full(state.shape[:2] + (1,), fps)], axis=-1), axis=0)
    episode_reward = 0

    for step in range(100):  # Max steps per episode
        action = actor.predict(state_fps)[0]
        next_state, next_fps, reward, done = env.step(action)

        # Add to buffer
        next_state_fps = np.expand_dims(
            np.concatenate([next_state, np.full(next_state.shape[:2] + (1,), next_fps)], axis=-1), axis=0)
        buffer.add((state_fps, action, reward, next_state_fps, done))

        # Train
        if len(buffer.buffer) >= batch_size:
            samples = buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*samples)
            states = np.concatenate(states)
            actions = np.array(actions)
            rewards = np.array(rewards).reshape(-1, 1)
            next_states = np.concatenate(next_states)
            dones = np.array(dones).reshape(-1, 1)

            # Train critic
            target_qs = rewards + gamma * (1 - dones) * target_critic.predict(
                [next_states, target_actor.predict(next_states)])
            critic_loss = critic.train_on_batch([states, actions], target_qs)

            # Train actor
            with tf.GradientTape() as tape:
                actions_pred = actor(states)
                actor_loss = -tf.reduce_mean(critic([states, actions_pred]))
            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables))

            # Update target networks
            update_target_networks(target_actor, actor, tau)
            update_target_networks(target_critic, critic, tau)

        # Update state
        state_fps = next_state_fps
        episode_reward += reward
        if done:
            break

    print(f"Episode {episode + 1}/{episodes}: Reward = {episode_reward:.2f}")
