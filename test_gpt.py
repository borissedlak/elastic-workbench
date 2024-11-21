from random import randint

import keras
import numpy as np
import tensorflow as tf
from keras import layers

# num_states = env.observation_space.shape[0]
num_states = 2
print("Size of State Space ->  {}".format(num_states))
# num_actions = env.action_space.shape[0]
num_actions = 1
print("Size of Action Space ->  {}".format(num_actions))

# upper_bound = env.action_space.high[0]
# lower_bound = env.action_space.low[0]
lower_bound = 100.0
upper_bound = 2000.0

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # self.reset()  # TODO: Remove??
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process
        x = (
                self.x_prev
                + self.theta * (self.mean - self.x_prev) * self.dt
                + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') observation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
            self,
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""


def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    # last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(num_states,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    # outputs = layers.Dense(1, activation="relu", kernel_initializer=last_init)(out) # TODO: Correct?
    outputs = layers.Dense(1, activation="sigmoid")(out)  # TODO: Correct?

    # Our upper bound is 2.0 for Pendulum.
    # outputs = outputs * upper_bound
    # outputs = ((outputs + 1.0) / 2 * (upper_bound - lower_bound)) + lower_bound
    outputs = (outputs * (upper_bound - lower_bound)) + lower_bound
    model = keras.Model(inputs, outputs)
    return model


def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(num_actions,))
    action_out = layers.Dense(32, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(s, noise_object):
    sampled_actions = keras.ops.squeeze(actor_model(s))
    noise = noise_object()
    # Adding noise to action
    sampled_actions_noisy = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions_noisy, lower_bound, upper_bound)
    # print(sampled_actions.numpy(), sampled_actions_noisy, lower_bound, upper_bound, legal_action,
    #       [np.squeeze(legal_action)])

    return [np.squeeze(legal_action)]


"""
## Training hyperparameters
"""

# TODO: Must be relative to value range
std_dev = 250  # 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = keras.optimizers.Adam(critic_lr)
actor_optimizer = keras.optimizers.Adam(actor_lr)

total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.25  # 0.005

buffer = Buffer(50000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


# prev_state_g = None
# action_g = None


def get_action(prev_state, random=False):
    if random:
        # ou_noise.reset()

        # STill doing this
        # tf_prev_state = keras.ops.expand_dims(keras.ops.convert_to_tensor(prev_state), 0)
        # action = policy(tf_prev_state, ou_noise)

        return [randint(int(lower_bound), int(upper_bound))]

    tf_prev_state = keras.ops.expand_dims(keras.ops.convert_to_tensor(prev_state), 0)
    action = policy(tf_prev_state, ou_noise)

    if np.isnan(action[0]):
        pass

    return [int(a) for a in action]


def evaluate_result(prev_state, action, reward, new_state):
    # global prev_state_g, action_g

    buffer.record((prev_state, action, reward, new_state))
    buffer.learn()

    update_target(target_actor, actor_model, tau)
    update_target(target_critic, critic_model, tau)


# Takes about 4 min to train
# for ep in range(total_episodes):
#     prev_state, _ = env.reset()  # [0.99757475 0.0696037  0.10989507]
#     episodic_reward = 0
#
#     while True:
#         tf_prev_state = keras.ops.expand_dims(
#             keras.ops.convert_to_tensor(prev_state), 0
#         )
#
#         action = policy(tf_prev_state, ou_noise)
#         # Receive state and reward from environment.
#         state, reward, done, truncated, _ = env.step(
#             action)  # ([0.9960107  0.08923335 0.22799788], -0.008827652903753114, False, False, {})
#
#         buffer.record((prev_state, action, reward, state))
#         episodic_reward += reward
#
#         buffer.learn()
#
#         update_target(target_actor, actor_model, tau)
#         update_target(target_critic, critic_model, tau)
#
#         # End this episode when `done` or `truncated` is True
#         if done or truncated:
#             break
#
#         prev_state = state
#
#     ep_reward_list.append(episodic_reward)
#
#     # Mean of last 40 episodes
#     avg_reward = np.mean(ep_reward_list[-40:])
#     print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
#     avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Episodic Reward")
# plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLander-v2` continuous, but it will take
more episodes to obtain good results.
"""

# Save the weights
# actor_model.save_weights("pendulum_actor.weights.h5")
# critic_model.save_weights("pendulum_critic.weights.h5")
#
# target_actor.save_weights("pendulum_target_actor.weights.h5")
# target_critic.save_weights("pendulum_target_critic.weights.h5")

"""
Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)
"""
