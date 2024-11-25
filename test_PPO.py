import numpy as np
from stable_baselines3 import PPO, A2C

from ScalingEnv import ScalingEnv

# Create the environment
env = ScalingEnv()

# Create the model using PPO and the multi-discrete action space
model = A2C("MlpPolicy", env, verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")

for _ in range(10):
    model.learn(total_timesteps=500)

# # Train the model
# model_dumb.learn(total_timesteps=100)

# Save the model
model.save("ppo_multidiscrete_model")
model = PPO.load("ppo_multidiscrete_model")

# Test the model after training
state, _ = env.reset()

for _ in range(10):
    action, state_p = model.predict(np.array([0, 20]))
    new_state, reward, done, info, _ = env.step(action)
    print(f"State: {state}, Action: {action}, Reward: {reward}")
    state = new_state
