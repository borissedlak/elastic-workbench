import numpy as np
import copy
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pandas as pd

from agent.daci.daci_agent import SimpleMCDACIAgent


def plot_trajectory_comparison(actual_obs, predicted_obs, type):
    actual_obs = np.array(actual_obs)
    predicted_obs = np.array(predicted_obs)
    steps = np.arange(len(actual_obs))

    plt.figure(figsize=(12, 5))

    # Position
    plt.subplot(1, 2, 1)
    plt.plot(steps, actual_obs[:, 0], label="Actual Position", linewidth=2)
    plt.plot(steps, predicted_obs[:, 0], "--", label="Predicted Position", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Position: Actual vs. Predicted")
    plt.legend()
    plt.grid(True)

    # Velocity
    plt.subplot(1, 2, 2)
    plt.plot(steps, actual_obs[:, 1], label="Actual Velocity", linewidth=2)
    plt.plot(steps, predicted_obs[:, 1], "--", label="Predicted Velocity", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.title("Velocity: Actual vs. Predicted")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"trajectories_efe_plot_{type}_{name}.png")
    plt.close()


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
i = 0
num_steps = 50000
sample_action = True
actions = []


class MCDaciTrainer:
    def __init__(self, boundaries: dict, cv_slo_targets: dict, qr_slo_targets):
        self.agent = SimpleMCDACIAgent(boundaries, cv_slo_targets, qr_slo_targets)

    def train(self):
        epochs = 5000
        max_steps = 200
        # Tracking
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        loss_log = defaultdict(list)
        mcts_stats = []
        efe_stats = pd.DataFrame(
            columns=[
                "episode",
                "step",
                "obs_x",
                "obs_y",
                "action",
                "prob",
            ]
        )
        name = "_efe_no_start_ig_theta"
        precision = 50
        positions = np.linspace(-1.2, 0.6, precision)
        velocities = np.linspace(-0.07, 0.07, precision)
        mesh = np.stack(np.meshgrid(positions, velocities), axis=-1)
        pos_center = (positions.min() + positions.max()) / 2.0
        pos_w = np.abs(positions - pos_center)
        pos_p = pos_w / pos_w.sum()

        vel_center = 0.0
        vel_w = np.abs(velocities - vel_center)
        vel_p = vel_w / vel_w.sum()
        end_training = False

        for epoch in range(epochs):
            # sampling start states, bc w/o this part encoder/decoder see only a tiny part of the env
            # so the overall worl model is bad
            start_position = np.random.choice(range(precision), p=pos_p)
            start_velocity = np.random.choice(range(precision), p=vel_p)
            desired_state = np.array(mesh[start_position][start_velocity], dtype=np.float32)
            # desired_state = np.array([-0.5, 0], dtype=np.float32) #default start state

            state = desired_state
            done = False
            step = 0
            trajectory = []
            total_reward = 0
            while not done and step < max_steps:
                if sample_action and self.agent.train_transition:
                    # if we are in the transition model training phase and want to use EFE
                    # super
                    if len(actions) == 0:
                        actions, stat = self.agent.select_action(state, step, epoch)
                        df_this_call = pd.DataFrame.from_records(
                            stat, columns=["action", "prob"]
                        )
                        df_this_call["episode"] = epoch
                        df_this_call["step"] = step
                        df_this_call["obs_x"] = state[0]
                        df_this_call["obs_y"] = state[1]
                        efe_stats = pd.concat([efe_stats, df_this_call], ignore_index=True)

                    action = [actions.pop(0)]
                else:
                    action = [np.random.choice(range(3))]
                (
                    next_state,
                    reward,
                ) = self.agent.probe_transition(state, None, action[0])
                if step % 25 != 0:
                    # put one experience in the validation set
                    self.agent.save_experience(state, action, next_state)
                else:
                    self.agent.save_experience(state, action, next_state, to_train=False)

                state = copy.deepcopy(next_state)
                total_reward += reward
                step += 1
                if step == max_steps:
                    done = True

            episode_rewards.append(total_reward)
            episode_lengths.append(step)
            print(f"[Episode {epoch}] Reward: {total_reward:.2f}, Steps: {step}")

            loss, dict_loss, end_training = self.agent.fit_experience(epoch, epochs)
            training_losses.append((epoch, loss))
            for k, v in dict_loss.items():
                loss_log[k].append((epoch, v))

            if epoch % 10 == 0:
                self.agent.validate(epoch)

            if end_training or epoch == (epochs - 10):
                start_state = np.array(
                    [0.4, 0.000], dtype=np.float32
                )  # np.array([-0.5, 0.000], dtype=np.float32)
                traject = {  # TODO: traject for ES env
                    "left": [0, 0, 0, 0, 0],
                    "right": [2, 2, 2, 2, 2],
                    "stay": [1, 1, 1, 1, 1],
                    "mix": [2, 1, 1, 2, 0],
                }
                for k, v in traject.items():
                    actual_obs, reconstr_obs, _ = self.agent.pretend_a_trajectory(start_state, v)
                    plot_trajectory_comparison(
                        actual_obs, reconstr_obs, type="encoder" + "_" + k
                    )

                    actual_obs, reconstr_obs, next_obs = agent.pretend_a_trajectory(
                        start_state, v, use_transition=True
                    )
                    plot_trajectory_comparison(
                        actual_obs, next_obs, type="transition" + "_" + k
                    )
            if end_training:
                break


import matplotlib.pyplot as plt

efe_stats.to_csv(f"./efe_logs_{name}.csv", sep="|")
# Plot episode reward
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Episode Reward Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")

# Plot training loss
if training_losses:
    episodes, losses = zip(*training_losses)
    plt.subplot(1, 2, 2)
    plt.plot(episodes, losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Loss")

plt.tight_layout()
plt.savefig(f"reward_and_loss_6{name}.png")
plt.close()

plt.figure(figsize=(12, 6))
n = len(loss_log)
for i, (k, values) in enumerate(loss_log.items()):
    episodes, vals = zip(*values)
    plt.plot(episodes, vals, label=k)

plt.title("Per-Component Loss Over Time")
plt.xlabel("Episode")
plt.ylabel("Loss Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"loss_components_6_{name}.png")
plt.close()

log_df = pd.DataFrame(
    {
        "episode": list(range(episode + 1)),
        "reward": episode_rewards,
        "steps": episode_lengths,
    }
)

if training_losses:
    _, losses = zip(*training_losses)
    log_df["total_loss"] = losses

for k, values in loss_log.items():
    episodes_k, vals = zip(*values)
    loss_series = pd.Series(data=vals, index=episodes_k)
    log_df[k] = loss_series

# Save to CSV
log_df.to_csv(f"training_log_{name}.csv", index=False)
print(f"📄 Saved training log to mcts_training_log_{name}.csv")

agent.buffer = []
agent.val_buffer = []
torch.save({"agent": agent}, f"aif_agent_checkpoint_{name}.pth")
