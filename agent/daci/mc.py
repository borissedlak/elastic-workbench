import itertools
import os

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
actions_qr = []
ctions_cv = []


class MCDaciTrainer:
    def __init__(
        self,
        boundaries: dict,
        cv_slo_targets: dict,
        qr_slo_targets,
        action_dim_cv: int = 7,
        action_dim_qr: int = 5,
        # device: str = "cuda:0",
        device: str = "cuda:0",
    ):
        self.boundaries = boundaries
        self.qr_slo_targets = qr_slo_targets
        self.cv_slo_targets = cv_slo_targets
        self.agent = SimpleMCDACIAgent(boundaries, cv_slo_targets, qr_slo_targets, action_dim_cv, action_dim_qr, device=device)
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.device=device

    def train(self):
        epochs = 5000
        max_steps = 100
        # Tracking
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        joint_actions = []
        loss_log = defaultdict(list)
        mcts_stats = []
        efe_stats = pd.DataFrame(
            columns=[
                "episode",
                "step",
                f"cv_data_quality"
                f"cv_data_quality_target"
                f"cv_throughput"
                f"cv_throughput_target"
                f"cv_model_size"
                f"cv_model_size_target"
                f"cv_cores"
                f"cv_free_cores",
                f"qr_data_quality"
                f"qr_data_quality_target"
                f"qr_throughput"
                f"qr_throughput_target"
                f"qr_model_size"
                f"qr_model_size_target"
                f"qr_cores"
                f"qr_free_cores",
            ],
            index=None,
        )
        name = "_efe_no_start_ig_theta"
        data_qualities = np.linspace(
            self.boundaries["data_quality"]["min"],
            self.boundaries["data_quality"]["max"],
            10,
        )
        throughputs = np.linspace(
            self.boundaries["throughput"]["min"],
            self.boundaries["throughput"]["max"],
            10,
        )
        model_sizes_cv = np.linspace(
            self.boundaries["model_size"]["min"],
            self.boundaries["model_size"]["max"],
            5,
        )
        model_sizes_qr = np.asarray([1])
        # @boris:
        #   here we could configure it being able
        #   to dynamically adapt to different targets by increasing the mesh size
        data_quality_target_cv = np.asarray([self.cv_slo_targets["data_quality"]])
        data_quality_target_qr = np.asarray([self.qr_slo_targets["data_quality"]])
        throughput_target_cv = np.asarray([self.cv_slo_targets["throughput"]])
        throughput_target_qr = np.asarray([self.qr_slo_targets["throughput"]])
        model_target_cv = np.asarray([self.cv_slo_targets["model_size"]])
        model_target_qr = np.asarray([1])

        cores = np.linspace(
            self.boundaries["cores"]["min"],
            self.boundaries["cores"]["max"],
            8,
        )

        filename = "joint_states.npy"
        if os.path.exists(filename):
            joint_states = np.load(filename)
            print("Loaded joint_states from file.")
        else:
            mesh_cv = np.stack(
                np.meshgrid(
                    data_qualities,
                    data_quality_target_cv,
                    throughputs,
                    throughput_target_cv,
                    model_sizes_cv,
                    model_target_cv,
                    cores,
                    indexing="ij",
                ),
                axis=-1,
            )
            mesh_qr = np.stack(
                np.meshgrid(
                    data_qualities,
                    data_quality_target_qr,
                    throughputs,
                    throughput_target_qr,
                    model_sizes_qr,
                    model_target_qr,
                    cores,
                    indexing="ij",
                ),
                axis=-1,
            )

            # Get all possible index tuples
            shape_cv = mesh_cv.shape[:-1]
            shape_qr = mesh_qr.shape[:-1]
            indices_cv = list(itertools.product(*[range(s) for s in shape_cv]))
            indices_qr = list(itertools.product(*[range(s) for s in shape_qr]))

            # Prepare to collect valid joint states
            joint_states = []

            for idx_cv in indices_cv:
                for idx_qr in indices_qr:
                    start_state_cv = mesh_cv[idx_cv]
                    start_state_qr = mesh_qr[idx_qr]
                    cores_cv = start_state_cv[-1]
                    cores_qr = start_state_qr[-1]
                    free_cores = cores_cv - cores_qr
                    if free_cores >= 0:
                        # Augment each sample separately
                        aug_cv = np.concatenate([start_state_cv, [free_cores]])
                        aug_qr = np.concatenate([start_state_qr, [free_cores]])
                        # Concatenate to get final joint state (length 14)
                        joint_state = np.concatenate([aug_cv, aug_qr])
                        joint_states.append(joint_state)
            np.save("joint_states.npy", joint_states)
        # Example: sample a random valid joint state
        joint_states = np.array(joint_states)  # shape: (num_valid, 14)
        print(f"Number of valid joint states: {len(joint_states)}")

        end_training = False
        for epoch in range(epochs):
            # sampling start states, bc w/o this part encoder/decoder see only a tiny part of the env
            # so the overall worl model is bad
            idx = np.random.randint(len(joint_states))
            joint_state = joint_states[idx]
            state_cv, state_qr = np.split(joint_state, 2)
            state_cv, state_qr = self.agent.min_max_scale(state_cv), self.agent.min_max_scale(state_qr)
            done = False
            step = 0
            trajectory = []
            total_reward = 0
            while not done and step < max_steps:
                """
                Loop "Collects experiences" => Collecting samples for training
                """
                if sample_action and self.agent.train_transition:
                    # if we are in the transition model training phase and want to use EFE
                    if len(joint_actions) == 0:
                        # joint_actions: [Horizon, #services]

                        joint_actions, stat = self.agent.select_joint_action(
                            joint_state, step, epoch
                        )
                        efe_stats = self.create_efe_stats(
                            stat, joint_state, epoch, step, efe_stats
                        )

                    # first action remaining in the sequence (ASSUMING THAT THIS IS A TUPLE/TENSOR OF SHAPE (2,)
                    joint_action = [joint_actions.pop(0)]
                else:
                    joint_action = [
                        (
                            np.random.choice(range(self.action_dim_cv)),
                            np.random.choice(range(self.action_dim_qr)),
                        )
                    ]
                action_cv, action_qr = joint_action[0]
                state_cv, state_qr = np.split(joint_state, 2)
                next_state_cv, reward_cv = self.agent.probe_transition(
                    state_cv, None, action_cv, service_type="cv"
                )
                next_state_qr, reward_qr = self.agent.probe_transition(
                    state_qr, None, action_qr, service_type="qr"

                )
                # ckp
                joint_state_next = np.concatenate([next_state_cv, next_state_qr])
                assert len(joint_action) == 1
                if step % 25 != 0:
                    # put one experience in the train set
                    self.agent.save_experience(
                        joint_state, joint_action[0], joint_state_next
                    )
                else:
                    self.agent.save_experience(
                        joint_state, joint_action[0], joint_state_next, to_train=False
                    )

                joint_state = copy.deepcopy(joint_state_next)
                total_reward += reward_cv + reward_qr
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

            # if end_training or epoch == (epochs - 10):
            #     start_state = np.array(
            #         [0.4, 0.000], dtype=np.float32
            #     )  # np.array([-0.5, 0.000], dtype=np.float32)
            #     traject = {  # TODO: test trajectories for ES env
            #         "left": [0, 0, 0, 0, 0],
            #         "right": [2, 2, 2, 2, 2],
            #         "stay": [1, 1, 1, 1, 1],
            #         "mix": [2, 1, 1, 2, 0],
            #     }
            #     for k, v in traject.items():
            #         actual_obs, reconstr_obs, _ = self.agent.pretend_a_trajectory(
            #             start_state, v
            #         )
            #         plot_trajectory_comparison(
            #             actual_obs, reconstr_obs, type="encoder" + "_" + k
            #         )
            #
            #         actual_obs, reconstr_obs, next_obs = agent.pretend_a_trajectory(
            #             start_state, v, use_transition=True
            #         )
            #         plot_trajectory_comparison(
            #             actual_obs, next_obs, type="transition" + "_" + k
            #         )
            if end_training:
                break
        import matplotlib.pyplot as plt

        efe_stats.to_csv(f"./efe_logs_{name}.csv", sep="|")
        # Plot episode reward
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Reward Over Time")
        plt.xlabel("Epoch")
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
                "episode": list(range(epochs + 1)),
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

        self.agent.buffer = []
        self.agent.val_buffer = []
        torch.save({"agent": self.agent}, f"aif_agent_checkpoint_{name}.pth")

    def create_efe_stats(self, stat, joint_state, epoch, step, efe_stats):
        df_this_call = pd.DataFrame.from_records(stat, columns=["action", "prob"])
        df_this_call["epoch"] = epoch
        df_this_call["step"] = step
        for idx, ser in enumerate(["cv", "qr"]):
            df_this_call[f"{ser}_data_quality"] = joint_state[0]
            df_this_call[f"{ser}_data_quality_target"] = joint_state[1]
            df_this_call[f"{ser}_throughput"] = joint_state[2]
            df_this_call[f"{ser}_throughput_target"] = joint_state[3]
            df_this_call[f"{ser}_model_size"] = joint_state[4]
            df_this_call[f"{ser}_model_size_target"] = joint_state[5]
            df_this_call[f"{ser}_cores"] = joint_state[6]
            df_this_call[f"{ser}_free_cores"] = joint_state[7]
        efe_stats = pd.concat([efe_stats, df_this_call], ignore_index=True)
        return efe_stats


if __name__ == "__main__":
    trainer = MCDaciTrainer(
        boundaries={
            "model_size": {"min": 1, "max": 5},
            "data_quality": {"min": 100, "max": 1000},
            "cores": {"min": 100, "max": 8},
            "throughput": {"min": 0, "max": 100},
        },
        cv_slo_targets={
            "data_quality": 288,
            "model_size": 4,
            "throughput": 5,
        },
        qr_slo_targets={
            "data_quality": 900,
            "throughput": 75,
        },
        action_dim_cv=7,
        action_dim_qr=5,
    )

    trainer.train()
