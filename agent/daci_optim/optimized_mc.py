import itertools
import os
import numpy as np
import copy
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pandas as pd
import time

from agent.daci_optim.optimized_daci_agent import OptimizedMCDACIAgent


def plot_trajectory_comparison(actual_obs, predicted_obs, type_name, save_name):
    actual_obs = np.array(actual_obs)
    predicted_obs = np.array(predicted_obs)
    steps = np.arange(len(actual_obs))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps, actual_obs[:, 0], label="Actual Position", linewidth=2)
    plt.plot(steps, predicted_obs[:, 0], "--", label="Predicted Position", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Position")
    plt.title("Position: Actual vs. Predicted")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, actual_obs[:, 1], label="Actual Velocity", linewidth=2)
    plt.plot(steps, predicted_obs[:, 1], "--", label="Predicted Velocity", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Velocity")
    plt.title("Velocity: Actual vs. Predicted")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"trajectories_efe_plot_{type_name}_{save_name}.png")
    plt.close()


class OptimizedMCDaciTrainer:
    def __init__(
            self,
            boundaries: dict,
            cv_slo_targets: dict,
            qr_slo_targets: dict,
            action_dim_cv: int = 7,
            action_dim_qr: int = 5,
            device: str = "cuda:0",
            parallel_episodes: int = 8,  # Number of parallel episodes
    ):
        self.boundaries = boundaries
        self.qr_slo_targets = qr_slo_targets
        self.cv_slo_targets = cv_slo_targets
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.device = device
        self.parallel_episodes = parallel_episodes

        # Initialize optimized agent
        self.agent = OptimizedMCDACIAgent(
            boundaries, cv_slo_targets, qr_slo_targets,
            action_dim_cv, action_dim_qr, device=device,
            batch_size=8  # Increased batch size for better GPU utilization
        )

        # Pre-generate joint states on GPU for faster sampling
        self.joint_states = self._generate_joint_states()

    def _generate_joint_states(self):
        """Pre-generate all valid joint states and store on GPU"""
        filename = "joint_states_gpu.pt"
        if os.path.exists(filename):
            joint_states = torch.load(filename, map_location=self.device)
            print(f"Loaded {len(joint_states)} joint states from file.")
            return joint_states

        print("Generating joint states...")
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

        mesh_cv = np.stack(
            np.meshgrid(
                data_qualities, data_quality_target_cv, throughputs, throughput_target_cv,
                model_sizes_cv, model_target_cv, cores, indexing="ij",
            ),
            axis=-1,
        )
        mesh_qr = np.stack(
            np.meshgrid(
                data_qualities, data_quality_target_qr, throughputs, throughput_target_qr,
                model_sizes_qr, model_target_qr, cores, indexing="ij",
            ),
            axis=-1,
        )

        shape_cv = mesh_cv.shape[:-1]
        shape_qr = mesh_qr.shape[:-1]
        indices_cv = list(itertools.product(*[range(s) for s in shape_cv]))
        indices_qr = list(itertools.product(*[range(s) for s in shape_qr]))

        joint_states = []
        for idx_cv in indices_cv:
            for idx_qr in indices_qr:
                start_state_cv = mesh_cv[idx_cv]
                start_state_qr = mesh_qr[idx_qr]
                cores_cv = start_state_cv[-1]
                cores_qr = start_state_qr[-1]
                free_cores = cores_cv - cores_qr
                if free_cores >= 0:
                    aug_cv = np.concatenate([start_state_cv, [free_cores]])
                    aug_qr = np.concatenate([start_state_qr, [free_cores]])
                    joint_state = np.concatenate([aug_cv, aug_qr])
                    joint_states.append(joint_state)

        joint_states = torch.tensor(joint_states, dtype=torch.float32, device=self.device)
        torch.save(joint_states, filename)
        print(f"Generated and saved {len(joint_states)} joint states.")
        return joint_states

    def sample_initial_states(self, batch_size: int) -> torch.Tensor:
        """Sample initial states efficiently on GPU"""
        indices = torch.randint(0, len(self.joint_states), (batch_size,), device=self.device)
        sampled_states = self.joint_states[indices]

        # Apply min-max scaling on GPU
        return self.agent.vec_env.min_max_scale(sampled_states)

    def parallel_episode_generation(self, max_steps: int = 100) -> dict:
        """Generate multiple episodes in parallel using vectorized operations"""

        # Sample initial states for parallel episodes
        joint_states = self.sample_initial_states(self.parallel_episodes)

        episode_data = {
            'observations': [],
            'actions_cv': [],
            'actions_qr': [],
            'next_observations': [],
            'rewards': [],
            'episode_lengths': [],
            'total_rewards': []
        }

        current_states = joint_states.clone()
        episode_rewards = torch.zeros(self.parallel_episodes, device=self.device)
        episode_lengths = torch.zeros(self.parallel_episodes, dtype=torch.long, device=self.device)
        done_episodes = torch.zeros(self.parallel_episodes, dtype=torch.bool, device=self.device)

        for step in range(max_steps):
            if done_episodes.all():
                break

            # Select actions for all active episodes
            if self.agent.train_transition and random.random() < 0.7:  # Use EFE 70% of the time
                # For simplicity in parallel execution, use random actions for now
                # In practice, you'd want to vectorize EFE computation across episodes
                actions_cv = torch.randint(0, self.action_dim_cv, (self.parallel_episodes,), device=self.device)
                actions_qr = torch.randint(0, self.action_dim_qr, (self.parallel_episodes,), device=self.device)
            else:
                actions_cv = torch.randint(0, self.action_dim_cv, (self.parallel_episodes,), device=self.device)
                actions_qr = torch.randint(0, self.action_dim_qr, (self.parallel_episodes,), device=self.device)

            # Execute transitions in parallel
            next_states, rewards = self.agent.vectorized_probe_transition(
                current_states, actions_cv, actions_qr
            )

            # Store transitions for active episodes
            active_mask = ~done_episodes
            if active_mask.any():
                episode_data['observations'].extend(current_states[active_mask])
                episode_data['actions_cv'].extend(actions_cv[active_mask])
                episode_data['actions_qr'].extend(actions_qr[active_mask])
                episode_data['next_observations'].extend(next_states[active_mask])
                episode_data['rewards'].extend(rewards[active_mask])

            # Update episode stats
            episode_rewards += rewards * active_mask.float()
            episode_lengths += active_mask.long()

            # Check for episode termination (you can add more sophisticated termination conditions)
            # For now, episodes end after max_steps or based on some other criteria
            current_states = next_states

            # Mark episodes as done if they reach max length
            done_episodes = (episode_lengths >= max_steps)

        # Record final episode statistics
        episode_data['total_rewards'] = episode_rewards.cpu().numpy().tolist()
        episode_data['episode_lengths'] = episode_lengths.cpu().numpy().tolist()

        return episode_data

    def efficient_experience_storage(self, episode_data: dict):
        """Efficiently store episode data in agent's buffer"""
        observations = episode_data['observations']
        actions_cv = episode_data['actions_cv']
        actions_qr = episode_data['actions_qr']
        next_observations = episode_data['next_observations']

        # Batch the experience storage
        for i, (obs, act_cv, act_qr, next_obs) in enumerate(zip(
                observations, actions_cv, actions_qr, next_observations
        )):
            # Determine train/validation split
            to_train = (i % 25 != 0)  # Every 25th sample goes to validation

            # Convert to proper format
            joint_action = (act_cv.item(), act_qr.item())
            self.agent.save_experience(obs, joint_action, next_obs, to_train=to_train)

    def train(self):
        epochs = 2000  # Reduced epochs since we're processing more data per epoch

        # Tracking
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        loss_log = defaultdict(list)

        name = "_optimized_gpu"

        print(f"Starting optimized training on {self.device}")
        print(f"Parallel episodes: {self.parallel_episodes}")
        print(f"Total joint states: {len(self.joint_states)}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Generate multiple episodes in parallel
            episode_data = self.parallel_episode_generation(max_steps=100)

            # Store experiences efficiently
            self.efficient_experience_storage(episode_data)

            # Record episode statistics
            episode_rewards.extend(episode_data['total_rewards'])
            episode_lengths.extend(episode_data['episode_lengths'])

            avg_reward = np.mean(episode_data['total_rewards'])
            avg_length = np.mean(episode_data['episode_lengths'])

            print(f"[Epoch {epoch}] Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}")

            # Train the models with accumulated experience
            if len(self.agent.buffer_obs) >= self.agent.batch_size:
                loss, dict_loss, end_training = self.agent.fit_experience(epoch, epochs)
                training_losses.append((epoch, loss))

                for k, v in dict_loss.items():
                    loss_log[k].append((epoch, v))

                print(f"  Training Loss: {loss:.4f}")

                if end_training:
                    print("Early stopping triggered")
                    break

            # GPU memory management
            if epoch % 100 == 0:
                torch.cuda.empty_cache()

            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0:
                print(f"  Epoch time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f}s")

        # Plotting and saving results
        self._save_results(episode_rewards, episode_lengths, training_losses, loss_log, name)

        # Save the trained agent
        torch.save({"agent": self.agent}, f"optimized_agent_checkpoint_{name}.pth")
        print(f"Saved agent checkpoint")

    def _save_results(self, episode_rewards, episode_lengths, training_losses, loss_log, name):
        """Save training results and create plots"""

        # Plot episode rewards and losses
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)

        plt.subplot(1, 3, 2)
        if training_losses:
            episodes, losses = zip(*training_losses)
            plt.plot(episodes, losses)
            plt.title("Training Loss Over Time")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

        plt.subplot(1, 3, 3)
        plt.plot(episode_lengths)
        plt.title("Episode Lengths Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"training_overview_{name}.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Plot detailed loss components
        if loss_log:
            plt.figure(figsize=(15, 10))
            n_components = len(loss_log)
            cols = 3
            rows = (n_components + cols - 1) // cols

            for i, (k, values) in enumerate(loss_log.items()):
                plt.subplot(rows, cols, i + 1)
                episodes, vals = zip(*values)
                plt.plot(episodes, vals, label=k)
                plt.title(f"{k}")
                plt.xlabel("Epoch")
                plt.ylabel("Loss Value")
                plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"loss_components_{name}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Save detailed logs
        log_df = pd.DataFrame({
            "episode_reward": episode_rewards,
            "episode_length": episode_lengths,
        })

        if training_losses:
            # Align training losses with episodes
            loss_dict = dict(training_losses)
            log_df["total_loss"] = log_df.index.map(lambda x: loss_dict.get(x, np.nan))

        # Add component losses
        for k, values in loss_log.items():
            loss_dict = dict(values)
            log_df[k] = log_df.index.map(lambda x: loss_dict.get(x, np.nan))

        log_df.to_csv(f"optimized_training_log_{name}.csv", index=False)
        print(f"📄 Saved training log to optimized_training_log_{name}.csv")


if __name__ == "__main__":
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Ensure CUDA is available
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = "cpu"
        print("CUDA not available, using CPU")

    trainer = OptimizedMCDaciTrainer(
        boundaries={
            "model_size": {"min": 1, "max": 5},
            "data_quality": {"min": 100, "max": 1000},
            "cores": {"min": 1, "max": 8},  # Fixed min cores to be >= 1
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
        device=device,
        parallel_episodes=16,  # Increase for better GPU utilization
    )

    trainer.train()