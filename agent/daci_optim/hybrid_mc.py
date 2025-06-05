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

from agent.daci_optim.hybrid_daci_agent import HybridMCDACIAgent


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


class HybridMCDaciTrainer:
    def __init__(
            self,
            boundaries: dict,
            cv_slo_targets: dict,
            qr_slo_targets: dict,
            action_dim_cv: int = 7,
            action_dim_qr: int = 5,
            device: str = "cuda:0",
    ):
        self.boundaries = boundaries
        self.qr_slo_targets = qr_slo_targets
        self.cv_slo_targets = cv_slo_targets
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.device = device

        # Initialize hybrid agent
        self.agent = HybridMCDACIAgent(
            boundaries, cv_slo_targets, qr_slo_targets,
            action_dim_cv, action_dim_qr, device=device,
        )

        # Pre-generate joint states on GPU for faster sampling in later phases
        self.joint_states = self._generate_joint_states()

    def _generate_joint_states(self):
        """Pre-generate all valid joint states and store on GPU"""
        filename = "joint_states_hybrid.pt"
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

    def sample_initial_state(self) -> np.ndarray:
        """Sample a single initial state (for world model phase)"""
        idx = torch.randint(0, len(self.joint_states), (1,)).item()
        sampled_state = self.joint_states[idx].cpu().numpy()

        # Apply min-max scaling using agent's method
        state_cv, state_qr = np.split(sampled_state, 2)
        state_cv = self.agent.min_max_scale(state_cv)
        state_qr = self.agent.min_max_scale(state_qr)

        return np.concatenate([state_cv, state_qr])

    def simple_episode_generation(self, max_steps: int = 200) -> dict:
        """
        Simple single episode generation for world model training phase
        (Similar to original but optimized)
        """
        # Sample initial state
        joint_state = self.sample_initial_state()

        episode_data = {
            'observations': [],
            'actions_cv': [],
            'actions_qr': [],
            'next_observations': [],
            'rewards': [],
            'total_reward': 0,
            'episode_length': 0
        }

        done = False
        step = 0
        total_reward = 0
        joint_actions = []  # For EFE-based action selection

        while not done and step < max_steps:
            # Action selection
            if self.agent.train_transition and len(joint_actions) == 0:
                # Use EFE-based action selection when transition model is available
                joint_actions, _ = self.agent.select_joint_action(joint_state, step, 0)
                action_cv, action_qr = joint_actions.pop(0)
            else:
                # Random action selection during world model training
                action_cv = np.random.choice(range(self.action_dim_cv))
                action_qr = np.random.choice(range(self.action_dim_qr))

            # Environment transitions using lightweight CPU operations
            state_cv, state_qr = np.split(joint_state, 2)

            next_state_cv, reward_cv = self.agent.simple_probe_transition(
                state_cv, None, action_cv, service_type="cv"
            )
            next_state_qr, reward_qr = self.agent.simple_probe_transition(
                state_qr, None, action_qr, service_type="qr"
            )

            joint_state_next = np.concatenate([next_state_cv, next_state_qr])

            # Store transition data
            episode_data['observations'].append(joint_state.copy())
            episode_data['actions_cv'].append(action_cv)
            episode_data['actions_qr'].append(action_qr)
            episode_data['next_observations'].append(joint_state_next.copy())
            episode_data['rewards'].append(reward_cv + reward_qr)

            joint_state = joint_state_next
            total_reward += reward_cv + reward_qr
            step += 1

            if step == max_steps:
                done = True

        episode_data['total_reward'] = total_reward
        episode_data['episode_length'] = step

        return episode_data

    def parallel_episode_generation(self, max_steps: int = 100, num_episodes: int = 16) -> dict:
        """
        Parallel episode generation for transition/all training phases
        """
        # Sample initial states for parallel episodes
        indices = torch.randint(0, len(self.joint_states), (num_episodes,), device=self.device)
        joint_states = self.joint_states[indices]

        # Apply scaling
        joint_states = self.agent.vec_env.min_max_scale(joint_states)

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
        episode_rewards = torch.zeros(num_episodes, device=self.device)
        episode_lengths = torch.zeros(num_episodes, dtype=torch.long, device=self.device)
        done_episodes = torch.zeros(num_episodes, dtype=torch.bool, device=self.device)

        for step in range(max_steps):
            if done_episodes.all():
                break

            # Select actions for all active episodes
            # For simplicity, use random actions (EFE computation would be expensive for all episodes)
            actions_cv = torch.randint(0, self.action_dim_cv, (num_episodes,), device=self.device)
            actions_qr = torch.randint(0, self.action_dim_qr, (num_episodes,), device=self.device)

            # Execute transitions in parallel using vectorized environment
            traj_cv, traj_qr = self.agent.vec_env.vectorized_multistep_rollout(
                current_states,  # ✅ Fixed: removed incorrect .unsqueeze(1)
                actions_cv.unsqueeze(1),
                actions_qr.unsqueeze(1),
                1
            )
            # Extract single step results (take the first and only timestep)
            next_states_cv = traj_cv[:, 0]  # [batch_size, 8]
            next_states_qr = traj_qr[:, 0]  # [batch_size, 8]
            next_states = torch.cat([next_states_cv, next_states_qr], dim=1)  # [batch_size, 16]

            # Calculate rewards from the actual next states
            rewards_cv = self.agent.vec_env.calculate_rewards_cv(next_states_cv)
            rewards_qr = self.agent.vec_env.calculate_rewards_qr(next_states_qr)
            rewards = rewards_cv + rewards_qr

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

            current_states = next_states
            done_episodes = (episode_lengths >= max_steps)

        episode_data['total_rewards'] = episode_rewards.cpu().numpy().tolist()
        episode_data['episode_lengths'] = episode_lengths.cpu().numpy().tolist()

        return episode_data

    def efficient_experience_storage(self, episode_data: dict):
        """Store episode data using agent's adaptive method"""
        observations = episode_data['observations']
        actions_cv = episode_data['actions_cv']
        actions_qr = episode_data['actions_qr']
        next_observations = episode_data['next_observations']

        for i, (obs, act_cv, act_qr, next_obs) in enumerate(zip(
                observations, actions_cv, actions_qr, next_observations
        )):
            # Determine train/validation split
            to_train = (i % 25 != 0)

            # Convert to proper format based on agent's current phase
            if self.agent.current_phase == "world_model":
                joint_action = (int(act_cv), int(act_qr))
                self.agent.adaptive_save_experience(obs, joint_action, next_obs, to_train=to_train)
            else:
                # For GPU phases, actions might already be tensors
                act_cv_val = act_cv.item() if torch.is_tensor(act_cv) else int(act_cv)
                act_qr_val = act_qr.item() if torch.is_tensor(act_qr) else int(act_qr)
                joint_action = (act_cv_val, act_qr_val)

                # Convert observations to numpy if they're tensors
                obs_np = obs.cpu().numpy() if torch.is_tensor(obs) else obs
                next_obs_np = next_obs.cpu().numpy() if torch.is_tensor(next_obs) else next_obs

                self.agent.adaptive_save_experience(obs_np, joint_action, next_obs_np, to_train=to_train)

    def train(self):
        epochs = 5000

        # Tracking
        episode_rewards = []
        episode_lengths = []
        training_losses = []
        loss_log = defaultdict(list)

        name = "_hybrid_adaptive"

        print(f"Starting hybrid adaptive training on {self.device}")
        print(f"Phase-aware training: world_model -> transition -> all")
        print(f"Total joint states: {len(self.joint_states)}")

        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Generate episodes based on current training phase
            if self.agent.current_phase == "world_model":
                # Single episode generation with lightweight operations
                episode_data = self.simple_episode_generation(max_steps=200)

                # Convert single episode to list format for consistency
                episode_data['total_rewards'] = [episode_data['total_reward']]
                episode_data['episode_lengths'] = [episode_data['episode_length']]

                print(f"[Epoch {epoch}] Phase: {self.agent.current_phase}, "
                      f"Reward: {episode_data['total_reward']:.2f}, "
                      f"Length: {episode_data['episode_length']}")
            else:
                # Parallel episode generation with vectorized operations
                num_parallel = 8 if self.agent.current_phase == "transition" else 16
                episode_data = self.parallel_episode_generation(max_steps=200, num_episodes=num_parallel)

                avg_reward = np.mean(episode_data['total_rewards'])
                avg_length = np.mean(episode_data['episode_lengths'])

                print(f"[Epoch {epoch}] Phase: {self.agent.current_phase}, "
                      f"Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.1f}, "
                      f"Episodes: {len(episode_data['total_rewards'])}")

            # Store experiences efficiently
            self.efficient_experience_storage(episode_data)

            # Record episode statistics
            episode_rewards.extend(episode_data['total_rewards'])
            episode_lengths.extend(episode_data['episode_lengths'])

            # Train the models with accumulated experience
            buffer_size = len(self.agent.buffer) if self.agent.current_phase == "world_model" else len(
                self.agent.gpu_buffer_obs)

            if buffer_size >= self.agent.batch_size:
                loss, dict_loss, end_training = self.agent.fit_experience(epoch, epochs)
                training_losses.append((epoch, loss))

                for k, v in dict_loss.items():
                    loss_log[k].append((epoch, v))

                print(f"  Training Loss: {loss:.4f}")

                if end_training:
                    print("Early stopping triggered")
                    break

            # Performance monitoring
            if epoch % 50 == 0:
                phase_msg = f"Phase: {self.agent.current_phase}"
                buffer_msg = f"Buffer: {buffer_size}"
                batch_msg = f"Batch: {self.agent.batch_size}"
                print(f"  {phase_msg}, {buffer_msg}, {batch_msg}")

            # GPU memory management
            if epoch % 100 == 0:
                torch.cuda.empty_cache()

            epoch_time = time.time() - epoch_start
            if epoch % 10 == 0:
                print(f"  Epoch time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"Total training time: {total_time:.2f}s")
        print(f"Final phase: {self.agent.current_phase}")

        # Plotting and saving results
        self._save_results(episode_rewards, episode_lengths, training_losses, loss_log, name)

        # Save the trained agent
        torch.save({"agent": self.agent}, f"hybrid_agent_checkpoint_{name}.pth")
        print(f"Saved agent checkpoint")

    def _save_results(self, episode_rewards, episode_lengths, training_losses, loss_log, name):
        """Save training results and create plots"""

        # Plot episode rewards and losses
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.plot(episode_rewards)
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)

        plt.subplot(2, 3, 2)
        if training_losses:
            episodes, losses = zip(*training_losses)
            plt.plot(episodes, losses)
            plt.title("Training Loss Over Time")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)

        plt.subplot(2, 3, 3)
        plt.plot(episode_lengths)
        plt.title("Episode Lengths Over Time")
        plt.xlabel("Episode")
        plt.ylabel("Steps")
        plt.grid(True)

        # Plot phase transitions (assuming we can track them)
        plt.subplot(2, 3, 4)
        plt.hist(episode_rewards, bins=50, alpha=0.7)
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.grid(True)

        plt.subplot(2, 3, 5)
        plt.hist(episode_lengths, bins=30, alpha=0.7)
        plt.title("Episode Length Distribution")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.grid(True)

        # Running average
        plt.subplot(2, 3, 6)
        window = 100
        if len(episode_rewards) > window:
            running_avg = pd.Series(episode_rewards).rolling(window=window).mean()
            plt.plot(running_avg)
            plt.title(f"Running Average Reward (window={window})")
            plt.xlabel("Episode")
            plt.ylabel("Average Reward")
            plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"hybrid_training_overview_{name}.png", dpi=150, bbox_inches='tight')
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
            plt.savefig(f"hybrid_loss_components_{name}.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Save detailed logs
        log_df = pd.DataFrame({
            "episode_reward": episode_rewards,
            "episode_length": episode_lengths,
        })

        if training_losses:
            # Create a mapping from epoch to loss
            loss_dict = dict(training_losses)
            # Map episode indices to epochs (assuming one episode per epoch for simplicity)
            log_df["epoch"] = log_df.index
            log_df["total_loss"] = log_df["epoch"].map(lambda x: loss_dict.get(x, np.nan))

        # Add component losses
        for k, values in loss_log.items():
            loss_dict = dict(values)
            log_df[k] = log_df.get("epoch", log_df.index).map(lambda x: loss_dict.get(x, np.nan))

        log_df.to_csv(f"hybrid_training_log_{name}.csv", index=False)
        print(f"📄 Saved training log to hybrid_training_log_{name}.csv")


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

    trainer = HybridMCDaciTrainer(
        boundaries={
            "model_size": {"min": 1, "max": 5},
            "data_quality": {"min": 100, "max": 1000},
            "cores": {"min": 1, "max": 8},
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
    )

    trainer.train()