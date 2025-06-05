import torch
import numpy as np
from typing import Tuple, Dict


class VectorizedEnvironment:
    """GPU-accelerated vectorized environment for parallel state transitions"""

    def __init__(self, boundaries: Dict[str, Dict[str, float]], device: str = "cuda:1"):
        self.device = device
        self.boundaries = boundaries

        # Convert boundaries to tensors for GPU operations
        self.min_vals = torch.tensor([
            boundaries["data_quality"]["min"],
            boundaries["data_quality"]["min"],
            0, 0,
            boundaries["model_size"]["min"],
            boundaries["model_size"]["min"],
            boundaries["cores"]["min"],
            boundaries["cores"]["min"],
        ], device=device, dtype=torch.float32)

        self.max_vals = torch.tensor([
            boundaries["data_quality"]["max"],
            boundaries["data_quality"]["max"],
            100, 100,
            boundaries["model_size"]["max"],
            boundaries["model_size"]["max"],
            boundaries["cores"]["max"],
            boundaries["cores"]["max"],
        ], device=device, dtype=torch.float32)

        # Step sizes
        self.step_data_quality_cv = 100
        self.step_data_quality_qr = 32
        self.step_cores = 1
        self.step_model_size = 1

    def min_max_scale(self, vals: torch.Tensor) -> torch.Tensor:
        """Vectorized min-max scaling on GPU"""

        if vals.shape[1] != self.max_vals.shape[0]:
            scaled = torch.where(
                self.min_vals.repeat(2) == self.max_vals.repeat(2),
                torch.ones_like(vals),
                (vals - self.min_vals.repeat(2)) / (self.max_vals.repeat(2) - self.min_vals.repeat(2))
            )
        else:
            scaled = torch.where(
                self.min_vals == self.max_vals,
                torch.ones_like(vals),
                (vals - self.min_vals) / (self.max_vals - self.min_vals)
            )
        return torch.clamp(scaled, 0, 1)

    def min_max_rescale(self, scaled_vals: torch.Tensor) -> torch.Tensor:
        """Vectorized min-max rescaling on GPU"""
        if scaled_vals.shape != self.max_vals.shape[0]:
            rescaled = torch.where(
                self.min_vals.repeat(2) == self.max_vals.repeat(2),
                self.min_vals.repeat(2),
                scaled_vals * (self.max_vals.repeat(2) - self.min_vals.repeat(2)) + self.min_vals.repeat(2),
            )
        else:
            rescaled = torch.where(
                self.min_vals == self.max_vals,
                self.min_vals,
                scaled_vals * (self.max_vals - self.min_vals) + self.min_vals,
            )
        return rescaled

    def vectorized_transition_cv(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized state transitions for CV service
        states  : [batch_size, 8] – normalized
        actions : [batch_size]    – action indices
        Returns : (next_states, rewards)
        """
        thresholds_cv = {
            "max": {"data_quality": 320},
            "min": {"data_quality": 128},
        }
        dq_min, dq_max = (
            thresholds_cv["min"]["data_quality"],
            thresholds_cv["max"]["data_quality"],
        )

        new_states = states.clone()
        unnorm_states = self.min_max_rescale(new_states)  # to un-scaled space

        dq_mask_down = actions == 1
        dq_mask_up = actions == 2

        delta_dq_down = -self.step_data_quality_cv
        delta_dq_up = self.step_data_quality_cv

        new_dq = unnorm_states[:, 0].clone()
        new_dq[dq_mask_down] += delta_dq_down
        new_dq[dq_mask_up] += delta_dq_up

        valid_q_mask = (
                (new_dq >= self.boundaries["data_quality"]["min"])
                & (new_dq <= self.boundaries["data_quality"]["max"])
        )
        update_q_mask = (dq_mask_down | dq_mask_up) & valid_q_mask
        unnorm_states[update_q_mask, 0] = new_dq[update_q_mask]

        cores_mask_down = actions == 3
        cores_mask_up = actions == 4

        delta_cores_down = -self.step_cores
        delta_cores_up = self.step_cores

        new_cores = unnorm_states[:, 6].clone()
        new_cores[cores_mask_down] += delta_cores_down
        new_cores[cores_mask_up] += delta_cores_up

        valid_cores_mask = (
                (new_cores > 0)
                & (
                        new_cores
                        <= unnorm_states[:, 7]
                        + torch.abs(
                    torch.where(
                        cores_mask_down,
                        delta_cores_down,
                        torch.where(cores_mask_up, delta_cores_up, 0),
                    )
                )
                )
        )
        cores_upd_mask = (cores_mask_down | cores_mask_up) & valid_cores_mask
        old_cores = unnorm_states[cores_upd_mask, 6].clone()
        unnorm_states[cores_upd_mask, 6] = new_cores[cores_upd_mask]
        unnorm_states[cores_upd_mask, 7] -= new_cores[cores_upd_mask] - old_cores

        model_mask_down = actions == 5
        model_mask_up = actions == 6

        delta_model_down = -self.step_model_size
        delta_model_up = self.step_model_size

        new_model = unnorm_states[:, 4].clone()
        new_model[model_mask_down] += delta_model_down
        new_model[model_mask_up] += delta_model_up

        valid_model_mask = (
                (new_model >= self.boundaries["model_size"]["min"])
                & (new_model <= self.boundaries["model_size"]["max"])
        )
        model_upd_mask = (model_mask_down | model_mask_up) & valid_model_mask
        unnorm_states[model_upd_mask, 4] = new_model[model_upd_mask]

        unnorm_states[:, 0] = torch.clamp(unnorm_states[:, 0], dq_min, dq_max)
        unnorm_states[:, 1] = torch.clamp(unnorm_states[:, 1], dq_min, dq_max)

        new_states = self.min_max_scale(unnorm_states)  # back to [0, 1]
        rewards = self.calculate_rewards_cv(new_states)
        return new_states, rewards

    def vectorized_transition_qr(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized state transitions for QR service
        states  : [batch_size, 8] – normalized
        actions : [batch_size]    – action indices
        Returns : (next_states, rewards)
        """
        # ⇢ NEW — service-specific thresholds
        thresholds_qr = {
            "max": {"data_quality": 1000},
            "min": {"data_quality": 300},
        }
        dq_min, dq_max = (
            thresholds_qr["min"]["data_quality"],
            thresholds_qr["max"]["data_quality"],
        )

        new_states = states.clone()
        unnorm_states = self.min_max_rescale(new_states)

        dq_mask_down = actions == 1
        dq_mask_up = actions == 2

        delta_dq_down = -self.step_data_quality_qr
        delta_dq_up = self.step_data_quality_qr

        new_dq = unnorm_states[:, 0].clone()
        new_dq[dq_mask_down] += delta_dq_down
        new_dq[dq_mask_up] += delta_dq_up

        valid_q_mask = (
                (new_dq >= self.boundaries["data_quality"]["min"])
                & (new_dq <= self.boundaries["data_quality"]["max"])
        )
        update_q_mask = (dq_mask_down | dq_mask_up) & valid_q_mask
        unnorm_states[update_q_mask, 0] = new_dq[update_q_mask]

        cores_mask_down = actions == 3
        cores_mask_up = actions == 4

        delta_cores_down = -self.step_cores
        delta_cores_up = self.step_cores

        new_cores = unnorm_states[:, 6].clone()
        new_cores[cores_mask_down] += delta_cores_down
        new_cores[cores_mask_up] += delta_cores_up

        valid_cores_mask = (
                (new_cores > 0)
                & (
                        new_cores
                        <= unnorm_states[:, 7]
                        + torch.abs(
                    torch.where(
                        cores_mask_down,
                        delta_cores_down,
                        torch.where(cores_mask_up, delta_cores_up, 0),
                    )
                )
                )
        )
        cores_upd_mask = (cores_mask_down | cores_mask_up) & valid_cores_mask
        old_cores = unnorm_states[cores_upd_mask, 6].clone()
        unnorm_states[cores_upd_mask, 6] = new_cores[cores_upd_mask]
        unnorm_states[cores_upd_mask, 7] -= new_cores[cores_upd_mask] - old_cores

        unnorm_states[:, 0] = torch.clamp(unnorm_states[:, 0], dq_min, dq_max)
        unnorm_states[:, 1] = torch.clamp(unnorm_states[:, 1], dq_min, dq_max)

        new_states = self.min_max_scale(unnorm_states)
        rewards = self.calculate_rewards_qr(new_states)
        return new_states, rewards

    def calculate_rewards_cv(self, states: torch.Tensor) -> torch.Tensor:
        sol_qual = 0.25 * states[:, 0] + 0.75 * states[:, 4]
        tp = states[:, 2]
        # Use actual preferences if available, otherwise fallback to placeholder
        #if hasattr(self, 'preferences_cv') and self.preferences_cv is not None:
        threshold = self.preferences_cv[0].item()
        threshold_tp = self.preferences_cv[1].item()
        #else:
        #    threshold = 0.5  # Fallback threshold
        rewards_sol_qual = torch.where(sol_qual >= threshold, 1.0, -1.0)
        rewards_hroughputal = torch.where(tp >= threshold_tp, 1.0, -1.0)
        rewards = rewards_sol_qual + rewards_hroughputal
        return rewards

    def calculate_rewards_qr(self, states: torch.Tensor) -> torch.Tensor:
        sol_qual =  states[:, 0]
        tp = states[:, 2]
        # Use actual preferences if available, otherwise fallback to placeholder
        #if hasattr(self, 'preferences_cv') and self.preferences_cv is not None:
        threshold = self.preferences_qr[0].item()
        threshold_tp = self.preferences_qr[1].item()
        #else:
        #    threshold = 0.5  # Fallback threshold
        rewards_sol_qual = torch.where(sol_qual >= threshold, 1.0, -1.0)
        rewards_hroughputal = torch.where(tp >= threshold_tp, 1.0, -1.0)
        rewards = rewards_sol_qual + rewards_hroughputal
        return rewards


    def vectorized_multistep_rollout(self, initial_states: torch.Tensor,
                                     actions_cv: torch.Tensor, actions_qr: torch.Tensor,
                                     horizon: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform vectorized multi-step rollouts
        initial_states: [batch_size, 16] - joint initial states
        actions_cv: [batch_size, horizon] - CV actions
        actions_qr: [batch_size, horizon] - QR actions
        Returns: (cv_trajectories, qr_trajectories) each [batch_size, horizon, 8]
        """
        batch_size = initial_states.shape[0]

        # Split joint states
        states_cv, states_qr = torch.chunk(initial_states.squeeze(), 2, dim=1)

        # Storage for trajectories
        traj_cv = torch.zeros(batch_size, horizon, 8, device=self.device, dtype=torch.float32)
        traj_qr = torch.zeros(batch_size, horizon, 8, device=self.device, dtype=torch.float32)

        current_cv, current_qr = states_cv, states_qr

        for t in range(horizon):
            # Get actions for this timestep
            actions_cv_t = actions_cv[:, t]
            actions_qr_t = actions_qr[:, t]

            # Vectorized transitions
            next_cv, _ = self.vectorized_transition_cv(current_cv, actions_cv_t)
            next_qr, _ = self.vectorized_transition_qr(current_qr, actions_qr_t)

            # Store trajectories
            traj_cv[:, t] = next_cv
            traj_qr[:, t] = next_qr

            # Update current states
            current_cv, current_qr = next_cv, next_qr

        return traj_cv, traj_qr