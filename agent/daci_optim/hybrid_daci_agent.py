import os
from collections import deque

import pandas as pd
import torch.nn.functional as F
import copy
import itertools
import logging
import random
from typing import Dict, Tuple

import numpy as np
import torch

from agent.LGBN import LGBN
from agent.SLORegistry import SLO_Registry, calculate_slo_fulfillment, to_normalized_slo_f
from agent.agent_utils import FullStateDQN, min_max_scale
from agent.daci.aif_utils import calculate_expected_free_energy
from agent.daci.network import SimpleDeltaTransitionNetwork, SimpleMCDaciWorldModel
from agent.daci_optim.vectorized_env import VectorizedEnvironment
from torch.nn import functional as F

from agent.es_registry import ESRegistry, ServiceType
from iwai.proj_types import WorldModelLoss

logger = logging.getLogger("multiscale")
ROOT = os.path.dirname(__file__)
slo_registry = SLO_Registry(ROOT + "/../../config/slo_config.json")
es_registry = ESRegistry(ROOT + "/../../config/es_registry.json")
client_slos_qr = slo_registry.get_all_SLOs_for_assigned_clients(
    ServiceType.QR, {"C_1": 100}
)[0]
client_slos_cv = slo_registry.get_all_SLOs_for_assigned_clients(
    ServiceType.CV, {"C_1": 100}
)[0]

boundaries_cv = es_registry.get_boundaries_minimalistic(ServiceType.CV, 8)
boundaries_qr = es_registry.get_boundaries_minimalistic(ServiceType.QR, 8)


def convert_rescaled_joint_state_to_slof(rescaled_joint_state: torch.Tensor):
    # rescaled_joint_state: shape (B, 16)
    normalized_slo_cv_list = []
    normalized_slo_qr_list = []
    for b in range(rescaled_joint_state.shape[0]):
        state_cv = rescaled_joint_state[b][:8]
        state_qr = rescaled_joint_state[b][8:]

        full_state_cv = FullStateDQN(
            state_cv[0],
            state_cv[1],
            state_cv[2],
            state_cv[3],
            state_cv[4],
            state_cv[5],
            0,  # cores irrelevant for SLO-F
            0,  # cores irrelevant for SLO-F
            boundaries_cv,
        )

        full_state_qr = FullStateDQN(
            state_qr[0],
            state_qr[1],
            state_qr[2],
            state_qr[3],
            state_qr[4],
            state_qr[5],
            0,  # cores irrelevant for SLO-F
            0,  # cores irrelevant for SLO-F
            boundaries_qr,
        )

        normalized_slo_cv = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_cv.to_normalized_dict(), client_slos_cv),
            client_slos_cv)
        normalized_slo_qr = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_qr.to_normalized_dict(), client_slos_qr),
            client_slos_qr)

        normalized_slo_cv_list.append(normalized_slo_cv)
        normalized_slo_qr_list.append(normalized_slo_qr)

    # Compute mean over batch
    batch_mean_cv = torch.mean(torch.stack(normalized_slo_cv_list), dim=0)
    batch_mean_qr = torch.mean(torch.stack(normalized_slo_qr_list), dim=0)
    return batch_mean_cv, batch_mean_qr


def convert_rescaled_state_qr_to_slof(state_qr: np.ndarray):
    """
    state_qr: np.ndarray of shape (8,)
    Returns the normalized SLO-F value for this sample.
    """
    full_state_qr = FullStateDQN(
        state_qr[0], state_qr[1], state_qr[2], state_qr[3],
        state_qr[4], state_qr[5], 0, 0, boundaries_qr,
    )
    normalized_slo_qr = to_normalized_slo_f(
        calculate_slo_fulfillment(full_state_qr.to_normalized_dict(), client_slos_qr),
        client_slos_qr)
    return normalized_slo_qr

def convert_rescaled_state_cv_to_slof(state_cv: np.ndarray):
    """
    state_cv: np.ndarray of shape (8,)
    Returns the normalized SLO-F value for this sample.
    """
    full_state_cv = FullStateDQN(
        state_cv[0], state_cv[1], state_cv[2], state_cv[3],
        state_cv[4], state_cv[5], 0, 0, boundaries_cv,
    )
    normalized_slo_cv = to_normalized_slo_f(
        calculate_slo_fulfillment(full_state_cv.to_normalized_dict(), client_slos_cv),
        client_slos_cv)
    return normalized_slo_cv


def freeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = True


ROOT = os.path.dirname(__file__)
df_t = pd.read_csv(ROOT + "/../../share/metrics/LGBN.csv")
lgbn = LGBN(show_figures=False, structural_training=False, df=df_t)


def sample_throughput_from_lgbn(data_quality, cores, model_size, service_type):
    partial_state = {"data_quality": data_quality, "cores": cores, "model_size": model_size}
    full_state = lgbn.predict_lgbn_vars(partial_state, service_type)
    return full_state['throughput']


class HybridMCDACIAgent:
    """Hybrid agent that adapts performance strategy based on training phase"""

    def __init__(
            self,
            boundaries: Dict[str, Dict[str, float | int]],
            cv_slo_targets: dict,
            qr_slo_targets: dict,
            lr_wm: float = 3e-5,
            lr_tn: float = 5e-3,
            joint_obs_dim: int = 2 * 8,
            joint_latent_dim: int = 2 * 4,
            action_dim_cv: int = 7,
            action_dim_qr: int = 5,
            width: int = 24,
            batch_size: int = 16,  # Keep original batch size for world model phase
            early_stopping_rounds=1500,
            iters_joint=1200,
            iters_wm=10,
            iters_tran=1,
            device: str = "cuda:1",
            depth_increase: int = 1,
            train_transition_from_iter: int = 600,
    ):
        self.device = device
        self.boundaries = boundaries
        self.iters_joint = iters_joint
        self.iters_wm = iters_wm
        self.iter_tran = iters_tran
        # Initialize vectorized environment (used only when beneficial)
        self.vec_env = VectorizedEnvironment(boundaries, device)

        self.world_model = SimpleMCDaciWorldModel(
            in_dim=joint_obs_dim,
            world_latent_dim=joint_latent_dim,
            width=width,
            depth_increase=depth_increase,
        ).to(device)

        self.transition_model_cv = SimpleDeltaTransitionNetwork(
            joint_latent_dim // 2, action_dim_cv, width, depth_increase=depth_increase
        ).to(device)

        self.transition_model_qr = SimpleDeltaTransitionNetwork(
            joint_latent_dim // 2, action_dim_qr, width, depth_increase=depth_increase
        ).to(device)

        # Training state
        self.patience_enc_dec = 0
        self.patience_transition = 0
        self.patience_finetune = 0
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.batch_size = batch_size
        self.patience = early_stopping_rounds

        # Lightweight buffer for world model phase (CPU-based for simplicity)
        self.buffer = deque(maxlen=20000)
        self.val_buffer = deque(maxlen=20000)

        # Heavy-duty GPU buffers for transition phase
        self.gpu_buffer_obs = deque(maxlen=20000)
        self.gpu_buffer_actions_cv = deque(maxlen=20000)
        self.gpu_buffer_actions_qr = deque(maxlen=20000)
        self.gpu_buffer_next_obs = deque(maxlen=20000)

        self.gpu_val_buffer_obs = deque(maxlen=20000)
        self.gpu_val_buffer_actions_cv = deque(maxlen=20000)
        self.gpu_val_buffer_actions_qr = deque(maxlen=20000)
        self.gpu_val_buffer_next_obs = deque(maxlen=20000)

        self.val_loss_enc = np.inf
        self.val_loss_transition = np.inf
        self.train_loss_finetune = np.inf

        # Optimizers
        self.optim_world_model = torch.optim.Adam(
            params=self.world_model.parameters(),
            lr=lr_wm,
            weight_decay=1e-4,
        )
        self.optim_transition_network = torch.optim.Adam(
            params=[
                {"params": self.transition_model_cv.parameters()},
                {"params": self.transition_model_qr.parameters()},
            ],
            lr=lr_tn,
            weight_decay=5e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim_world_model, step_size=100, gamma=0.95,
        )
        self.scheduler_trans = torch.optim.lr_scheduler.StepLR(
            self.optim_transition_network, step_size=100, gamma=0.95
        )

        self.train_transition_from_iter = train_transition_from_iter
        self.beta = 1

        # Performance tracking
        self._step_data_quality_cv = 100
        self._step_data_quality_qr = 32
        self.step_model_size = 1
        self.step_cores = 1

        # SLO targets and preferences
        self.cv_solo_targets = cv_slo_targets
        self.qr_solo_targets = qr_slo_targets

        # Set up preferences on GPU
        data_quality_cv = min_max_scale(
            cv_slo_targets["data_quality"],
            boundaries["data_quality"]["min"],
            boundaries["data_quality"]["max"],
        )
        model_size_cv = min_max_scale(
            cv_slo_targets["model_size"],
            boundaries["model_size"]["min"],
            boundaries["model_size"]["max"],
        )
        target_quality = 0.25 * data_quality_cv + 0.75 * model_size_cv

        self.preferences_cv = torch.tensor(
            [
                target_quality,
                min_max_scale(
                    cv_slo_targets["throughput"],
                    boundaries["throughput"]["min"],
                    boundaries["throughput"]["max"],
                ),
            ],
            device=device, dtype=torch.float32
        ).unsqueeze(0)

        self.preferences_qr = torch.tensor(
            [
                min_max_scale(
                    qr_slo_targets["data_quality"],
                    boundaries["data_quality"]["min"],
                    boundaries["data_quality"]["max"],
                ),
                min_max_scale(
                    qr_slo_targets["throughput"],
                    boundaries["throughput"]["min"],
                    boundaries["throughput"]["max"],
                ),
            ],
            device=device, dtype=torch.float32
        ).unsqueeze(0)

        # Update environment with preferences
        self.vec_env.preferences_cv = self.preferences_cv[0]
        self.vec_env.preferences_qr = self.preferences_qr[0]

        self.train_transition = False
        self.train_all = False
        self.mean_deltas = None
        self.std_deltas = None

        # Phase tracking for adaptive performance
        self.current_phase = "world_model"  # "world_model", "transition", "all"

    def _get_feature_bounds(self):
        """Calculate feature bounds for normalization"""
        min_vals = np.asarray([
            self.boundaries["data_quality"]["min"],
            self.boundaries["data_quality"]["min"],
            0, 0,
            self.boundaries["model_size"]["min"] if "model_size" in self.boundaries else 1.0,
            self.boundaries["model_size"]["min"] if "model_size" in self.boundaries else 1.0,
            self.boundaries["cores"]["min"],
            self.boundaries["cores"]["min"],
        ])
        max_vals = np.asarray([
            self.boundaries["data_quality"]["max"],
            self.boundaries["data_quality"]["max"],
            100, 100,
            self.boundaries["model_size"]["max"] if "model_size" in self.boundaries else 1.0,
            self.boundaries["model_size"]["max"] if "model_size" in self.boundaries else 1.0,
            self.boundaries["cores"]["max"],
            self.boundaries["cores"]["max"],
        ])
        return min_vals, max_vals

    def min_max_scale(self, vals: np.ndarray):
        """CPU-based scaling for lightweight operations"""
        min_vals, max_vals = self._get_feature_bounds()
        scaled = np.where(
            min_vals == max_vals, 1.0, (vals - min_vals) / (max_vals - min_vals)
        )
        return np.clip(scaled, 0, 1)

    def min_max_rescale(self, scaled_vals: np.ndarray):
        """CPU-based rescaling for lightweight operations"""
        min_vals, max_vals = self._get_feature_bounds()
        rescaled = np.where(
            min_vals == max_vals,
            min_vals,
            scaled_vals * (max_vals - min_vals) + min_vals,
        )
        return rescaled

    def simple_probe_transition(self, obs, next_obs, action, service_type: str) -> tuple:
        """
        Lightweight CPU-based transition for world-model phase.
        Adds clamping of `data_quality` (indices 0 & 1) to service-specific thresholds.
        """
        thresholds_qr = {
            "max": {"data_quality": 1000},
            "min": {"data_quality": 300},
        }
        thresholds_cv = {
            "max": {"data_quality": 320},
            "min": {"data_quality": 128},
        }
        thresholds = thresholds_qr if service_type == "qr" else thresholds_cv

        new_state = self.min_max_rescale(copy.deepcopy(obs))

        step_data_quality = (
            self._step_data_quality_cv
            if service_type == "cv"
            else self._step_data_quality_qr
        )

        if action == 0:
            pass

        elif 1 <= action <= 2:
            delta_dq = -step_data_quality if action == 1 else step_data_quality
            new_dq = new_state[0] + delta_dq

            if not (
                    self.boundaries["data_quality"]["min"] > new_dq
                    or new_dq > self.boundaries["data_quality"]["max"]
            ):
                new_state[0] = new_dq

        elif 3 <= action <= 4:
            delta_cores = -self.step_cores if action == 3 else self.step_cores
            new_cores = new_state[6] + delta_cores
            if not (new_cores <= 0 or delta_cores > new_state[7]):
                new_state[6] = new_cores
                new_state[7] -= delta_cores

        elif 5 <= action <= 6:
            delta_model = -self.step_model_size if action == 5 else self.step_model_size
            new_model_s = new_state[4] + delta_model
            if not (
                    self.boundaries["model_size"]["min"] > new_model_s
                    or new_model_s > self.boundaries["model_size"]["max"]
            ):
                new_state[4] = new_model_s

        data_quality_min = thresholds["min"]["data_quality"]
        data_quality_max = thresholds["max"]["data_quality"]

        # set throughput according to current env
        service_type_obj = ServiceType.CV if service_type == "cv" else ServiceType.QR
        new_state[2] = sample_throughput_from_lgbn(new_state[0], new_state[6], new_state[4], service_type_obj)

        for idx in (0, 1):
            new_state[idx] = max(data_quality_min, min(data_quality_max, new_state[idx]))

        new_state = self.min_max_scale(new_state)

        reward = (
            self.check_reward_cv(new_state)
            if service_type == "cv"
            else self.check_reward_qr(new_state)
        )

        return new_state, reward

    def check_reward_cv(self, state):
        """Lightweight reward calculation for CV"""

        rescaled_state = self.min_max_rescale(state)
        reward_cv = convert_rescaled_state_cv_to_slof(state_cv=rescaled_state)
        return reward_cv

    def check_reward_qr(self, state):
        """Lightweight reward calculation for CV"""
        rescaled_state = self.min_max_rescale(state)

        reward_qr = convert_rescaled_state_qr_to_slof(state_qr=rescaled_state)
        return reward_qr

    def adaptive_save_experience(self, obs, joint_action, next_obs, to_train=True):
        """Save experience using appropriate method based on training phase"""
        if self.current_phase == "world_model":
            # Use lightweight CPU-based storage
            action_cv, action_qr = joint_action
            one_hot_action_cv = self.transform_action_cpu(action_cv, self.action_dim_cv)
            one_hot_action_qr = self.transform_action_cpu(action_qr, self.action_dim_qr)

            sample = (
                torch.tensor(obs, dtype=torch.float32),
                one_hot_action_cv,
                one_hot_action_qr,
                torch.tensor(next_obs, dtype=torch.float32),
            )

            if to_train:
                self.buffer.append(sample)
            else:
                self.val_buffer.append(sample)
        else:
            # Use GPU-based storage for transition/all phases
            action_cv, action_qr = joint_action

            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32)
            next_obs_tensor = torch.tensor(next_obs, device=self.device, dtype=torch.float32)

            one_hot_action_cv = self.transform_action_gpu(action_cv, self.action_dim_cv)
            one_hot_action_qr = self.transform_action_gpu(action_qr, self.action_dim_qr)

            if to_train:
                self.gpu_buffer_obs.append(obs_tensor)
                self.gpu_buffer_actions_cv.append(one_hot_action_cv)
                self.gpu_buffer_actions_qr.append(one_hot_action_qr)
                self.gpu_buffer_next_obs.append(next_obs_tensor)
            else:
                self.gpu_val_buffer_obs.append(obs_tensor)
                self.gpu_val_buffer_actions_cv.append(one_hot_action_cv)
                self.gpu_val_buffer_actions_qr.append(one_hot_action_qr)
                self.gpu_val_buffer_next_obs.append(next_obs_tensor)

    def transform_action_cpu(self, actions, action_dim):
        """Lightweight CPU action transformation"""
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.long).squeeze()
        else:
            actions = actions.to(torch.long)
        return F.one_hot(actions, num_classes=action_dim).float()

    def transform_action_gpu(self, actions, action_dim):
        """GPU action transformation"""
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).squeeze()
        else:
            actions = actions.to(torch.long).to(self.device)
        return F.one_hot(actions, num_classes=action_dim).float().to(self.device)

    def adaptive_sample(self):
        """Sample batch using appropriate method based on training phase"""
        if self.current_phase == "world_model":
            # Use lightweight CPU sampling
            if len(self.buffer) < self.batch_size:
                # Handle insufficient data
                available_samples = self.buffer[:]
                while len(available_samples) < self.batch_size:
                    available_samples.extend(
                        self.buffer[:min(len(self.buffer), self.batch_size - len(available_samples))])
                samples = available_samples[:self.batch_size]
            else:
                samples = random.sample(self.buffer, self.batch_size)

            states, actions_cv, actions_qr, next_states = zip(*samples)
            return {
                "states": torch.stack(states).to(self.device, dtype=torch.float32),
                "actions_cv": torch.stack(actions_cv).to(self.device, dtype=torch.float32),
                "actions_qr": torch.stack(actions_qr).to(self.device, dtype=torch.float32),
                "next_states": torch.stack(next_states).to(self.device, dtype=torch.float32),
            }
        else:
            # Use GPU sampling for transition phases
            if len(self.gpu_buffer_obs) < self.batch_size:
                available_size = len(self.gpu_buffer_obs)
                if available_size == 0:
                    return {
                        "states": torch.randn(1, 16, device=self.device),
                        "actions_cv": torch.randn(1, self.action_dim_cv, device=self.device),
                        "actions_qr": torch.randn(1, self.action_dim_qr, device=self.device),
                        "next_states": torch.randn(1, 16, device=self.device),
                    }
                indices = torch.randint(0, available_size, (self.batch_size,), device=self.device) % available_size
            else:
                indices = torch.randint(0, len(self.gpu_buffer_obs), (self.batch_size,), device=self.device)

            return {
                "states": torch.stack([self.gpu_buffer_obs[i] for i in indices]),
                "actions_cv": torch.stack([self.gpu_buffer_actions_cv[i] for i in indices]),
                "actions_qr": torch.stack([self.gpu_buffer_actions_qr[i] for i in indices]),
                "next_states": torch.stack([self.gpu_buffer_next_obs[i] for i in indices]),
            }

    def migrate_buffer_to_gpu(self):
        """Migrate CPU buffer to GPU when transitioning phases"""
        print("Migrating buffer from CPU to GPU for transition training...")

        for obs, action_cv, action_qr, next_obs in self.buffer:
            self.gpu_buffer_obs.append(obs.to(self.device))
            self.gpu_buffer_actions_cv.append(action_cv.to(self.device))
            self.gpu_buffer_actions_qr.append(action_qr.to(self.device))
            self.gpu_buffer_next_obs.append(next_obs.to(self.device))

        for obs, action_cv, action_qr, next_obs in self.val_buffer:
            self.gpu_val_buffer_obs.append(obs.to(self.device))
            self.gpu_val_buffer_actions_cv.append(action_cv.to(self.device))
            self.gpu_val_buffer_actions_qr.append(action_qr.to(self.device))
            self.gpu_val_buffer_next_obs.append(next_obs.to(self.device))

        # Clear CPU buffers to save memory
        self.buffer.clear()
        self.val_buffer.clear()

        # Increase batch size for better GPU utilization
        # self.batch_size = min(self.batch_size, len(self.gpu_buffer_obs))
        print(f"Updated batch size to {self.batch_size} for GPU training")

    def normalize_obs(self, obs: torch.Tensor):
        return torch.clamp(obs, min=0, max=1.0)

    def compute_stats(self):
        """Compute normalization statistics for deltas"""
        delta_mus = []
        with torch.no_grad():
            buffer_size = len(self.gpu_buffer_obs) if self.current_phase != "world_model" else len(self.buffer)
            max_iterations = min(2000, buffer_size // self.batch_size + 1)

            for it in range(max_iterations):
                obs_batch = self.adaptive_sample()
                current_obs = self.normalize_obs(obs_batch["states"])
                next_obs = self.normalize_obs(obs_batch["next_states"])

                enc_mu, _ = self.world_model.encode(current_obs, sample=False)["s_dist_params"]
                enc_next_mu, _ = self.world_model.encode(next_obs, sample=False)["s_dist_params"]
                delta_mu = enc_next_mu - enc_mu
                delta_mus.append(delta_mu)

        if delta_mus:
            all_deltas = torch.cat(delta_mus, dim=0)
            self.mean_deltas = torch.mean(all_deltas, dim=0)
            self.std_deltas = torch.clamp(torch.std(all_deltas, dim=0), min=0.001)
        else:
            latent_dim = self.world_model.enc_mu.out_features
            self.mean_deltas = torch.zeros(latent_dim, device=self.device)
            self.std_deltas = torch.ones(latent_dim, device=self.device) * 0.1

    def normalize_deltas(self, deltas):
        if self.mean_deltas is None or self.std_deltas is None:
            return deltas
        return (deltas - self.mean_deltas) / self.std_deltas

    def denormalize_deltas(self, deltas):
        if self.mean_deltas is None or self.std_deltas is None:
            return deltas
        return deltas * self.std_deltas + self.mean_deltas

    def phase_transition_check(self, i, num_episodes):
        """Check if we should transition between training phases"""
        if self.current_phase == "world_model" and not self.train_transition:
            if i > self.iters_wm:  # and self.validate_enc_dec(i):
                print(f"🔄 Transitioning from world_model to transition phase at iteration {i}")
                self.current_phase = "transition"
                self.train_transition = True
                self.migrate_buffer_to_gpu()
                self.compute_stats()
                return True
        elif self.current_phase == "transition" and not self.train_all:
            if i - self.iters_wm > self.iter_tran:
                #            if self.validate_transition_model(i):
                print(f"🔄 Transitioning from transition to joint phase at iteration {i}")
                self.train_all = True
                self.start_multi = i
                self.current_phase = "joint"
                return True
        return False

    # Include efficient implementations for transition phase
    def vectorized_sample_multistep_batch(self, radius: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Vectorized batch sampling for transition training phase"""
        if len(self.gpu_buffer_obs) < self.batch_size:
            available_obs = torch.stack(self.gpu_buffer_obs) if self.gpu_buffer_obs else torch.randn(1, 16,
                                                                                                     device=self.device)
            indices = torch.randint(0, len(available_obs), (self.batch_size,), device=self.device)
            obs0_batch = available_obs[indices]
        else:
            indices = torch.randint(0, len(self.gpu_buffer_obs), (self.batch_size,), device=self.device)
            obs0_batch = torch.stack([self.gpu_buffer_obs[i] for i in indices])

        actions_cv = torch.randint(0, self.action_dim_cv, (self.batch_size, radius), device=self.device)
        actions_qr = torch.randint(0, self.action_dim_qr, (self.batch_size, radius), device=self.device)

        traj_cv, traj_qr = self.vec_env.vectorized_multistep_rollout(
            obs0_batch, actions_cv, actions_qr, radius
        )

        actions_cv_onehot = F.one_hot(actions_cv, num_classes=self.action_dim_cv).float()
        actions_qr_onehot = F.one_hot(actions_qr, num_classes=self.action_dim_qr).float()
        real_batch = torch.cat([traj_cv, traj_qr], dim=2)

        return obs0_batch, actions_cv_onehot, actions_qr_onehot, real_batch

    def vectorized_multi_step_loss(self, p_gt: float = 0.3, radius: int = 3):
        """Vectorized multi-step loss for transition training"""
        obs, actions_cv, actions_qr, next_obs = self.vectorized_sample_multistep_batch(radius)
        z, _ = self.world_model.encode(self.normalize_obs(obs), sample=False)["s_dist_params"]

        total_loss = 0.0
        joint_cur_z = z

        for t in range(radius):
            if t > 0 and random.random() < p_gt:
                joint_cur_z, _ = self.world_model.encode(
                    self.normalize_obs(next_obs[:, t - 1]), sample=False
                )["s_dist_params"]

            joint_cur_z_n = self.denormalize_deltas(joint_cur_z)
            cur_z_cv, cur_z_qr = torch.chunk(joint_cur_z_n, chunks=2, dim=1)

            dz_cv = self.transition_model_cv(cur_z_cv, actions_cv[:, t])["delta"]
            dz_qr = self.transition_model_qr(cur_z_qr, actions_qr[:, t])["delta"]
            joint_delta_z_n = torch.cat([dz_cv, dz_qr], dim=1)
            joint_cur_z = joint_cur_z_n + joint_delta_z_n

            joint_target_mu, _ = self.world_model.encode(
                self.normalize_obs(next_obs[:, t])
            )["s_dist_params"]

            if t > 0:
                joint_base_mu, _ = self.world_model.encode(
                    self.normalize_obs(next_obs[:, t - 1]), sample=False
                )["s_dist_params"]
            else:
                joint_base_mu, _ = self.world_model.encode(self.normalize_obs(obs), sample=False)["s_dist_params"]

            target_joint_delta = self.normalize_deltas(joint_target_mu - joint_base_mu)
            total_loss += F.mse_loss(joint_delta_z_n, target_joint_delta, reduction="mean")

        return total_loss / radius

    @torch.no_grad()
    def calculate_efe_policies(self, joint_obs, joint_policies):
        """EFE calculation for action selection"""
        B, H = len(joint_policies), len(joint_policies[0])
        obs = joint_obs.expand(B, -1).to(self.device, dtype=torch.float32)

        acts = torch.stack(joint_policies, 0).to(self.device, dtype=torch.float32)
        act_cv = F.one_hot(acts[:, :, 0].to(dtype=torch.long), self.action_dim_cv).float()
        act_qr = F.one_hot(acts[:, :, 1].to(dtype=torch.long), self.action_dim_qr).float()

        mu, _ = self.world_model.encode(self.normalize_obs(obs), sample=False)["s_dist_params"]
        mu_cv, mu_qr = mu.chunk(2, 1)

        mu_cv_rep = mu_cv.unsqueeze(1).expand(-1, H, -1).reshape(-1, mu_cv.size(1))
        mu_qr_rep = mu_qr.unsqueeze(1).expand(-1, H, -1).reshape(-1, mu_qr.size(1))
        d_cv = self.transition_model_cv(mu_cv_rep, act_cv.reshape(-1, self.action_dim_cv))["delta"]
        d_qr = self.transition_model_qr(mu_qr_rep, act_qr.reshape(-1, self.action_dim_qr))["delta"]
        joint_delta = torch.cat([d_cv, d_qr], 1).view(B, H, -1).sum(1)

        mu_prior = mu + self.denormalize_deltas(joint_delta)
        recon = self.world_model.decode(mu_prior, sample=True)["o_pred"]
        mu_post, logvar_post = self.world_model.encode(self.normalize_obs(recon), sample=False)["s_dist_params"]

        efe_cv, efe_qr, *_ = calculate_expected_free_energy(
            self.normalize_obs(recon),
            self.preferences_cv, self.preferences_qr,
            mu_prior, mu_post, logvar_post,
            self.transition_model_cv, self.transition_model_qr,
        )
        return efe_cv, efe_qr

    def select_joint_action(self, joint_obs, step, episode, horizon=5):
        """Action selection with EFE"""
        single_step_actions = list(
            itertools.product(range(self.action_dim_cv), range(self.action_dim_qr))
        )
        joint_policies = [
            torch.tensor(seq, dtype=torch.long)
            for seq in itertools.product(single_step_actions, repeat=horizon)
        ]

        efe_cv, efe_qr = self.calculate_efe_policies(
            torch.tensor(joint_obs, device=self.device), joint_policies=joint_policies
        )

        efe = efe_cv + efe_qr
        probs = self.power_normalize(efe)
        choosen_policy = torch.multinomial(probs, num_samples=1).item()
        actions = ["-".join(map(str, row.detach().cpu().numpy())) for row in joint_policies]
        pairs = list(zip(actions, probs.detach().cpu().numpy()))
        return joint_policies[choosen_policy].detach().cpu().numpy().tolist(), pairs

    def power_normalize(self, x: torch.Tensor, alpha: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        x_shifted = x - torch.min(x)
        x_shifted = x_shifted + eps
        x_pow = x_shifted.pow(alpha)
        return x_pow / x_pow.sum()

    # Copy remaining methods from the optimized agent with minor adaptations
    def compute_world_model_loss(self, iter: int, obs_normalized: torch.Tensor, recon_mu: torch.Tensor,
                                 mu: torch.Tensor, logvar: torch.Tensor, pred_mu_next: torch.Tensor,
                                 pred_obs_next_normalized: torch.Tensor) -> Tuple[float, WorldModelLoss]:
        obs_weight = 1
        beta = torch.clamp(torch.tensor(0.01 * iter, device=self.device), max=1)
        recon_loss = F.mse_loss(recon_mu * obs_weight, obs_normalized * obs_weight, reduction="mean")
        kl_loss = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        spread_weight = torch.clamp(torch.tensor(0.1 * iter, device=self.device), max=5)
        # spread_loss = spread_weight * torch.clamp(0.05 - torch.std(mu, dim=0), min=0).mean()
        # decoding_loss = F.mse_loss(pred_mu_next, pred_obs_next_normalized, reduction="mean")

        loss = recon_loss + kl_loss
        return loss, {
            "reconstruction loss": recon_loss.item(),
            "kl loss": kl_loss.item(),
            #            "spread loss": spread_loss.item(),
            #            "decoder loss": decoding_loss.item(),
        }

    def compute_transition_loss(self, next_latent_deltas, transition_latent_deltas, i, T, is_multi=False):
        transition_loss = F.mse_loss(transition_latent_deltas, next_latent_deltas, reduction="mean")

        if is_multi:
            multi_step_loss = self.vectorized_multi_step_loss(p_gt=(i / 2000) ** 5, radius=5)
        else:
            multi_step_loss = torch.tensor(0.0, device=self.device)

        alpha = 1
        total_loss = transition_loss + alpha * multi_step_loss
        return total_loss, {
            "transition loss": transition_loss.item(),
            "multi-step loss": multi_step_loss.item(),
        }

    def validate_enc_dec(self, i):
        """Validate encoder-decoder with appropriate buffer"""
        if self.current_phase == "world_model":
            if not self.val_buffer:
                return False

            val_size = min(len(self.val_buffer), self.batch_size)
            samples = random.sample(self.val_buffer, val_size)
            obs, actions_cv, actions_qr, next_states = zip(*samples)

            stacked_obs = torch.stack(obs).to(dtype=torch.float32, device=self.device)
            stacked_next_obs = torch.stack(next_states).to(dtype=torch.float32, device=self.device)
        else:
            if not self.gpu_val_buffer_obs:
                return False

            val_size = min(len(self.gpu_val_buffer_obs), self.batch_size)
            indices = torch.randint(0, len(self.gpu_val_buffer_obs), (val_size,), device=self.device)

            stacked_obs = torch.stack([self.gpu_val_buffer_obs[i] for i in indices])
            stacked_next_obs = torch.stack([self.gpu_val_buffer_next_obs[i] for i in indices])

        self.world_model.eval()
        with torch.no_grad():
            obs_norm = self.normalize_obs(stacked_obs)
            next_obs_norm = self.normalize_obs(stacked_next_obs)
            mu, logvar = self.world_model.encode(obs_norm, sample=False)["s_dist_params"]
            recon_mu, recon_logvar = self.world_model.decode(mu, sample=False)["o_dist_params"]

            mu_next, logvar_next = self.world_model.encode(next_obs_norm, sample=False)["s_dist_params"]
            recon_mu_next, recon_logvar_next = self.world_model.decode(mu_next, sample=False)["o_dist_params"]

            val_loss_vae, loss_dict = self.compute_world_model_loss(
                iter=i, obs_normalized=obs_norm, recon_mu=recon_mu, mu=mu, logvar=logvar,
                pred_mu_next=recon_mu_next, pred_obs_next_normalized=next_obs_norm,
            )

        self.world_model.train()
        val_loss_vae_det = val_loss_vae.detach().cpu().numpy()

        if np.round(self.val_loss_enc, decimals=4) <= np.round(val_loss_vae_det, decimals=4):
            self.patience_enc_dec += 1
        else:
            self.val_loss_enc = val_loss_vae_det
            self.patience_enc_dec = 0

        return self.patience_enc_dec > self.patience

    def validate_transition_model(self, i):
        """Validate transition model with GPU buffers"""
        if not self.gpu_val_buffer_obs:
            return False

        val_size = min(len(self.gpu_val_buffer_obs), self.batch_size)
        indices = torch.randint(0, len(self.gpu_val_buffer_obs), (val_size,), device=self.device)

        stacked_obs = torch.stack([self.gpu_val_buffer_obs[i] for i in indices])
        stacked_actions_cv = torch.stack([self.gpu_val_buffer_actions_cv[i] for i in indices])
        stacked_actions_qr = torch.stack([self.gpu_val_buffer_actions_qr[i] for i in indices])
        stacked_next_obs = torch.stack([self.gpu_val_buffer_next_obs[i] for i in indices])

        self.world_model.eval()
        self.transition_model_cv.eval()
        self.transition_model_qr.eval()

        with torch.no_grad():
            obs_norm = self.normalize_obs(stacked_obs)
            next_obs_norm = self.normalize_obs(stacked_next_obs)

            mu, logvar = self.world_model.encode(obs_norm, sample=False)["s_dist_params"]
            next_mu, next_logvar = self.world_model.encode(next_obs_norm, sample=False)["s_dist_params"]

            target_delta_mu = next_mu.detach() - mu.detach()
            norm_target_delta_mu = self.normalize_deltas(target_delta_mu)

            mu_cv_detached, mu_qr_detached = torch.chunk(mu.detach(), chunks=2, dim=1)

            pred_delta_cv = self.transition_model_cv(mu_cv_detached, stacked_actions_cv)["delta"]
            pred_delta_qr = self.transition_model_qr(mu_qr_detached, stacked_actions_qr)["delta"]

            joint_pred_delta = torch.cat([pred_delta_cv, pred_delta_qr], dim=1)

            val_loss_transition, _ = self.compute_transition_loss(
                next_latent_deltas=norm_target_delta_mu,
                transition_latent_deltas=joint_pred_delta,
                i=i, T=None, is_multi=False,
            )

        self.world_model.train()
        self.transition_model_cv.train()
        self.transition_model_qr.train()

        if np.round(self.val_loss_transition, decimals=4) <= np.round(val_loss_transition.item(), decimals=4):
            self.patience_transition += 1
        else:
            self.val_loss_transition = val_loss_transition.item()
            self.patience_transition = 0

        return self.patience_transition > self.patience

    def fit_experience(self, i, num_episodes, lambda_trans_start: float = 0.1, lambda_trans_end: float = 1.0):
        """Adaptive training based on current phase"""
        # Check for phase transitions
        self.phase_transition_check(i, num_episodes)

        num_epochs = 3
        end_training = False

        self.world_model.train()
        self.transition_model_cv.train()
        self.transition_model_qr.train()

        total_loss_dict = {
            "reconstruction loss": 0, "kl loss": 0, "spread loss": 0,
            "transition loss": 0, "decoder loss": 0, "multi-step loss": 0,
        }

        for epoch in range(num_epochs):
            # Check if we have enough data
            buffer_size = len(self.buffer) if self.current_phase == "world_model" else len(self.gpu_buffer_obs)
            if buffer_size < self.batch_size:
                continue

            obs_batch = self.adaptive_sample()
            total_loss = 0.0

            joint_current_obs = obs_batch["states"]
            joint_next_obs = obs_batch["next_states"]
            actions_cv = obs_batch["actions_cv"]
            actions_qr = obs_batch["actions_qr"]

            joint_obs_norm = self.normalize_obs(joint_current_obs)
            joint_next_obs_norm = self.normalize_obs(joint_next_obs)
            train_multi = self.current_phase == "joint"
            # train_multi = False
            if self.current_phase == "world_model":
                # Lightweight world model training
                self.optim_world_model.zero_grad()

                joint_latent_mu, joint_latent_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                # joint_latent = self.world_model.reparameterize(joint_latent_mu, joint_latent_logvar)
                joint_recon_mu, joint_recon_logvar = self.world_model.decode(joint_latent_mu, sample=False)[
                    "o_dist_params"]
                # joint_recon = self.world_model.reparameterize(joint_recon_mu, joint_recon_logvar)

                # joint_latent_next = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]
                joint_latent_next_mu, _ = self.world_model.encode(joint_next_obs_norm, sample=False)["s_dist_params"]
                recon_next_mu, recon_next_logvar = self.world_model.decode(joint_latent_next_mu, sample=False)[
                    "o_dist_params"]
                # joint_recon_next = self.world_model.reparameterize(recon_next_mu, recon_next_logvar)

                loss, loss_dict = self.compute_world_model_loss(
                    i, joint_obs_norm, joint_recon_mu, joint_latent_mu, joint_latent_logvar,
                    recon_next_mu, joint_next_obs_norm,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                self.optim_world_model.step()
                self.scheduler.step()

            elif self.current_phase == "transition":
                # Transition model training with frozen world model
                self.world_model.eval()
                for p in self.world_model.parameters():
                    p.requires_grad = False

                with torch.no_grad():
                    joint_mu, joint_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                    # joint_latent = self.world_model.reparameterize(joint_mu, joint_logvar)
                    #                    joint_next_latent = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]
                    joint_next_mu, joint_next_logvar = self.world_model.encode(joint_next_obs_norm, sample=False)[
                        "s_dist_params"]

                self.optim_transition_network.zero_grad()

                target_delta = joint_next_mu.detach() - joint_mu.detach()
                norm_target_deltas = self.normalize_deltas(target_delta)

                cv_latent_detach, qr_latent_detach = torch.chunk(joint_mu.detach(), chunks=2, dim=1)

                z_delta_pred_cv = self.transition_model_cv(cv_latent_detach, actions_cv)["delta"]
                z_delta_pred_qr = self.transition_model_qr(qr_latent_detach, actions_qr)["delta"]
                joint_deltas = torch.cat([z_delta_pred_cv, z_delta_pred_qr], dim=1)
                loss, loss_dict = self.compute_transition_loss(
                    self.denormalize_deltas(norm_target_deltas), joint_deltas, i, num_episodes, is_multi=train_multi
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model_cv.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.transition_model_qr.parameters(), max_norm=1.0)
                self.optim_transition_network.step()
                self.scheduler_trans.step()

            elif self.current_phase == "joint":
                # Joint training with all optimizations
                for p in self.world_model.parameters():
                    p.requires_grad = True
                for p in self.transition_model_cv.parameters():
                    p.requires_grad = True
                for p in self.transition_model_qr.parameters():
                    p.requires_grad = True

                self.optim_world_model.zero_grad()
                self.optim_transition_network.zero_grad()
                # World model forward pass
                joint_latent_mu, joint_latent_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                # joint_latent = self.world_model.reparameterize(joint_latent_mu, joint_latent_logvar)
                joint_recon_mu, joint_recon_logvar = self.world_model.decode(joint_latent_mu, sample=False)[
                    "o_dist_params"]
                # joint_recon = self.world_model.reparameterize(joint_recon_mu, joint_recon_logvar)

                # joint_latent_next = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]
                joint_latent_next_mu, _ = self.world_model.encode(joint_next_obs_norm, sample=False)["s_dist_params"]
                recon_next_mu, recon_next_logvar = self.world_model.decode(joint_latent_next_mu, sample=False)[
                    "o_dist_params"]
                # joint_recon_next = self.world_model.reparameterize(recon_next_mu, recon_next_logvar)

                loss_vae, wm_loss_dict = self.compute_world_model_loss(
                    i, joint_obs_norm, joint_recon_mu, joint_latent_mu, joint_latent_logvar,
                    recon_next_mu, joint_next_obs_norm,
                )

                # Transition model forward pass
                joint_delta_latent = joint_latent_next_mu.detach() - joint_latent_mu.detach()
                norm_target_deltas = self.normalize_deltas(joint_delta_latent)

                cv_latent, qr_latent = torch.chunk(joint_latent_mu, chunks=2, dim=1)
                z_delta_pred_cv = self.transition_model_cv(cv_latent, actions_cv)["delta"]
                z_delta_pred_qr = self.transition_model_qr(qr_latent, actions_qr)["delta"]
                joint_pred_deltas = torch.cat([z_delta_pred_cv, z_delta_pred_qr], dim=1)
                # joint_pred_deltas = self.normalize_deltas(joint_pred_deltas)
                # joint_pred_deltas = joint_pred_deltas
                loss_trans, trans_dict = self.compute_transition_loss(
                    norm_target_deltas, joint_pred_deltas, i, num_episodes, is_multi=True
                )

                # Additional position losses
                recon_obs_after_transition = self.world_model.decode(
                    joint_pred_deltas + joint_latent_mu, sample=True,
                    #                    joint_pred_deltas + joint_latent_mu, sample=True,
                )["o_pred"]

                scale = lambda_trans_start + (lambda_trans_end - lambda_trans_start) * (i / num_episodes)

                cv_next_obs_norm, qr_next_obs_norm = torch.chunk(joint_next_obs_norm, chunks=2, dim=1)
                cv_recon_obs_after_transition, qr_recon_obs_after_transition = torch.chunk(recon_obs_after_transition,
                                                                                           chunks=2, dim=1)

                l_tp_cv = 1 * F.mse_loss(cv_recon_obs_after_transition[:, 2], cv_next_obs_norm[:, 2])
                l_tp_qr = 1 * F.mse_loss(qr_recon_obs_after_transition[:, 2], qr_next_obs_norm[:, 2])

                cv_sol_qual_recon = 0.25 * cv_recon_obs_after_transition[:, 0] + 0.75 * cv_recon_obs_after_transition[:,
                                                                                        4]
                cv_sol_qual_next_obs = 0.25 * cv_next_obs_norm[:, 0] + 0.75 * cv_next_obs_norm[:, 4]
                l_sol_qual_cv = F.mse_loss(cv_sol_qual_recon, cv_sol_qual_next_obs)
                l_sol_qual_qr = F.mse_loss(cv_recon_obs_after_transition[:, 0], cv_next_obs_norm[:, 0])

                loss_qual = l_sol_qual_cv + l_sol_qual_qr
                loss_tp = l_tp_cv + l_tp_qr

                # loss = loss_vae + scale * loss_trans + loss_qual + loss_tp
                loss = loss_vae + scale * loss_trans + loss_qual + loss_tp

                if loss.item() < self.train_loss_finetune:
                    self.train_loss_finetune = loss.item()
                    self.patience_finetune = 0
                else:
                    self.patience_finetune += 1

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model_cv.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.transition_model_qr.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                self.optim_world_model.step()
                self.optim_transition_network.step()
                self.scheduler.step()
                self.scheduler_trans.step()

                trans_dict.update(wm_loss_dict)
                loss_dict = trans_dict
                loss_dict.update({"tp loss": loss_tp.item(), "qual loss": loss_qual.item()})

            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k in total_loss_dict:
                    total_loss_dict[k] += v / num_epochs

        avg_loss = total_loss / num_epochs

        self.world_model.eval()
        self.transition_model_cv.eval()
        self.transition_model_qr.eval()

        if self.patience_finetune >= self.patience:
            end_training = True
        if i - self.iters_wm - self.iter_tran > self.iters_joint:
            end_training = True
        return avg_loss, total_loss_dict, end_training
