import torch.nn.functional as F
import copy
import itertools
import logging
import random
from typing import Dict, Tuple

import numpy as np
import torch

from agent.agent_utils import min_max_scale
from agent.daci.aif_utils import calculate_expected_free_energy
from agent.daci.network import SimpleDeltaTransitionNetwork, SimpleMCDaciWorldModel
from agent.daci_optim.vectorized_env import VectorizedEnvironment
from torch.nn import functional as F

from iwai.proj_types import WorldModelLoss

logger = logging.getLogger("multiscale")


def freeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module_params(module):
    for param in module.parameters():
        param.requires_grad = True


class OptimizedMCDACIAgent:
    """GPU-Optimized MCDaci agent with vectorized environment operations"""

    def __init__(
            self,
            boundaries: Dict[str, Dict[str, float | int]],
            cv_slo_targets: dict,
            qr_slo_targets: dict,
            lr_wm: float = 1e-4,
            lr_tn: float = 1e-3,
            joint_obs_dim: int = 2 * 8,
            joint_latent_dim: int = 2 * 4,
            action_dim_cv: int = 7,
            action_dim_qr: int = 5,
            width: int = 48,
            batch_size: int = 32,  # Increased batch size for better GPU utilization
            early_stopping_rounds=60,
            device: str = "cuda:0",
            depth_increase: int = 0,
            train_transition_from_iter: int = 600,
    ):
        self.device = device
        self.boundaries = boundaries

        # Initialize vectorized environment
        self.vec_env = VectorizedEnvironment(boundaries, device)

        self.world_model = SimpleMCDaciWorldModel(
            in_dim=joint_obs_dim,
            world_latent_dim=joint_latent_dim,
            width=width,
            depth_increase=0,
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

        # Use tensors for buffers to keep everything on GPU
        self.buffer_obs = []
        self.buffer_actions_cv = []
        self.buffer_actions_qr = []
        self.buffer_next_obs = []

        self.val_buffer_obs = []
        self.val_buffer_actions_cv = []
        self.val_buffer_actions_qr = []
        self.val_buffer_next_obs = []

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
            self.optim_world_model, step_size=50, gamma=0.95
        )
        self.scheduler_trans = torch.optim.lr_scheduler.StepLR(
            self.optim_transition_network, step_size=50, gamma=0.95
        )

        self.train_transition_from_iter = train_transition_from_iter
        self.beta = 1

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

    @torch.no_grad()
    def calculate_efe_policies(self, joint_obs, joint_policies):
        B, H = len(joint_policies), len(joint_policies[0])
        obs = joint_obs.expand(B, -1).to(self.device, dtype=torch.float32)

        acts = torch.stack(joint_policies, 0).to(self.device, dtype=torch.float32)
        act_cv = F.one_hot(acts[:, :, 0].to(dtype=torch.long), self.action_dim_cv).float()
        act_qr = F.one_hot(acts[:, :, 1].to(dtype=torch.long), self.action_dim_qr).float()

        mu, _ = self.world_model.encode(self.normalize_obs(obs), sample=False)["s_dist_params"]
        mu_cv, mu_qr = mu.chunk(2, 1)

        # Vectorized transition prediction
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

    def select_joint_action(self, joint_obs, step, episode, horizon=3):
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

    def vectorized_sample_multistep_batch(self, radius: int) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fully vectorized batch sampling using GPU operations"""

        # Sample random starting observations from buffer
        if len(self.buffer_obs) < self.batch_size:
            # Not enough samples yet, pad with random observations
            available_obs = torch.stack(self.buffer_obs) if self.buffer_obs else torch.randn(1, 16, device=self.device)
            indices = torch.randint(0, len(available_obs), (self.batch_size,), device=self.device)
            obs0_batch = available_obs[indices]
        else:
            indices = torch.randint(0, len(self.buffer_obs), (self.batch_size,), device=self.device)
            obs0_batch = torch.stack([self.buffer_obs[i] for i in indices])

        # Sample random actions for the entire batch and horizon
        actions_cv = torch.randint(0, self.action_dim_cv, (self.batch_size, radius), device=self.device)
        actions_qr = torch.randint(0, self.action_dim_qr, (self.batch_size, radius), device=self.device)

        # Vectorized rollout using the vectorized environment
        traj_cv, traj_qr = self.vec_env.vectorized_multistep_rollout(
            obs0_batch, actions_cv, actions_qr, radius
        )

        # Convert actions to one-hot
        actions_cv_onehot = F.one_hot(actions_cv, num_classes=self.action_dim_cv).float()
        actions_qr_onehot = F.one_hot(actions_qr, num_classes=self.action_dim_qr).float()

        # Combine trajectories
        real_batch = torch.cat([traj_cv, traj_qr], dim=2)  # [batch, horizon, 16]

        return obs0_batch, actions_cv_onehot, actions_qr_onehot, real_batch

    def vectorized_multi_step_loss(self, p_gt: float = 0.3, radius: int = 3):
        """Vectorized multi-step loss computation"""
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

    def vectorized_probe_transition(self, obs_batch: torch.Tensor, actions_cv: torch.Tensor,
                                    actions_qr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized environment transition for a batch of states
        obs_batch: [batch_size, 16] joint observations
        actions_cv: [batch_size] CV actions
        actions_qr: [batch_size] QR actions
        Returns: (next_obs_batch, rewards_batch)
        """
        obs_cv, obs_qr = torch.chunk(obs_batch, 2, dim=1)

        next_obs_cv, rewards_cv = self.vec_env.vectorized_transition_cv(obs_cv, actions_cv)
        next_obs_qr, rewards_qr = self.vec_env.vectorized_transition_qr(obs_qr, actions_qr)

        next_obs_batch = torch.cat([next_obs_cv, next_obs_qr], dim=1)
        rewards_batch = rewards_cv + rewards_qr

        return next_obs_batch, rewards_batch

    def normalize_obs(self, obs: torch.Tensor):
        return torch.clamp(obs, min=0, max=1.0)

    def sample(self):
        """Sample batch from buffer, keeping everything on GPU"""
        if len(self.buffer_obs) < self.batch_size:
            # Not enough samples, return what we have
            available_size = len(self.buffer_obs)
            if available_size == 0:
                # Return dummy batch
                return {
                    "states": torch.randn(1, 16, device=self.device),
                    "actions_cv": torch.randn(1, self.action_dim_cv, device=self.device),
                    "actions_qr": torch.randn(1, self.action_dim_qr, device=self.device),
                    "next_states": torch.randn(1, 16, device=self.device),
                }

            indices = torch.randint(0, available_size, (self.batch_size,), device=self.device)
            indices = indices % available_size  # Handle case where batch_size > available_size
        else:
            indices = torch.randint(0, len(self.buffer_obs), (self.batch_size,), device=self.device)

        return {
            "states": torch.stack([self.buffer_obs[i] for i in indices]),
            "actions_cv": torch.stack([self.buffer_actions_cv[i] for i in indices]),
            "actions_qr": torch.stack([self.buffer_actions_qr[i] for i in indices]),
            "next_states": torch.stack([self.buffer_next_obs[i] for i in indices]),
        }

    def compute_stats(self):
        """Compute normalization statistics for deltas"""
        delta_mus = []
        with torch.no_grad():
            for it in range(min(500, len(self.buffer_obs) // self.batch_size + 1)):
                obs_batch = self.sample()
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
            # Default values if no data
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

    def save_experience(self, obs, joint_action, next_obs, to_train=True):
        """Save experience to buffer, keeping tensors on GPU"""
        action_cv, action_qr = joint_action

        # Ensure all tensors are on GPU
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32) if not torch.is_tensor(obs) else obs.to(
            self.device)
        next_obs_tensor = torch.tensor(next_obs, device=self.device, dtype=torch.float32) if not torch.is_tensor(
            next_obs) else next_obs.to(self.device)

        one_hot_action_cv = self.transform_action(action_cv, self.action_dim_cv)
        one_hot_action_qr = self.transform_action(action_qr, self.action_dim_qr)

        if to_train:
            self.buffer_obs.append(obs_tensor)
            self.buffer_actions_cv.append(one_hot_action_cv)
            self.buffer_actions_qr.append(one_hot_action_qr)
            self.buffer_next_obs.append(next_obs_tensor)
        else:
            self.val_buffer_obs.append(obs_tensor)
            self.val_buffer_actions_cv.append(one_hot_action_cv)
            self.val_buffer_actions_qr.append(one_hot_action_qr)
            self.val_buffer_next_obs.append(next_obs_tensor)

    def transform_action(self, actions, action_dim):
        if not torch.is_tensor(actions):
            actions = torch.tensor(actions, dtype=torch.long, device=self.device).squeeze()
        else:
            actions = actions.to(torch.long).to(self.device)

        return F.one_hot(actions, num_classes=action_dim).float().to(self.device)

    def compute_world_model_loss(self, iter: int, obs_normalized: torch.Tensor, recon_mu: torch.Tensor,
                                 mu: torch.Tensor, logvar: torch.Tensor, pred_mu_next: torch.Tensor,
                                 pred_obs_next_normalized: torch.Tensor) -> Tuple[float, WorldModelLoss]:
        obs_weight = 1
        beta = torch.clamp(torch.tensor(0.01 * iter, device=self.device), max=1)
        recon_loss = F.mse_loss(recon_mu * obs_weight, obs_normalized * obs_weight, reduction="mean")
        kl_loss = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        spread_weight = torch.clamp(torch.tensor(0.1 * iter, device=self.device), max=5)
        spread_loss = spread_weight * torch.clamp(0.05 - torch.std(mu, dim=0), min=0).mean()
        decoding_loss = F.mse_loss(pred_mu_next, pred_obs_next_normalized, reduction="mean")

        loss = recon_loss + kl_loss + spread_loss + decoding_loss
        return loss, {
            "reconstruction loss": recon_loss.item(),
            "kl loss": kl_loss.item(),
            "spread loss": spread_loss.item(),
            "decoder loss": decoding_loss.item(),
        }

    def compute_transition_loss(self, next_latent_deltas, transition_latent_deltas, i, T, is_multi=False):
        transition_loss = F.mse_loss(transition_latent_deltas, next_latent_deltas, reduction="mean")

        if is_multi:
            multi_step_loss = self.vectorized_multi_step_loss(p_gt=(i / 2000) ** 5)
        else:
            multi_step_loss = torch.tensor(0.0, device=self.device)

        alpha = 1
        total_loss = transition_loss + alpha * multi_step_loss
        return total_loss, {
            "transition loss": transition_loss.item(),
            "multi-step loss": multi_step_loss.item(),
        }

    def validate_enc_dec(self, i):
        if not self.val_buffer_obs:
            return False

        val_size = min(len(self.val_buffer_obs), self.batch_size)
        indices = torch.randint(0, len(self.val_buffer_obs), (val_size,), device=self.device)

        stacked_obs = torch.stack([self.val_buffer_obs[i] for i in indices])
        stacked_next_obs = torch.stack([self.val_buffer_next_obs[i] for i in indices])

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
        if not self.val_buffer_obs:
            return False

        val_size = min(len(self.val_buffer_obs), self.batch_size)
        indices = torch.randint(0, len(self.val_buffer_obs), (val_size,), device=self.device)

        stacked_obs = torch.stack([self.val_buffer_obs[i] for i in indices])
        stacked_actions_cv = torch.stack([self.val_buffer_actions_cv[i] for i in indices])
        stacked_actions_qr = torch.stack([self.val_buffer_actions_qr[i] for i in indices])
        stacked_next_obs = torch.stack([self.val_buffer_next_obs[i] for i in indices])

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
            if len(self.buffer_obs) < self.batch_size:
                continue

            obs_batch = self.sample()
            total_loss = 0.0

            joint_current_obs = obs_batch["states"]
            joint_next_obs = obs_batch["next_states"]
            actions_cv = obs_batch["actions_cv"]
            actions_qr = obs_batch["actions_qr"]

            joint_obs_norm = self.normalize_obs(joint_current_obs)
            joint_next_obs_norm = self.normalize_obs(joint_next_obs)

            if not self.train_all and not self.train_transition:
                # Train only encoder-decoder
                self.optim_world_model.zero_grad()

                joint_latent_mu, joint_latent_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                joint_latent = self.world_model.reparameterize(joint_latent_mu, joint_latent_logvar)
                joint_recon_mu, joint_recon_logvar = self.world_model.decode(joint_latent, sample=False)[
                    "o_dist_params"]
                joint_recon = self.world_model.reparameterize(joint_recon_mu, joint_recon_logvar)

                joint_latent_next = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]
                recon_next_mu, recon_next_logvar = self.world_model.decode(joint_latent_next, sample=False)[
                    "o_dist_params"]
                joint_recon_next = self.world_model.reparameterize(recon_next_mu, recon_next_logvar)

                loss, loss_dict = self.compute_world_model_loss(
                    i, joint_obs_norm, joint_recon, joint_latent_mu, joint_latent_logvar,
                    joint_recon_next, joint_next_obs_norm,
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                self.optim_world_model.step()
                self.scheduler.step()

#                if i > 1000 and self.validate_enc_dec(i) and not self.train_transition:
                if i > 500 and self.validate_enc_dec(i) and not self.train_transition:
                    self.train_transition = True
                    self.compute_stats()

            elif self.train_transition and not self.train_all:
                # Train only transition model
                self.world_model.eval()
                for p in self.world_model.parameters():
                    p.requires_grad = False

                with torch.no_grad():
                    joint_mu, joint_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                    joint_latent = self.world_model.reparameterize(joint_mu, joint_logvar)
                    joint_next_latent = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]

                self.optim_transition_network.zero_grad()

                target_delta = joint_next_latent.detach() - joint_latent.detach()
                norm_target_deltas = self.normalize_deltas(target_delta)

                cv_latent_detach, qr_latent_detach = torch.chunk(joint_latent.detach(), chunks=2, dim=1)

                z_delta_pred_cv = self.transition_model_cv(cv_latent_detach, actions_cv)["delta"]
                z_delta_pred_qr = self.transition_model_qr(qr_latent_detach, actions_qr)["delta"]
                joint_deltas = torch.cat([z_delta_pred_cv, z_delta_pred_qr], dim=1)

                loss, loss_dict = self.compute_transition_loss(
                    norm_target_deltas, joint_deltas, i, num_episodes, is_multi=False
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transition_model_cv.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.transition_model_qr.parameters(), max_norm=1.0)
                self.optim_transition_network.step()
                self.scheduler_trans.step()

                if i > 1000:
                    self.train_all = True
                    self.start_multi = i

            elif self.train_all:
                # Train everything together
                for p in self.world_model.parameters():
                    p.requires_grad = True

                # World model forward pass
                joint_latent_mu, joint_latent_logvar = self.world_model.encode(joint_obs_norm)["s_dist_params"]
                joint_latent = self.world_model.reparameterize(joint_latent_mu, joint_latent_logvar)
                joint_recon_mu, joint_recon_logvar = self.world_model.decode(joint_latent, sample=False)[
                    "o_dist_params"]
                joint_recon = self.world_model.reparameterize(joint_recon_mu, joint_recon_logvar)

                joint_latent_next = self.world_model.encode(joint_next_obs_norm, sample=True)["s"]
                recon_next_mu, recon_next_logvar = self.world_model.decode(joint_latent_next, sample=False)[
                    "o_dist_params"]
                joint_recon_next = self.world_model.reparameterize(recon_next_mu, recon_next_logvar)

                loss_vae, wm_loss_dict = self.compute_world_model_loss(
                    i, joint_obs_norm, joint_recon, joint_latent_mu, joint_latent_logvar,
                    joint_recon_next, joint_next_obs_norm,
                )

                # Transition model forward pass
                joint_delta_latent = joint_latent_next.detach() - joint_latent.detach()
                norm_target_deltas = self.normalize_deltas(joint_delta_latent)

                cv_latent, qr_latent = torch.chunk(joint_latent, chunks=2, dim=1)
                z_delta_pred_cv = self.transition_model_cv(cv_latent, actions_cv)["delta"]
                z_delta_pred_qr = self.transition_model_qr(qr_latent, actions_qr)["delta"]
                joint_pred_deltas = torch.cat([z_delta_pred_cv, z_delta_pred_qr], dim=1)

                loss_trans, trans_dict = self.compute_transition_loss(
                    norm_target_deltas, joint_pred_deltas, i, num_episodes, is_multi=True
                )

                # Additional losses for better reconstruction
                recon_obs_after_transition = self.world_model.decode(
                    self.denormalize_deltas(joint_pred_deltas) + joint_latent, sample=True,
                )["o_pred"]

                scale = lambda_trans_start + (lambda_trans_end - lambda_trans_start) * (i / num_episodes)

                # Position losses for better throughput and quality prediction
                cv_next_obs_norm, qr_next_obs_norm = torch.chunk(joint_next_obs_norm, chunks=2, dim=1)
                cv_recon_obs_after_transition, qr_recon_obs_after_transition = torch.chunk(recon_obs_after_transition,
                                                                                           chunks=2, dim=1)

                l_tp_cv = 5 * F.mse_loss(cv_recon_obs_after_transition[:, 2], cv_next_obs_norm[:, 2])
                l_tp_qr = 5 * F.mse_loss(qr_recon_obs_after_transition[:, 2], qr_next_obs_norm[:, 2])

                cv_sol_qual_recon = 0.25 * cv_recon_obs_after_transition[:, 0] + 0.75 * cv_recon_obs_after_transition[:,
                                                                                        4]
                cv_sol_qual_next_obs = 0.25 * cv_next_obs_norm[:, 0] + 0.75 * cv_next_obs_norm[:, 4]
                l_sol_qual_cv = F.mse_loss(cv_sol_qual_recon, cv_sol_qual_next_obs)
                l_sol_qual_qr = F.mse_loss(cv_recon_obs_after_transition[:, 0], cv_next_obs_norm[:, 0])

                loss_qual = l_sol_qual_cv + l_sol_qual_qr
                loss_tp = l_tp_cv + l_tp_qr
                loss = loss_vae + scale * loss_trans + loss_qual + loss_tp

                if loss.item() < self.train_loss_finetune:
                    self.train_loss_finetune = loss.item()
                    self.patience_finetune = 0
                else:
                    self.patience_finetune += 1

                self.optim_world_model.zero_grad()
                self.optim_transition_network.zero_grad()
                loss.backward()
                self.optim_world_model.step()
                self.optim_transition_network.step()

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

        return avg_loss, total_loss_dict, end_training

    def power_normalize(self, x: torch.Tensor, alpha: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
        x_shifted = x - torch.min(x)
        x_shifted = x_shifted + eps
        x_pow = x_shifted.pow(alpha)
        return x_pow / x_pow.sum()