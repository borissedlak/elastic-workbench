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
from torch.nn import functional as F

from proj_types import WorldModelLoss

logger = logging.getLogger("multiscale")

"""
"""


class SimpleMCDACIAgent:
    """Simplified MCDaci-Style agent without Habitual network

    Just hardcode uniform configurations to save time
    """

    def __init__(
            self,
            boundaries: Dict[str, Dict[str, float | int]],
            cv_slo_targets: dict,
            qr_slo_targets: dict,
            lr_wm: float = 1e-4,
            lr_tn: float = 1e-3,  # based on empirical results from 'stasiya transition network needs a higher start lr

            joint_obs_dim: int = 2 * 8,
            joint_latent_dim: int = 2 * 4,
            action_dim_cv: int = 5,
            action_dim_qr: int = 5,
            width: int = 32,
            batch_size: int = 32,
            early_stopping_rounds=60,
            device: str = "cpu",
            depth_increase: int = 0,
            train_transition_from_iter: int = 600,
    ):
        self.step_model_size = 1
        self.step_cores = 1
        self.world_model = SimpleMCDaciWorldModel(
            in_dim=joint_obs_dim,
            world_latent_dim=joint_latent_dim,
            width=width,
            depth_increase=0,
        )
        self.transition_model_cv = SimpleDeltaTransitionNetwork(
            joint_latent_dim // 2, action_dim_cv, width, depth_increase=depth_increase
        )

        self.transition_model_qr = SimpleDeltaTransitionNetwork(
            joint_latent_dim // 2, action_dim_qr, width, depth_increase=depth_increase
        )

        # preference are SLO targets [1.0, 1.0]
        self.patience_enc_dec = 0
        self.patience_transition = 0
        self.patience_finetune = 0
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.batch_size = batch_size
        self.patience = early_stopping_rounds
        self.buffer = []
        self.val_buffer = []
        self.val_loss_enc = np.inf
        self.val_loss_transition = np.inf
        self.train_loss_finetune = np.inf
        self.optim_world_model: torch.optim.Optimizer = torch.optim.Adam(
            params=self.world_model.parameters(),
            lr=lr_wm,
            weight_decay=1e-4,
        )
        self.optim_transition_network: torch.optim.Optimizer = torch.optim.Adam(
            params={
                "params_cv": self.transition_model_cv.parameters(),
                "params_qr": self.transition_model_qr.parameters(),
            },
            lr=lr_tn,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim_world_model, step_size=50, gamma=0.95
        )
        self.scheduler_trans = torch.optim.lr_scheduler.StepLR(
            self.optim_transition_network, step_size=50, gamma=0.95
        )
        self.train_transition_from_iter = train_transition_from_iter
        self.beta = 1
        self.mcts = None  # TODO
        self.device = device
        self.boundaries = boundaries

        self._step_data_quality_cv = 100
        self._step_data_quality_qr = 32
        self.cv_solo_targets = cv_slo_targets
        self.qr_solo_targets = qr_slo_targets
        # (data_quality, throughput)
        self.preferences_qr = (
            torch.tensor([min_max_scale(self.qr_solo_targets["data_quality"], self.boundaries["data_quality"]["min"],
                                        self.boundaries["data_quality"]["max"]),
                          min_max_scale(self.qr_solo_targets["throughput"], self.boundaries["throughput"]["min"],
                                        self.boundaries["throughput"]["max"])]).unsqueeze(0)
        )
        # todo: see whether we do quality mixture now
        # (data_quality, model_size, throughput)
        self.preferences_cv = (
            torch.tensor([min_max_scale(self.cv_solo_targets["data_quality"], self.boundaries["data_quality"]["min"],
                                        self.boundaries["data_quality"]["max"]),
                          min_max_scale(self.cv_solo_targets["model_size"], self.boundaries["model_size"]["min"],
                                        self.boundaries["model_size"]["max"]),
                          min_max_scale(self.cv_solo_targets["throughput"], self.boundaries["throughput"]["min"],
                                        self.boundaries["throughput"]["max"])
                          ]).unsqueeze(0)
        )

    def _get_feature_bounds(self):
        min_vals = np.asarray([
            self.boundaries["data_quality"]["min"],
            self.boundaries["data_quality"]["min"],
            0,
            0,
            self.boundaries["model_size"]["min"] if 'model_size' in self.boundaries else 1.0,
            self.boundaries["model_size"]["min"] if 'model_size' in self.boundaries else 1.0,
            self.boundaries["cores"]["min"],
            self.boundaries["cores"]["min"],
        ])
        max_vals = np.asarray([
            self.boundaries["data_quality"]["max"],
            self.boundaries["data_quality"]["max"],
            100,
            100,
            self.boundaries["model_size"]["max"] if 'model_size' in self.boundaries else 1.0,
            self.boundaries["model_size"]["max"] if 'model_size' in self.boundaries else 1.0,
            self.boundaries["cores"]["max"],
            self.boundaries["cores"]["max"],
        ])
        return min_vals, max_vals

    def calculate_efe_policies(self, obs, joint_policies):
        # so the point here is that in pymdp during EFE calculation AIF
        # lists ALL possible combinations of actions over the horizon T
        # this is obv. shit and people complain a lot and use all types of freestyle
        # to cut off unpromising policies
        # but I do not see a faster way of incorporating EFE in the training loop
        # w/o losing the richness of policies
        # mcdaci just emulated repeated actions over the horizon (by default 1, lol, deepness)
        obs_stacked_temp = torch.stack([obs for _ in range(len(joint_policies))], dim=0)
        total_efe = torch.zeros(len(joint_policies))

        for step in range(len(joint_policies[0])):  # horizon
            actions = [policy[step] for policy in joint_policies]  # tuple of ections (cv, qr)
            actions_cv = [tensor[0].unsqueeze(0) for tensor in actions]
            actions_qr = [tensor[1].unsqueeze(0) for tensor in actions]
            if True:  # with torch.no_grad():
                # reminder (Alireza): vq-VAE

                # 1) Encode current obs → posterior q(z|o)
                norm_obs = self.normalize_obs(obs_stacked_temp)
                joint_latent = self.world_model.encode(norm_obs)
                latent_cv, latent_qr = torch.chunk(joint_latent, chunks=2, dim=1)

                # 2) Predict prior over next latent: p(z'|z,a)
                action_oh_cv = self.transform_action(actions_cv)

                predicted_next_delta_cv = self.transition_model_cv(latent_cv, action_oh_cv)

                unnorm_predicted_next_delta_cv = self.denormalize_deltas(
                    predicted_next_delta_cv
                )

                action_oh_qr = self.transform_action(actions_qr)
                predicted_next_delta_qr = self.transition_model_qr(latent_qr, action_oh_qr)
                unnorm_predicted_next_delta_qr = self.denormalize_deltas(
                    predicted_next_delta_qr
                )
                cv_prior = unnorm_predicted_next_delta_cv + latent_cv
                qr_prior = unnorm_predicted_next_delta_qr + latent_qr
                joint_prior = torch.stack([cv_prior, qr_prior], dim=1)

                # 3) Decode prior latent → predicted observation distribution p(o'|z')
                recon_obs, recon_logvar = self.world_model.decode(joint_prior)

                recon_norm_obs = self.normalize_obs(recon_obs)

                # 4) Re‐encode predicted obs → posterior q(z'|o')
                mu_post, logvar_post = self.world_model.encode(recon_norm_obs)

            efe, ig, pv = calculate_expected_free_energy(
                recon_norm_obs,
                self.preferences_cv,
                self.preferences_qr,
                qr_prior,
                mu_post,
                logvar_post,
                self.transition_model,
            )
            total_efe += efe
            obs_stacked_temp = recon_obs
        return total_efe

    def select_action(self, obs, step, episode, horizon=3):
        # policies_cv = [
        #     torch.tensor(seq, dtype=torch.long)
        #     for seq in itertools.product(range(self.action_dim_cv), repeat=horizon)
        # ]
        single_step_actions = list(itertools.product(
            range(self.action_dim_cv),
            range(self.action_dim_qr)
        ))
        joint_policies = [
            torch.tensor(seq, dtype=torch.long)
            for seq in itertools.product(single_step_actions, repeat=horizon)
        ]

        efe = self.calculate_efe_policies(torch.tensor(obs), joint_policies=joint_policies)
        # softmax is a bit problematic here bc it amplifies the difference between
        # seemingly close values but more conservative transformation suffers from
        # exactly the opposite problem :(
        # probs = F.softmax(efe, dim=0).numpy()
        probs = self.power_normalize(efe)
        # sample the trajectory with the lowest EFE and use it
        # it is important to leave sampling here although it works different in slacio
        choosen_policy = torch.multinomial(probs, num_samples=1).item()
        actions = ["-".join(map(str, row.numpy())) for row in policies]
        pairs = list(zip(actions, probs.numpy()))
        return policies[choosen_policy].numpy().tolist(), pairs

    def min_max_scale(self, vals: np.ndarray):
        min_vals, max_vals = self._get_feature_bounds()
        scaled = np.where(
            min_vals == max_vals,
            1.0,
            (vals - min_vals) / (max_vals - min_vals)
        )
        return np.clip(scaled, 0, 1)

    def min_max_rescale(self, scaled_vals: np.ndarray):
        min_vals, max_vals = self._get_feature_bounds()
        rescaled = np.where(
            min_vals == max_vals,
            min_vals,
            scaled_vals * (max_vals - min_vals) + min_vals
        )
        return rescaled

    def compute_world_model_loss(
            self,
            iter: int,
            obs_normalized: torch.Tensor,
            recon_mu: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            pred_mu_next: torch.Tensor,
            pred_obs_next_normalized: torch.Tensor,
    ) -> WorldModelLoss:
        velocity_weight = 1
        obs_weight = torch.tensor([1.0, velocity_weight])

        beta = torch.clamp(torch.tensor(0.01 * iter), max=1)
        recon_loss = F.mse_loss(
            recon_mu * obs_weight, obs_normalized * obs_weight, reduction="mean"
        )
        kl_loss = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        spread_weight = torch.clamp(torch.tensor(0.1 * iter), max=5)
        spread_loss = (
                spread_weight * torch.clamp(0.05 - torch.std(mu, dim=0), min=0).mean()
        )
        decoding_loss = F.mse_loss(
            pred_mu_next * obs_weight,
            pred_obs_next_normalized * obs_weight,
            reduction="mean",
        )
        # logger.info(f"World Model Loss: {total_loss:.4f}")
        return {
            "reconstruction loss": recon_loss.item(),
            "kl loss": kl_loss.item(),
            "spread loss": spread_loss.item(),
            "decoder loss": decoding_loss.item(),
        }

    def sample_multistep_batch(
            self, radius: int
    ) -> Tuple[
        torch.FloatTensor, torch.LongTensor, torch.LongTensor, torch.FloatTensor
    ]:
        """Inefficient (SLOW) batch sampling

        Returns a batch of B=self.batch_size trajectories of length K. N O

        """
        obs0_list = []
        acts_cv_list = []
        acts_qr_list = []
        next_state_list = []

        for _ in range(self.batch_size):
            # 1) pick a random starting observation
            joint_obs0_tensor, _, _ = random.choice(
                self.buffer
            )  # obs0_tensor: [obs_dim]
            obs0_cv, obs0_qr = torch.chunk(joint_obs0_tensor, chunks=2, dim=1)
            obs0_cv_np = obs0_cv.cpu().numpy()  # starting point for probing transitions
            obs0_qr_np = obs0_qr.cpu().numpy()  # starting point for probing transitions

            # 2) sample K actions (integers in [0,action_dim))
            actions_cv = np.random.randint(0, self.action_dim_cv, size=radius)
            actions_qr = np.random.randint(0, self.action_dim_qr, size=radius)

            # 3) roll them out in the true dynamics
            real_steps_cv = []
            real_steps_qr = []
            cur_cv = obs0_cv_np
            # TODO: Parallelize
            for a in actions_cv:
                nxt, _ = self.probe_transition(cur_cv, None, int(a))
                real_steps_cv.append(nxt)
                cur_cv = nxt
            cur_qr = obs0_qr_np
            for a in actions_qr:
                nxt, _ = self.probe_transition(cur_qr, None, int(a))
                real_steps_qr.append(nxt)
                cur_qr = nxt

            # 4) collect
            obs0_list.append(joint_obs0_tensor)  # [obs_dim]
            one_hot_cv = F.one_hot(
                torch.from_numpy(actions_cv), num_classes=self.action_dim_cv
            ).float()
            one_hot_qr = F.one_hot(
                torch.from_numpy(actions_qr), num_classes=self.action_dim_qr
            ).float()
            acts_cv_list.append(one_hot_cv)  # [K]
            acts_qr_list.append(one_hot_qr)  # [K]
            next_states_cv = torch.stack(real_steps_cv).to(dtype=torch.float32)
            next_states_qr = torch.stack(real_steps_qr).to(dtype=torch.float32)
            next_states = torch.cat([next_states_cv, next_states_qr], dim=1)
            next_state_list.append(next_states)  # [K, obs_dim]

        # 5) stack into batch tensors
        obs0_batch = torch.stack(obs0_list, dim=0)  # [B, obs_dim]
        acts_batch_cv = torch.stack(acts_cv_list, dim=0)  # [B, K]
        acts_batch_qr = torch.stack(acts_cv_list, dim=0)  # [B, K]
        real_batch = torch.stack(next_state_list, dim=0)  # [B, K, obs_dim]

        return obs0_batch, acts_batch_cv, acts_batch_qr, real_batch

    def multi_step_loss(self, p_gt: float = 0.3, radius: int = 4, alpha=0.01):
        obs, actions_cv, actions_qr, next_obs = self.sample_multistep_batch(radius)
        z, _ = self.world_model.encode(self.normalize_obs(obs))
        total_loss = 0.0
        joint_cur_z = z
        for t in range(radius):
            if t > 0 and random.random() < p_gt:
                joint_cur_z, _ = self.world_model.encode(
                    self.normalize_obs(next_obs[:, t - 1])
                )
            joint_cur_z_n = self.denormalize_deltas(joint_cur_z)
            cur_z_cv, cur_z_qr = torch.chunk(joint_cur_z_n, chunks=2, dim=1)
            dz_cv = self.transition_model_cv(cur_z_cv, actions_cv[:, t])
            dz_qr = self.transition_model_qr(cur_z_qr, actions_qr[:, t])
            joint_delta_z_n = torch.cat([dz_cv, dz_qr], dim=1)
            joint_cur_z = joint_cur_z_n + joint_delta_z_n
            joint_target_mu, _ = self.world_model.encode(
                self.normalize_obs(next_obs[:, t])
            )
            if t > 0:
                joint_base_mu, _ = self.world_model.encode(
                    self.normalize_obs(next_obs[:, t - 1])
                )
            else:
                joint_base_mu, _ = self.world_model.encode(self.normalize_obs(obs))
            target_joint_delta = self.normalize_deltas(joint_target_mu - joint_base_mu)

            total_loss = F.mse_loss(
                joint_delta_z_n,
                target_joint_delta,
                reduction="mean",
            )

        return total_loss / radius

    def probe_transition(
            self, obs: torch.Tensor, next_obs: torch.Tensor, action: torch.Tensor, service_type: str
    ) -> Tuple[torch.Tensor, int]:
        """Emulate environment transation for a single service

        obs: [
            data_quality,
            data_quality_target,
            throughput,
            throughput_target,
            model_size
            model_size_target,
            cores,
            free_cores

        ]

        """
        new_state = self.min_max_rescale(copy.deepcopy(obs.detach().cpu().numpy()))
        step_data_quality = self._step_data_quality_cv if service_type == "cv" else self._step_data_quality_qr
        # Do nothing at 0
        if action == 0:
            pass

        if 1 <= action <= 2:
            delta_data_quality = -step_data_quality if action == 1 else step_data_quality
            new_data_quality = new_state[0] + delta_data_quality
            if not (self.boundaries["data_quality"]["min"] > new_data_quality > self.boundaries["data_quality"]["max"]):
                new_state[0] = new_data_quality

        elif 3 <= action <= 4:
            delta_cores = -self.step_cores if action == 3 else self.step_cores
            new_cores = new_state[6] + delta_cores
            if not (new_cores <= 0 and delta_cores > new_state[7]):
                new_state[6] = new_cores
                new_state[7] = new_state[6] - delta_cores

        elif 5 <= action <= 6:
            # step size is always 1
            delta_model = -self.step_model_size if action == 5 else self.step_model_size
            new_model_s = new_state[4] + delta_model

            if not (
                    self.boundaries["model_size"]["min"] > new_model_s > self.boundaries["model_size"]["max"]
            ):
                new_state[4] = new_model_s
        if service_type == "cv":
            reward = self.check_reward_cv(new_state)
        else:
            reward = self.check_reward_qr(new_state)

        return new_state, reward

    def check_reward_cv(self, state):
        # data quality bias TODO: check if we use quality mixture now
        if state[0] >= self.preferences_qr[0][0].item():
            return 1
        else:
            return -1

    def check_reward_qr(self, state):
        # data quality bias
        if state[0] >= self.preferences_qr[0][0].item():
            return 1
        else:
            return -1

    def normalize_obs(self, obs: torch.Tensor):
        return torch.clamp(obs, min=0, max=1.0)

    def sample(self):
        samples = random.sample(self.buffer, self.batch_size)
        states, actions, next_states = zip(*samples)
        return {
            "states": torch.stack(states).to(self.device),
            "actions": torch.stack(actions).to(self.device),
            "next_states": torch.stack(next_states).to(self.device),
        }

    def compute_stats(self):
        delta_mus = []
        with torch.no_grad():
            for it in range(2000):
                obs_batch = self.sample()
                current_obs = obs_batch["states"]
                next_obs = obs_batch["next_states"]
                current_obs = self.normalize_obs(obs_batch["states"])
                next_obs = self.normalize_obs(obs_batch["next_states"])
                enc_mu, enc_logvar = self.world_model.encode(current_obs)
                enc_next_mu, enc_next_logvar = self.world_model.encode(next_obs)
                delta_mu = enc_next_mu - enc_mu
                delta_mus.append(delta_mu)
        all_deltas = torch.cat(delta_mus, dim=0)
        self.mean_deltas = torch.mean(all_deltas, dim=0)
        self.std_deltas = torch.clamp(torch.std(all_deltas, dim=0), min=0.001)

    def normalize_deltas(self, deltas):
        return (deltas - self.mean_deltas) / self.std_deltas

    def denormalize_deltas(self, deltas):
        return deltas * self.std_deltas + self.mean_deltas

    def validate_enc_dec(self, i):
        obs, actions, next_states = zip(*self.val_buffer)
        stacked_obs = torch.stack(obs)
        stacked_next_obs = torch.stack(next_states)
        self.world_model.eval()
        with torch.no_grad():
            if True:
                obs_norm = self.normalize_obs(stacked_obs)
                next_obs_norm = self.normalize_obs(stacked_next_obs)
                mu, logvar = self.world_model.encode(obs_norm)
                recon_mu, recon_logvar = self.world_model.decode(mu)

                mu_next, logvar_next = self.world_model.encode(next_obs_norm)
                recon_mu_next, recon_logvar_next = self.world_model.decode(mu_next)

                loss_dict = self.compute_world_model_loss(
                    iter=i,
                    obs_normalized=obs_norm,
                    recon_mu=recon_mu,
                    mu=mu,
                    logvar=logvar,
                    pred_mu_next=recon_mu_next,
                    pred_obs_next_normalized=next_obs_norm,
                )
                val_loss_vae = sum(loss_dict.values())
        self.world_model.train()
        if np.round(self.val_loss_enc, decimals=4) <= np.round(
                val_loss_vae, decimals=4
        ):
            self.patience_enc_dec += 1
        else:
            self.val_loss_enc = val_loss_vae
            self.patience_enc_dec = 0
        if self.patience_enc_dec > self.patience:
            return True
        else:
            return False

    def compute_transition_loss(
            self, next_mu_delta, transition_mu_delta, i, is_multi=False
    ):
        transition_loss = F.mse_loss(
            transition_mu_delta, next_mu_delta, reduction="mean"
        )
        if is_multi:
            multi_step_loss = self.multi_step_loss(p_gt=(self.start_multi / i) ** 5)
        else:
            multi_step_loss = torch.tensor(0)
        alpha = 1
        total_loss = transition_loss + alpha * multi_step_loss
        print(f"transition loss {transition_loss:.5f}")
        return total_loss, {
            "transition loss": transition_loss.item(),
            "multi-step loss": multi_step_loss.item(),
        }

    def validate_transition_model(self, i):
        # check if the model stopped improving
        obs, actions, next_states = zip(*self.val_buffer)
        stacked_obs = torch.stack(obs)
        stacked_next_obs = torch.stack(next_states)
        stacked_actions = torch.stack(actions)
        self.world_model.eval()
        self.transition_model_qr.eval()
        with torch.no_grad():
            if True:
                obs_norm = self.normalize_obs(stacked_obs)
                next_obs_norm = self.normalize_obs(stacked_next_obs)
                mu, logvar = self.world_model.encode(obs_norm)
                next_mu, next_logvar = self.world_model.encode(next_obs_norm)
                delta_mu = next_mu.detach() - mu.detach()
                norm_deltas = self.normalize_deltas(delta_mu)
                pred_delta = self.transition_model_qr(mu, stacked_actions)
                val_loss_transition, _ = self.compute_transition_loss(
                    next_mu_delta=norm_deltas,
                    transition_mu_delta=pred_delta,
                    i=i,
                    is_multi=False,
                )

        self.world_model.train()
        self.transition_model_qr.train()
        if np.round(self.val_loss_transition, decimals=4) <= np.round(
                val_loss_transition, decimals=4
        ):
            self.patience_transition += 1
        else:
            self.val_loss_transition = val_loss_transition
            self.patience_transition = 0
        if self.patience_transition > self.patience:
            return True
        else:
            return False

    def transform_action(self, actions):
        if not torch.is_tensor(actions):
            actions = torch.tensor(
                actions, dtype=torch.long, device=self.device
            ).squeeze(0)
        else:
            actions = actions.to(torch.long).to(self.device)

        one_hot = F.one_hot(actions, num_classes=self.action_dim_qr).float()
        return one_hot

    def save_experience(self, obs, action, next_obs, to_train=True):
        one_hot_action = self.transform_action(action)
        sample = (
            torch.tensor(obs),
            one_hot_action,
            torch.tensor(next_obs),
        )
        if to_train:
            self.buffer.append(sample)
        else:
            self.val_buffer.append(sample)

    def fit_experience(
            self,
            i,
            num_episodes,
            lambda_trans_start: float = 0.1,
            lambda_trans_end: float = 1.0,
    ):
        num_epochs = 3
        end_training = False
        self.world_model.train()
        self.transition_model_qr.train()
        total_loss_dict = {
            "reconstruction loss": 0,
            "kl loss": 0,
            "spread loss": 0,
            "transition loss": 0,
            "decoder loss": 0,
            "multi-step loss": 0,
            "position loss": 0,
        }

        for epoch in range(num_epochs):
            obs_batch = self.sample()
            total_loss = 0.0
            current_obs = obs_batch["states"]
            next_obs = obs_batch["next_states"]
            actions = obs_batch["actions"]
            if True:
                obs_norm = self.normalize_obs(current_obs)
                next_obs_norm = self.normalize_obs(next_obs)
                if not self.train_all and not self.train_transition:
                    # train only encoder-decoder until they are kinda ok
                    self.optim_world_model.zero_grad()
                    mu, logvar = self.world_model.encode(obs_norm)
                    recon_mu, recon_logvar = self.world_model.decode(mu)
                    next_mu, next_logvar = self.world_model.encode(next_obs_norm)
                    recon_next_mu, recon_next_logvar = self.world_model.decode(next_mu)
                    loss_dict = self.compute_world_model_loss(
                        i,
                        obs_norm,
                        recon_mu,
                        mu,
                        logvar,
                        recon_next_mu,
                        next_obs_norm,
                    )

                    loss = sum(loss_dict.values())

                    loss.backward()
                    self.world_model.optimizer.step()
                    self.scheduler.step()
                    if (
                            i > 600
                            and self.validate_enc_dec(i)
                            and not self.train_transition
                    ):
                        # 600 does not allow transition model to start training too early, acts like a guard
                        self.train_transition = True
                        self.compute_stats()

                elif self.train_transition:
                    self.world_model.eval()
                    for p in self.world_model.parameters():
                        p.requires_grad = False
                    with torch.no_grad():
                        mu, logvar = self.world_model.encode(obs_norm)
                        next_mu, next_logvar = self.world_model.encode(next_obs_norm)

                    self.optim_transition_network.zero_grad()
                    mu_detached = mu.detach()
                    next_mu_detached = next_mu.detach()
                    target_mu = next_mu_detached - mu_detached
                    norm_deltas = self.normalize_deltas(target_mu)
                    z_delta_pred = self.transition_model_qr(mu_detached, actions)
                    if not self.train_all:
                        loss, loss_dict = self.compute_transition_loss(
                            norm_deltas, z_delta_pred, i, num_episodes
                        )
                        loss.backward()
                        self.optim_transition_network.step()
                        self.scheduler_trans.step()
                        if self.validate_transition_model(i):
                            self.train_all = True
                            self.start_multi = i
                    else:
                        # train all together
                        for p in self.world_model.parameters():
                            p.requires_grad = True
                        for m in (
                                self.world_model.encode,
                                self.world_model.decode,
                                self.transition_model_qr,
                        ):
                            m.train()

                        mu, logvar = self.world_model.encode(obs_norm)
                        recon_mu, recon_logvar = self.world_model.decode(mu)

                        next_mu, next_logvar = self.world_model.encode(next_obs_norm)
                        recon_next_mu, recon_next_logvar = self.world_model.decode(
                            next_mu
                        )

                        vae_loss_dict = self.compute_world_model_loss(
                            iter=i,
                            obs_normalized=obs_norm,
                            recon_mu=recon_mu,
                            mu=mu,
                            logvar=logvar,
                            pred_mu_next=recon_next_mu,
                            pred_obs_next_normalized=next_obs_norm,
                        )
                        loss_vae = sum(vae_loss_dict.values())

                        delta_mu = next_mu.detach() - mu.detach()
                        norm_delta = self.normalize_deltas(delta_mu)

                        pred_delta = self.transition_model_qr(mu, actions)
                        loss_trans, trans_dict = self.compute_transition_loss(
                            norm_delta, pred_delta, i, is_multi=True
                        )

                        recon_after_transition_mu, _ = self.world_model.decode(
                            self.denormalize_deltas(pred_delta) + mu
                        )

                        scale = lambda_trans_start + (
                                lambda_trans_end - lambda_trans_start
                        ) * (i / num_episodes)

                        # Lpos tries to enforce better position prediction, bc that's what we care about in the end
                        #  TODO: replace with es env obj
                        l_pos = 10 * F.mse_loss(
                            recon_after_transition_mu[:, 0], next_obs_norm[:, 0]
                        )

                        # joint loss
                        loss = loss_vae + scale * loss_trans + l_pos
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
                        trans_dict.update(vae_loss_dict)
                        loss_dict = trans_dict
                        loss_dict.update({"position loss": l_pos.item()})
                else:
                    exit(-1)

                total_loss += loss.item()
                for k, v in loss_dict.items():
                    total_loss_dict[k] += loss_dict[k] / num_epochs

            print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")

        avg_loss = total_loss / num_epochs

        self.world_model.eval()
        self.transition_model_qr.eval()
        if self.patience_finetune >= self.patience:
            end_training = True
        return avg_loss, total_loss_dict, end_training

    def pretend_a_trajectory(self, obs, trajectory, use_transition=False):
        # just visualize a trajectory to test encoder/decoder and transition model
        actual_obs_list = []
        recon_obs_list = []
        next_obs = [obs]
        imagined_obs = obs
        with torch.no_grad():
            for action in trajectory:
                actual_obs_list.append(obs)
                norm_obs = self.normalize_obs(torch.tensor(imagined_obs).unsqueeze(0))
                enc_mu, enc_var = self.world_model.encode(norm_obs)
                dec_mu, dec_var = self.world_model.decode(enc_mu)
                recon_obs = dec_mu
                recon_obs_list.append(recon_obs.squeeze(0))
                obs, _ = self.probe_transition(obs, None, action)
                if not use_transition:
                    imagined_obs = obs
                else:
                    one_hot_action = self.transform_action(action)
                    trans_mu = self.transition_model_qr(
                        enc_mu, one_hot_action.unsqueeze(0)
                    )
                    unnorm_trans_mu = self.denormalize_deltas(trans_mu)
                    norm_dec_next_obs_mu, norm_dec_next_obs_logvar = (
                        self.world_model.decode(unnorm_trans_mu + enc_mu)
                    )
                    imagined_obs = np.array(norm_dec_next_obs_mu.squeeze(0))
                    next_obs.append(imagined_obs)
        if use_transition:
            actual_obs_list.append(obs)
        return actual_obs_list, recon_obs_list, next_obs
