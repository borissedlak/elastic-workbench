import os

import torch
import numpy as np

from agent.SLORegistry import SLO_Registry, to_normalized_slo_f, calculate_slo_fulfillment
from agent.agent_utils import FullStateDQN
from agent.es_registry import ESRegistry, ServiceType

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


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def entropy_normal_from_logvar(logvar):
    constant = torch.tensor(
        np.log(2 * np.pi * np.e), dtype=logvar.dtype, device=logvar.device
    )
    return 0.5 * (constant + logvar)


def convert_rescaled_state_qr_to_slof(state_qr: torch.Tensor):
    """
    state_qr: torch.Tensor of shape (B, 8)
    Returns the mean normalized SLO-F values for the batch.
    """
    normalized_slo_qr_list = []
    state_qr_np = state_qr.detach().cpu().numpy()
    for b in range(state_qr_np.shape[0]):
        s = state_qr_np[b]
        full_state_qr = FullStateDQN(
            s[0], s[1], s[2], s[3], s[4], s[5],
            0, 0,  # cores irrelevant for SLO-F
            boundaries_qr,
        )
        normalized_slo_qr = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_qr.to_normalized_dict(), client_slos_qr),
            client_slos_qr)
        normalized_slo_qr_list.append(normalized_slo_qr)
    #    batch_mean_qr = torch.mean(torch.tensor(normalized_slo_qr_list, device=state_qr.device, dtype=torch.float32), dim=0)
    batch_mean_qr = torch.tensor(normalized_slo_qr_list, device=state_qr.device, dtype=torch.float32)
    return batch_mean_qr


def convert_rescaled_state_cv_to_slof(state_cv: torch.Tensor):
    """
    state_cv: torch.Tensor of shape (B, 8)
    Returns the mean normalized SLO-F values for the batch.
    """
    normalized_slo_cv_list = []
    state_cv_np = state_cv.detach().cpu().numpy()
    for b in range(state_cv_np.shape[0]):
        s = state_cv_np[b]
        full_state_cv = FullStateDQN(
            s[0], s[1], s[2], s[3], s[4], s[5],
            0, 0,  # cores irrelevant for SLO-F
            boundaries_cv,
        )
        normalized_slo_cv = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_cv.to_normalized_dict(), client_slos_cv),
            client_slos_cv)
        normalized_slo_cv_list.append(normalized_slo_cv)
    #    batch_mean_cv = torch.mean(torch.tensor(normalized_slo_cv_list, device=state_cv.device, dtype=torch.float32), dim=0)
    batch_mean_cv = torch.tensor(normalized_slo_cv_list, device=state_cv.device, dtype=torch.float32)
    return batch_mean_cv


import os

import torch
import numpy as np

from agent.SLORegistry import SLO_Registry, to_normalized_slo_f, calculate_slo_fulfillment
from agent.agent_utils import FullStateDQN
from agent.es_registry import ESRegistry, ServiceType

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


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def entropy_normal_from_logvar(logvar):
    constant = torch.tensor(
        np.log(2 * np.pi * np.e), dtype=logvar.dtype, device=logvar.device
    )
    return 0.5 * (constant + logvar)


def convert_rescaled_state_qr_to_slof(state_qr: torch.Tensor):
    """
    state_qr: torch.Tensor of shape (B, 8)
    Returns the mean normalized SLO-F values for the batch.
    """
    normalized_slo_qr_list = []
    state_qr_np = state_qr.detach().cpu().numpy()
    for b in range(state_qr_np.shape[0]):
        s = state_qr_np[b]
        full_state_qr = FullStateDQN(
            s[0], s[1], s[2], s[3], s[4], s[5],
            0, 0,  # cores irrelevant for SLO-F
            boundaries_qr,
        )
        normalized_slo_qr = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_qr.to_normalized_dict(), client_slos_qr),
            client_slos_qr)
        normalized_slo_qr_list.append(normalized_slo_qr)
    #    batch_mean_qr = torch.mean(torch.tensor(normalized_slo_qr_list, device=state_qr.device, dtype=torch.float32), dim=0)
    batch_mean_qr = torch.tensor(normalized_slo_qr_list, device=state_qr.device, dtype=torch.float32)
    return batch_mean_qr


def convert_rescaled_state_cv_to_slof(state_cv: torch.Tensor):
    """
    state_cv: torch.Tensor of shape (B, 8)
    Returns the mean normalized SLO-F values for the batch.
    """
    normalized_slo_cv_list = []
    state_cv_np = state_cv.detach().cpu().numpy()
    for b in range(state_cv_np.shape[0]):
        s = state_cv_np[b]
        full_state_cv = FullStateDQN(
            s[0], s[1], s[2], s[3], s[4], s[5],
            0, 0,  # cores irrelevant for SLO-F
            boundaries_cv,
        )
        normalized_slo_cv = to_normalized_slo_f(
            calculate_slo_fulfillment(full_state_cv.to_normalized_dict(), client_slos_cv),
            client_slos_cv)
        normalized_slo_cv_list.append(normalized_slo_cv)
    #    batch_mean_cv = torch.mean(torch.tensor(normalized_slo_cv_list, device=state_cv.device, dtype=torch.float32), dim=0)
    batch_mean_cv = torch.tensor(normalized_slo_cv_list, device=state_cv.device, dtype=torch.float32)
    return batch_mean_cv


def calculate_expected_free_energy(
        joint_recon_norm_obs,
        preferences_cv,
        preferences_qr,
        joint_mu_prior,
        joint_mu_post,
        joint_logvar_post,
        transition_model_cv,
        transition_model_qr,
):
    """
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
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)

    salient_feat_cv = torch.empty(recon_norm_obs_cv.shape[0], 2, device=joint_recon_norm_obs.device)
    salient_feat_qr = torch.empty(recon_norm_obs_qr.shape[0], 2, device=joint_recon_norm_obs.device)
    # interpolate solution quality, extract throughput
    salient_feat_cv[:, 0] = (
            recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    )
    salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 3]
    # extract data quality (<=> solution quality), throughpuit
    salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]
    salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 3]

    # variance per target (we have two)

    # @Boris: You can bias towards an objective by changing the variance (the higher the variance, the
    #  more relaxed the objective is for an SLO).
    #  However, I'm already biasing the reward towards sol quality
    var_qr, var_cv = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device), torch.tensor([0.1, 0.1],
                                                                                                device=joint_recon_norm_obs.device)
    var_joint = torch.cat([var_qr, var_cv], dim=1)
    salient_feat_join = torch.cat([salient_feat_cv, salient_feat_qr])
    preferences_joint = torch.cat([preferences_cv, preferences_qr])

    prag_val_joint = -0.5 * (
            (salient_feat_join.detach() - preferences_joint) ** 2 / var_joint + torch.log(var_joint)
    ).sum(dim=1)
    # a fixed prior bc transition network does not output logvars, 0.1 is mid, not too strict, not too free
    joint_fixed_prior_logvar = torch.ones_like(joint_mu_prior) * np.log(0.1)

    # 6) Information‐gain term: KL[q(z'|o') ∥ p(z'|z,a)]
    ig_joint = torch.sum(
        0.5
        * (
                joint_fixed_prior_logvar
                - joint_logvar_post.detach()
                + (joint_logvar_post.detach().exp() + (joint_mu_post.detach() - joint_mu_prior.detach()).pow(2))
                / joint_fixed_prior_logvar.exp()
                - 1
        ),
        dim=1,
    )

    efe = -ig_joint - prag_val_joint
    return efe, prag_val_joint


def calculate_expected_free_energy_enhanced(
        joint_recon_norm_obs,
        preferences_cv,
        preferences_qr,
        joint_mu_prior,
        joint_mu_post,
        joint_logvar_post,
        transition_model_cv,
        transition_model_qr,
        balance_weight=1.0,
        utilization_weight=0.5,
):
    """
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
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)

    salient_feat_cv = torch.empty(recon_norm_obs_cv.shape[0], 2, device=joint_recon_norm_obs.device)
    salient_feat_qr = torch.empty(recon_norm_obs_qr.shape[0], 2, device=joint_recon_norm_obs.device)

    # Extract features
    salient_feat_cv[:, 0] = (
            recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    )
    salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 2]  # throughput
    salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]  # data quality
    salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 2]  # throughput

    # Variance parameters
    var_qr = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device)
    var_cv = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device)
    var_joint = torch.cat([var_cv, var_qr], dim=0)
    salient_feat_joint = torch.cat([salient_feat_cv, salient_feat_qr], dim=1)
    preferences_joint = torch.cat([preferences_cv, preferences_qr], dim=1)

    # Pragmatic value
    prag_val_joint = -0.5 * (
            (salient_feat_joint.detach() - preferences_joint) ** 2 / var_joint + torch.log(var_joint)
    ).sum(dim=1)

    # Information gain
    joint_fixed_prior_logvar = torch.ones_like(joint_mu_prior) * np.log(0.1)
    ig_joint = torch.sum(
        0.5 * (
                joint_fixed_prior_logvar
                - joint_logvar_post.detach()
                + (joint_logvar_post.detach().exp() + (joint_mu_post.detach() - joint_mu_prior.detach()).pow(2))
                / joint_fixed_prior_logvar.exp()
                - 1
        ),
        dim=1,
    )

    # NEW: Equilibrium term
    # Calculate normalized fulfillment for each service
    cv_targets = torch.stack([recon_norm_obs_cv[:, 1], recon_norm_obs_cv[:, 3]], dim=1)  # targets
    qr_targets = torch.stack([recon_norm_obs_qr[:, 1], recon_norm_obs_qr[:, 3]], dim=1)  # targets

    cv_fulfillment = 1.0 - torch.abs(salient_feat_cv - cv_targets).mean(dim=1)
    qr_fulfillment = 1.0 - torch.abs(salient_feat_qr - qr_targets).mean(dim=1)

    # Penalty for imbalance (higher is better, so negative when imbalanced)
    imbalance = torch.abs(cv_fulfillment - qr_fulfillment)
    equilibrium_term = -0.5 * (imbalance ** 2) / 0.1

    # Utilization term
    cores_cv = recon_norm_obs_cv[:, 6]
    cores_qr = recon_norm_obs_qr[:, 6]
    free_cores = recon_norm_obs_cv[:, 7]

    total_cores = cores_cv + cores_qr + free_cores
    utilization_rate = (cores_cv + cores_qr) / (total_cores + 1e-8)

    # Reward high utilization (log probability of beta distribution)
    # Using Beta(2, 1) to favor high utilization
    utilization_term = torch.log(utilization_rate + 1e-8) * 2.0

    # Combined EFE (negative because we minimize)
    efe = ig_joint -(
            prag_val_joint
            + balance_weight * equilibrium_term
            + utilization_weight * utilization_term
    )

    # Return additional components for debugging
    return efe, {
        "pragmatic_value": prag_val_joint.mean().item(),
        "information_gain": ig_joint.mean().item(),
        "equilibrium_term": equilibrium_term.mean().item(),
        "utilization_term": utilization_term.mean().item(),
        "utilization_rate": utilization_rate.mean().item(),
    }

def calculate_expected_free_energy_eh(
        joint_recon_norm_obs,
        preferences_cv,
        preferences_qr,
        joint_mu_prior,
        joint_mu_post,
        joint_logvar_post,
        transition_model_cv,
        transition_model_qr,
):
    """
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
    # todo: extract throughput and accuracy stuff from recon obs (split
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)
    pragmatic_value_cv = convert_rescaled_state_cv_to_slof(recon_norm_obs_cv)
    pragmatic_value_qr = convert_rescaled_state_qr_to_slof(recon_norm_obs_qr)

    # salient_feat_cv = torch.empty(recon_norm_obs_cv.shape[0], 2, device=joint_recon_norm_obs.device)
    # salient_feat_qr = torch.empty(recon_norm_obs_qr.shape[0], 2, device=joint_recon_norm_obs.device)
    # # interpolate solution quality, extract throughput
    # salient_feat_cv[:, 0] = (
    #         recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    # )
    # salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 3]
    # # extract data quality (<=> solution quality), throughpuit
    # salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]
    # salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 3]

    # variance per target (we have two)

    # @Boris: You can bias towards an objective by changing the variance (the higher the variance, the
    #  more relaxed the objective is for an SLO).
    #  However, I'm already biasing the reward towards sol quality
    # var_qr, var_cv = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device), torch.tensor([0.1, 0.1],
    #                                                                                             device=joint_recon_norm_obs.device)
    # pragmatic_value_cv = -0.5 * (
    #         (salient_feat_cv.detach() - preferences_cv) ** 2 / var_cv + torch.log(var_cv)
    # ).sum(dim=1)
    # pragmatic_value_qr = -0.5 * (
    #         (salient_feat_qr.detach() - preferences_qr) ** 2 / var_qr + torch.log(var_qr)
    # ).sum(dim=1)

    # a fixed prior bc transition network does not output logvars, 0.1 is mid, not too strict, not too free
    joint_fixed_prior_logvar = torch.ones_like(joint_mu_prior) * np.log(0.1)
    mu_prior_cv, mu_prior_qr = torch.chunk(joint_mu_prior, chunks=2, dim=1)

    fixed_prior_logvar_cv, fixed_prior_logvar_qr = torch.chunk(joint_fixed_prior_logvar, chunks=2, dim=1)
    mu_post_cv, mu_post_qr = torch.chunk(joint_mu_post, chunks=2, dim=1)
    logvar_post_cv, logvar_post_qr = torch.chunk(joint_logvar_post, chunks=2, dim=1)

    # WIP: basically I was trying to come up with the curiosity term from mcdaci
    # bc IG is not IGaining rn

    # N = len(recon_norm_obs)
    # ig_theta = torch.zeros(N, device=mu_prior.device)
    # D_latent = mu_prior.shape[1]

    # for i in range(N):
    #     mu_prior_i = mu_prior[i : i+1]
    #     sqnorm = torch.tensor(0.0, device=mu_prior.device)

    #     for d in range(D_latent):
    #         # Take the single scalar mu_prior_i[0,d]:
    #         scalar = mu_prior_i[0, d]
    #         # Compute gradient of that scalar w.r.t. all θ
    #         grads = torch.autograd.grad(
    #             outputs=scalar,
    #             inputs=list(transition_model.parameters()),
    #             retain_graph=True,     # need to keep the graph for the next d
    #             create_graph=False,
    #             allow_unused=False
    #         )
    #         # Sum up squared entries of those grads:
    #         for g in grads:
    #             sqnorm = sqnorm + (g.detach() ** 2).sum()

    #     # Divide by (2 * sigma_prior^2) to get nats.  e.g. sigma_prior_sq = 1.0
    #     sigma_prior_sq = 0.0005
    #     ig_theta[i] = 0.5 * (sqnorm / sigma_prior_sq)

    # 6) Information‐gain term: KL[q(z'|o') ∥ p(z'|z,a)]
    ig_cv = torch.sum(
        0.5
        * (
                fixed_prior_logvar_cv
                - logvar_post_cv.detach()
                + (logvar_post_cv.detach().exp() + (mu_post_cv.detach() - mu_prior_cv.detach()).pow(2))
                / fixed_prior_logvar_cv.exp()
                - 1
        ),
        dim=1,
    )
    ig_qr = torch.sum(
        0.5
        * (
                fixed_prior_logvar_qr
                - logvar_post_qr.detach()
                + (logvar_post_qr.detach().exp() + (mu_post_qr.detach() - mu_prior_qr.detach()).pow(2))
                / fixed_prior_logvar_qr.exp()
                - 1
        ),
        dim=1,
    )

    # entropy is omitted compared to mcdaci bc logvars are not used
    # entropy_decoded_mean = entropy_normal_from_logvar(fixed_prior_logvar).sum(
    #     dim=1
    # )
    # entropy_decoded_sampled = entropy_normal_from_logvar(logvar_post).sum(dim=1)
    # entr = entropy_decoded_sampled - entropy_decoded_mean

    efe_cv = -ig_cv - pragmatic_value_cv  # - ig_theta
    efe_qr = -ig_qr - pragmatic_value_qr  # - ig_theta
    return efe_cv, efe_qr, ig_cv, ig_qr, pragmatic_value_cv, pragmatic_value_qr


def calculate_expected_free_energy_cls(
        joint_recon_norm_obs,
        preferences_cv,
        preferences_qr,
        joint_mu_prior,
        joint_mu_post,
        joint_logvar_post,
        transition_model_cv,
        transition_model_qr,
):
    """
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
    # todo: extract throughput and accuracy stuff from recon obs (split
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)

    salient_feat_cv = torch.empty(recon_norm_obs_cv.shape[0], 2, device=joint_recon_norm_obs.device)
    salient_feat_qr = torch.empty(recon_norm_obs_qr.shape[0], 2, device=joint_recon_norm_obs.device)
    # interpolate solution quality, extract throughput
    salient_feat_cv[:, 0] = (
            recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    )
    salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 3]
    # extract data quality (<=> solution quality), throughpuit
    salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]
    salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 3]

    # variance per target (we have two)

    # @Boris: You can bias towards an objective by changing the variance (the higher the variance, the
    #  more relaxed the objective is for an SLO).
    #  However, I'm already biasing the reward towards sol quality
    var_qr, var_cv = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device), torch.tensor([0.1, 0.1],
                                                                                                device=joint_recon_norm_obs.device)
    pragmatic_value_cv = -0.5 * (
            (salient_feat_cv.detach() - preferences_cv) ** 2 / var_cv + torch.log(var_cv)
    ).sum(dim=1)
    pragmatic_value_qr = -0.5 * (
            (salient_feat_qr.detach() - preferences_qr) ** 2 / var_qr + torch.log(var_qr)
    ).sum(dim=1)

    # a fixed prior bc transition network does not output logvars, 0.1 is mid, not too strict, not too free
    joint_fixed_prior_logvar = torch.ones_like(joint_mu_prior) * np.log(0.1)
    mu_prior_cv, mu_prior_qr = torch.chunk(joint_mu_prior, chunks=2, dim=1)

    fixed_prior_logvar_cv, fixed_prior_logvar_qr = torch.chunk(joint_fixed_prior_logvar, chunks=2, dim=1)
    mu_post_cv, mu_post_qr = torch.chunk(joint_mu_post, chunks=2, dim=1)
    logvar_post_cv, logvar_post_qr = torch.chunk(joint_logvar_post, chunks=2, dim=1)

    # WIP: basically I was trying to come up with the curiosity term from mcdaci
    # bc IG is not IGaining rn

    # N = len(recon_norm_obs)
    # ig_theta = torch.zeros(N, device=mu_prior.device)
    # D_latent = mu_prior.shape[1]

    # for i in range(N):
    #     mu_prior_i = mu_prior[i : i+1]
    #     sqnorm = torch.tensor(0.0, device=mu_prior.device)

    #     for d in range(D_latent):
    #         # Take the single scalar mu_prior_i[0,d]:
    #         scalar = mu_prior_i[0, d]
    #         # Compute gradient of that scalar w.r.t. all θ
    #         grads = torch.autograd.grad(
    #             outputs=scalar,
    #             inputs=list(transition_model.parameters()),
    #             retain_graph=True,     # need to keep the graph for the next d
    #             create_graph=False,
    #             allow_unused=False
    #         )
    #         # Sum up squared entries of those grads:
    #         for g in grads:
    #             sqnorm = sqnorm + (g.detach() ** 2).sum()

    #     # Divide by (2 * sigma_prior^2) to get nats.  e.g. sigma_prior_sq = 1.0
    #     sigma_prior_sq = 0.0005
    #     ig_theta[i] = 0.5 * (sqnorm / sigma_prior_sq)

    # 6) Information‐gain term: KL[q(z'|o') ∥ p(z'|z,a)]
    ig_cv = torch.sum(
        0.5
        * (
                fixed_prior_logvar_cv
                - logvar_post_cv.detach()
                + (logvar_post_cv.detach().exp() + (mu_post_cv.detach() - mu_prior_cv.detach()).pow(2))
                / fixed_prior_logvar_cv.exp()
                - 1
        ),
        dim=1,
    )
    ig_qr = torch.sum(
        0.5
        * (
                fixed_prior_logvar_qr
                - logvar_post_qr.detach()
                + (logvar_post_qr.detach().exp() + (mu_post_qr.detach() - mu_prior_qr.detach()).pow(2))
                / fixed_prior_logvar_qr.exp()
                - 1
        ),
        dim=1,
    )

    # entropy is omitted compared to mcdaci bc logvars are not used
    # entropy_decoded_mean = entropy_normal_from_logvar(fixed_prior_logvar).sum(
    #     dim=1
    # )
    # entropy_decoded_sampled = entropy_normal_from_logvar(logvar_post).sum(dim=1)
    # entr = entropy_decoded_sampled - entropy_decoded_mean

    efe_cv = -ig_cv - pragmatic_value_cv  # - ig_theta
    efe_qr = -ig_qr - pragmatic_value_qr  # - ig_theta
    return efe_cv, efe_qr, ig_cv, ig_qr, pragmatic_value_cv, pragmatic_value_qr


def calculate_expected_free_energy_eh(
        joint_recon_norm_obs,
        preferences_cv,
        preferences_qr,
        joint_mu_prior,
        joint_mu_post,
        joint_logvar_post,
        transition_model_cv,
        transition_model_qr,
):
    """
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
    # todo: extract throughput and accuracy stuff from recon obs (split
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)
    pragmatic_value_cv = convert_rescaled_state_cv_to_slof(recon_norm_obs_cv)
    pragmatic_value_cv = torch.nan_to_num(pragmatic_value_cv, nan=1e-7)
    pragmatic_value_qr = convert_rescaled_state_qr_to_slof(recon_norm_obs_qr)
    pragmatic_value_qr = torch.nan_to_num(pragmatic_value_qr, nan=1e-7)

    # salient_feat_cv = torch.empty(recon_norm_obs_cv.shape[0], 2, device=joint_recon_norm_obs.device)
    # salient_feat_qr = torch.empty(recon_norm_obs_qr.shape[0], 2, device=joint_recon_norm_obs.device)
    # # interpolate solution quality, extract throughput
    # salient_feat_cv[:, 0] = (
    #         recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    # )
    # salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 3]
    # # extract data quality (<=> solution quality), throughpuit
    # salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]
    # salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 3]

    # variance per target (we have two)

    # @Boris: You can bias towards an objective by changing the variance (the higher the variance, the
    #  more relaxed the objective is for an SLO).
    #  However, I'm already biasing the reward towards sol quality
    # var_qr, var_cv = torch.tensor([0.1, 0.1], device=joint_recon_norm_obs.device), torch.tensor([0.1, 0.1],
    #                                                                                             device=joint_recon_norm_obs.device)
    # pragmatic_value_cv = -0.5 * (
    #         (salient_feat_cv.detach() - preferences_cv) ** 2 / var_cv + torch.log(var_cv)
    # ).sum(dim=1)
    # pragmatic_value_qr = -0.5 * (
    #         (salient_feat_qr.detach() - preferences_qr) ** 2 / var_qr + torch.log(var_qr)
    # ).sum(dim=1)

    # a fixed prior bc transition network does not output logvars, 0.1 is mid, not too strict, not too free
    joint_fixed_prior_logvar = torch.ones_like(joint_mu_prior) * np.log(0.1)
    mu_prior_cv, mu_prior_qr = torch.chunk(joint_mu_prior, chunks=2, dim=1)

    fixed_prior_logvar_cv, fixed_prior_logvar_qr = torch.chunk(joint_fixed_prior_logvar, chunks=2, dim=1)
    mu_post_cv, mu_post_qr = torch.chunk(joint_mu_post, chunks=2, dim=1)
    logvar_post_cv, logvar_post_qr = torch.chunk(joint_logvar_post, chunks=2, dim=1)

    # WIP: basically I was trying to come up with the curiosity term from mcdaci
    # bc IG is not IGaining rn

    # N = len(recon_norm_obs)
    # ig_theta = torch.zeros(N, device=mu_prior.device)
    # D_latent = mu_prior.shape[1]

    # for i in range(N):
    #     mu_prior_i = mu_prior[i : i+1]
    #     sqnorm = torch.tensor(0.0, device=mu_prior.device)

    #     for d in range(D_latent):
    #         # Take the single scalar mu_prior_i[0,d]:
    #         scalar = mu_prior_i[0, d]
    #         # Compute gradient of that scalar w.r.t. all θ
    #         grads = torch.autograd.grad(
    #             outputs=scalar,
    #             inputs=list(transition_model.parameters()),
    #             retain_graph=True,     # need to keep the graph for the next d
    #             create_graph=False,
    #             allow_unused=False
    #         )
    #         # Sum up squared entries of those grads:
    #         for g in grads:
    #             sqnorm = sqnorm + (g.detach() ** 2).sum()

    #     # Divide by (2 * sigma_prior^2) to get nats.  e.g. sigma_prior_sq = 1.0
    #     sigma_prior_sq = 0.0005
    #     ig_theta[i] = 0.5 * (sqnorm / sigma_prior_sq)

    # 6) Information‐gain term: KL[q(z'|o') ∥ p(z'|z,a)]
    ig_cv = torch.sum(
        0.5
        * (
                fixed_prior_logvar_cv
                - logvar_post_cv.detach()
                + (logvar_post_cv.detach().exp() + (mu_post_cv.detach() - mu_prior_cv.detach()).pow(2))
                / fixed_prior_logvar_cv.exp()
                - 1
        ),
        dim=1,
    )
    ig_qr = torch.sum(
        0.5
        * (
                fixed_prior_logvar_qr
                - logvar_post_qr.detach()
                + (logvar_post_qr.detach().exp() + (mu_post_qr.detach() - mu_prior_qr.detach()).pow(2))
                / fixed_prior_logvar_qr.exp()
                - 1
        ),
        dim=1,
    )

    # entropy is omitted compared to mcdaci bc logvars are not used
    # entropy_decoded_mean = entropy_normal_from_logvar(fixed_prior_logvar).sum(
    #     dim=1
    # )
    # entropy_decoded_sampled = entropy_normal_from_logvar(logvar_post).sum(dim=1)
    # entr = entropy_decoded_sampled - entropy_decoded_mean

    efe_cv = -ig_cv - pragmatic_value_cv  # - ig_theta
    efe_qr = -ig_qr - pragmatic_value_qr  # - ig_theta
    return efe_cv, efe_qr, ig_cv, ig_qr, pragmatic_value_cv, pragmatic_value_qr
