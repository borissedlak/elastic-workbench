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
