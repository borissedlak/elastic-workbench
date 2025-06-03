import torch
import numpy as np


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def entropy_normal_from_logvar(logvar):
    constant = torch.tensor(
        np.log(2 * np.pi * np.e), dtype=logvar.dtype, device=logvar.device
    )
    return 0.5 * (constant + logvar)


def normalize_obs(obs):
    pos, vel = obs[:, 0], obs[:, 1]
    norm_pos = 2 * (pos - (-1.2)) / (0.6 - (-1.2)) - 1
    norm_vel = 2 * (vel - (-0.07)) / (0.07 - (-0.07)) - 1
    return torch.stack([norm_pos, norm_vel], dim=1)


def denormalize_obs(norm_obs):
    norm_pos, norm_vel = norm_obs[:, 0], norm_obs[:, 1]
    pos = 0.5 * (norm_pos + 1) * (0.6 - (-1.2)) + (-1.2)
    vel = 0.5 * (norm_vel + 1) * (0.07 - (-0.07)) + (-0.07)
    return torch.stack([pos, vel], dim=1)


def calculate_expected_free_energy(
    recon_norm_obs, preferences_cv, preferences_qr, mu_prior, mu_post, logvar_post, transition_model
):
    # todo: extract throughput and accuracy stuff from recon obs (split
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(recon_norm_obs, chunks=2, dim=1)

    # variance per target (we have two)
    var = torch.tensor([0.1, 0.1])  # how close we want to be to the target
    pragmatic_value = -0.5 * (
        (recon_norm_obs.detach() - normalize_obs(mu_target)) ** 2 / var + torch.log(var)
    ).sum(dim=1)

    # a fixed prior bc transition network does not output logvars, 0.1 is mid, not too strict, not too free
    fixed_prior_logvar = torch.ones_like(mu_prior) * np.log(0.1)

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
    ig = torch.sum(
        0.5
        * (
            fixed_prior_logvar
            - logvar_post.detach()
            + (
                logvar_post.detach().exp()
                + (mu_post.detach() - mu_prior.detach()).pow(2)
            )
            / fixed_prior_logvar.exp()
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

    efe = -ig - pragmatic_value  # - ig_theta
    return efe, ig, pragmatic_value
