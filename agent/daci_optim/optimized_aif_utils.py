import torch
import numpy as np


def reparameterize(mu, logvar):
    """GPU-optimized reparameterization trick"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def entropy_normal_from_logvar(logvar):
    """GPU-optimized entropy calculation"""
    constant = torch.tensor(
        np.log(2 * np.pi * np.e), dtype=logvar.dtype, device=logvar.device
    )
    return 0.5 * (constant + logvar)


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
    GPU-optimized Expected Free Energy calculation

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
    device = joint_recon_norm_obs.device
    batch_size = joint_recon_norm_obs.shape[0]

    # Split observations for CV and QR services
    recon_norm_obs_cv, recon_norm_obs_qr = torch.chunk(joint_recon_norm_obs, chunks=2, dim=1)

    # Pre-allocate tensors for salient features
    salient_feat_cv = torch.empty(batch_size, 2, device=device, dtype=torch.float32)
    salient_feat_qr = torch.empty(batch_size, 2, device=device, dtype=torch.float32)

    # Vectorized feature extraction for CV service
    # Interpolate solution quality: 0.25 * data_quality + 0.75 * model_size
    salient_feat_cv[:, 0] = recon_norm_obs_cv[:, 0] * 0.25 + recon_norm_obs_cv[:, 4] * 0.75
    salient_feat_cv[:, 1] = recon_norm_obs_cv[:, 2]  # throughput

    # Extract features for QR service
    salient_feat_qr[:, 0] = recon_norm_obs_qr[:, 0]  # data quality (solution quality)
    salient_feat_qr[:, 1] = recon_norm_obs_qr[:, 2]  # throughput

    # Variance parameters (configurable for SLO bias)
    # Higher variance = more relaxed objective
    var_qr = torch.tensor([0.1, 0.1], device=device, dtype=torch.float32)
    var_cv = torch.tensor([0.1, 0.1], device=device, dtype=torch.float32)

    # Ensure preferences are on the correct device and have the right shape
    if preferences_cv.device != device:
        preferences_cv = preferences_cv.to(device)
    if preferences_qr.device != device:
        preferences_qr = preferences_qr.to(device)

    # Expand preferences to match batch size if needed
    if preferences_cv.shape[0] == 1 and batch_size > 1:
        preferences_cv = preferences_cv.expand(batch_size, -1)
    if preferences_qr.shape[0] == 1 and batch_size > 1:
        preferences_qr = preferences_qr.expand(batch_size, -1)

    # Calculate pragmatic values (vectorized)
    pragmatic_value_cv = -0.5 * (
            (salient_feat_cv.detach() - preferences_cv) ** 2 / var_cv + torch.log(var_cv)
    ).sum(dim=1)

    pragmatic_value_qr = -0.5 * (
            (salient_feat_qr.detach() - preferences_qr) ** 2 / var_qr + torch.log(var_qr)
    ).sum(dim=1)

    # Fixed prior variance for transition network
    # 0.1 is balanced - not too strict, not too free
    joint_fixed_prior_logvar = torch.full_like(joint_mu_prior, np.log(0.1))

    # Split priors and posteriors
    mu_prior_cv, mu_prior_qr = torch.chunk(joint_mu_prior, chunks=2, dim=1)
    fixed_prior_logvar_cv, fixed_prior_logvar_qr = torch.chunk(joint_fixed_prior_logvar, chunks=2, dim=1)
    mu_post_cv, mu_post_qr = torch.chunk(joint_mu_post, chunks=2, dim=1)
    logvar_post_cv, logvar_post_qr = torch.chunk(joint_logvar_post, chunks=2, dim=1)

    # Information gain: KL[q(z'|o') || p(z'|z,a)] - vectorized computation
    # KL divergence between posterior and prior
    def compute_kl_divergence(mu_post, logvar_post, mu_prior, fixed_prior_logvar):
        """Vectorized KL divergence computation"""
        return torch.sum(
            0.5 * (
                    fixed_prior_logvar
                    - logvar_post.detach()
                    + (logvar_post.detach().exp() + (mu_post.detach() - mu_prior.detach()).pow(2))
                    / fixed_prior_logvar.exp()
                    - 1
            ),
            dim=1,
        )

    ig_cv = compute_kl_divergence(mu_post_cv, logvar_post_cv, mu_prior_cv, fixed_prior_logvar_cv)
    ig_qr = compute_kl_divergence(mu_post_qr, logvar_post_qr, mu_prior_qr, fixed_prior_logvar_qr)

    # Expected Free Energy calculation
    # EFE = -Information_Gain - Pragmatic_Value
    # Lower EFE is better (more preferred)
    efe_cv = -ig_cv - pragmatic_value_cv
    efe_qr = -ig_qr - pragmatic_value_qr

    return efe_cv, efe_qr, ig_cv, ig_qr, pragmatic_value_cv, pragmatic_value_qr


def batch_calculate_expected_free_energy(
        joint_recon_norm_obs_batch,
        preferences_cv,
        preferences_qr,
        joint_mu_prior_batch,
        joint_mu_post_batch,
        joint_logvar_post_batch,
        transition_model_cv,
        transition_model_qr,
):
    """
    Batch-optimized version of EFE calculation for multiple policy rollouts

    Args:
        joint_recon_norm_obs_batch: [batch_size, num_policies, obs_dim]
        preferences_cv/qr: [batch_size, 2] or [1, 2]
        joint_mu_prior_batch: [batch_size, num_policies, latent_dim]
        joint_mu_post_batch: [batch_size, num_policies, latent_dim]
        joint_logvar_post_batch: [batch_size, num_policies, latent_dim]

    Returns:
        efe_cv, efe_qr: [batch_size, num_policies]
    """
    batch_size, num_policies, obs_dim = joint_recon_norm_obs_batch.shape
    device = joint_recon_norm_obs_batch.device

    # Reshape for batch processing
    flat_obs = joint_recon_norm_obs_batch.view(-1, obs_dim)
    flat_mu_prior = joint_mu_prior_batch.view(-1, joint_mu_prior_batch.shape[-1])
    flat_mu_post = joint_mu_post_batch.view(-1, joint_mu_post_batch.shape[-1])
    flat_logvar_post = joint_logvar_post_batch.view(-1, joint_logvar_post_batch.shape[-1])

    # Expand preferences for the flattened batch
    if preferences_cv.shape[0] == 1:
        preferences_cv_expanded = preferences_cv.expand(batch_size * num_policies, -1)
    else:
        preferences_cv_expanded = preferences_cv.unsqueeze(1).expand(-1, num_policies, -1).reshape(-1, 2)

    if preferences_qr.shape[0] == 1:
        preferences_qr_expanded = preferences_qr.expand(batch_size * num_policies, -1)
    else:
        preferences_qr_expanded = preferences_qr.unsqueeze(1).expand(-1, num_policies, -1).reshape(-1, 2)

    # Calculate EFE for flattened batch
    efe_cv_flat, efe_qr_flat, ig_cv_flat, ig_qr_flat, pv_cv_flat, pv_qr_flat = calculate_expected_free_energy(
        flat_obs,
        preferences_cv_expanded,
        preferences_qr_expanded,
        flat_mu_prior,
        flat_mu_post,
        flat_logvar_post,
        transition_model_cv,
        transition_model_qr,
    )

    # Reshape back to batch format
    efe_cv = efe_cv_flat.view(batch_size, num_policies)
    efe_qr = efe_qr_flat.view(batch_size, num_policies)
    ig_cv = ig_cv_flat.view(batch_size, num_policies)
    ig_qr = ig_qr_flat.view(batch_size, num_policies)
    pv_cv = pv_cv_flat.view(batch_size, num_policies)
    pv_qr = pv_qr_flat.view(batch_size, num_policies)

    return efe_cv, efe_qr, ig_cv, ig_qr, pv_cv, pv_qr


def compute_action_probabilities(efe_values, temperature=1.0, method='softmax'):
    """
    Compute action probabilities from EFE values

    Args:
        efe_values: [batch_size, num_actions] - Expected Free Energy values
        temperature: float - Temperature parameter for softmax
        method: str - 'softmax' or 'power_normalize'

    Returns:
        probabilities: [batch_size, num_actions] - Action probabilities
    """
    if method == 'softmax':
        # Lower EFE is better, so we use negative EFE for softmax
        return torch.softmax(-efe_values / temperature, dim=-1)
    elif method == 'power_normalize':
        return power_normalize(efe_values)
    else:
        raise ValueError(f"Unknown method: {method}")


def power_normalize(x: torch.Tensor, alpha: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """
    Power normalization for converting EFE to probabilities

    Args:
        x: [batch_size, num_actions] - EFE values
        alpha: float - Power parameter
        eps: float - Small constant for numerical stability

    Returns:
        probabilities: [batch_size, num_actions] - Normalized probabilities
    """
    # Shift to make all values positive (lower EFE should have higher probability)
    x_shifted = x - torch.min(x, dim=-1, keepdim=True)[0]
    x_shifted = x_shifted + eps

    # Apply power transformation
    x_pow = x_shifted.pow(alpha)

    # Normalize to get probabilities
    return x_pow / x_pow.sum(dim=-1, keepdim=True)


def sample_actions_from_efe(efe_cv, efe_qr, temperature=1.0, method='softmax'):
    """
    Sample actions based on combined EFE values

    Args:
        efe_cv: [batch_size, num_actions] - CV service EFE
        efe_qr: [batch_size, num_actions] - QR service EFE  
        temperature: float - Temperature for action selection
        method: str - Method for computing probabilities

    Returns:
        actions: [batch_size] - Sampled action indices
        probabilities: [batch_size, num_actions] - Action probabilities
    """
    # Combine EFE values (you might want to weight them differently)
    combined_efe = efe_cv + efe_qr

    # Compute probabilities
    probabilities = compute_action_probabilities(combined_efe, temperature, method)

    # Sample actions
    actions = torch.multinomial(probabilities, num_samples=1).squeeze(-1)

    return actions, probabilities