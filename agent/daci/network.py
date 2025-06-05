import collections
from abc import ABC
from itertools import repeat
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from torch.nn.init import trunc_normal_

from proj_types import (
    HabitualNetworkOutput,
    MCDaciWorldModelDecOutput,
    SimpleDeltaTransitionNetworkOutput,
    SimpleMCDaciWorldModelDecOutput,
    WorldModelEncoding,
    MCDaciWorldOutput,
    TransitionNetworkOutput,
)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


class FullyConnectedBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        non_linearity: Callable[[], nn.Module],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
        )
        if dropout > 0.0:
            self.block.append(nn.Dropout(dropout))
        self.block.append(non_linearity())

    def forward(self, x):
        return self.block(x)


class ParametricTransform(nn.Module, ABC):
    """ """

    def __init__(self):
        super().__init__()

    @torch.jit.ignore
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^conv1|bn1|maxpool",
            blocks=r"^layer(\d+)" if coarse else r"^layer(\d+)\.(\d+)",
        )
        return matcher


class HabitualNetwork(nn.Module):  # ModelTop
    """Q(a|s) ? "Habitual Prior" Q(a) ???

    "MCTS scheme used for planning and acting, using the habitual network to selectively explore new tree branche"

    Note: Not used to calculate EFE.
        My guess is, that this is used to find an initial policy and its corresponding state to use

    Predicts the conditional policy distribution given the latent (hidden state)
    """

    def __init__(
        self,
        latent_dim: int = 10,  # s_dim
        policy_dim: int = 4,  # pi_dim,
        width: int = 32,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.policies_dim = policy_dim
        self.width = width

        self.layers = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=self.width),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.width, out_features=self.width),
            nn.LeakyReLU(),
            nn.Linear(in_features=self.width, out_features=self.policies_dim),
        )

    def forward(
        self, s, logits: bool = True, eps: float = 1e-20
    ) -> HabitualNetworkOutput:  # encode_s
        policy_logits = self.layers(s)
        policy_q_p = F.softmax(policy_logits, dim=-1)
        policy_q_logp = torch.log(policy_q_p + eps)
        if logits:
            return HabitualNetworkOutput(
                policy_logits=policy_logits,
                policy_p=policy_q_p,
                policy_logp=policy_q_logp,
            )
        else:
            return HabitualNetworkOutput(
                policy_logits=None,
                policy_p=policy_q_p,
                policy_logp=policy_q_logp,
            )


class TransitionNetwork(nn.Module):  # Model Mid
    """P(s_t|s_t-1,a_t-1)

    Predict the latent (hidden state) gave the action and previous state
    """

    def __init__(
        self,
        world_latent_dim: int = 4,  # s_dim
        policy_dim: int = 7,  # pi_dim,
        width: int = 32,
        dropout: float = 0.5,
        depth_increase: int = 0,
    ):
        super().__init__()

        self.latent_dim = world_latent_dim
        self.policy_dim = policy_dim

        self.layers = nn.Sequential(
            FullyConnectedBlock(
                in_features=self.latent_dim + self.policy_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=dropout,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
            nn.Linear(
                in_features=width,
                out_features=self.latent_dim * 2,  # gauss params dim, mu logvar
            ),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, pi_t: torch.Tensor, s_t: torch.Tensor) -> TransitionNetworkOutput:
        # transition in_dim => [B, world_latent_dim + action_dim]
        gauss_params = self.layers(torch.cat([pi_t, s_t], dim=1))
        mu, logvar = gauss_params.chunk(chunks=2, dim=1)
        s_t1 = self.reparameterize(mu, logvar)
        return TransitionNetworkOutput(s=s_t1, s_dist_params=(mu, logvar))


class SimpleDeltaTransitionNetwork(nn.Module):  # Model Mid
    """P(s_t|s_t-1,a_t-1)

    Like TransitionNetwork, but for predicting deltas and setting scale = 0.0
    """

    def __init__(
        self,
        latent_dim: int = 4,
        policy_dim: int = 4,
        width: int = 32,
        dropout: float = 0.0,
        depth_increase: int = 0,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(latent_dim + policy_dim)

        self.joint_latent_dim = latent_dim
        self.policy_dim = policy_dim

        self.layers = nn.Sequential(
            FullyConnectedBlock(
                in_features=self.joint_latent_dim + self.policy_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=dropout,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
        )
        self.head = nn.Linear(
            in_features=width,
            out_features=self.joint_latent_dim,
        )

    def forward(
        self, s_t: torch.Tensor, pi_t: torch.Tensor
    ) -> SimpleDeltaTransitionNetworkOutput:
        # transition in_dim => [B, world_latent_dim + action_dim]
        t = self.norm(torch.cat([pi_t, s_t], dim=1))
        mu = self.layers(t)
        delta = self.head(mu)
        delta = torch.tanh(delta)
        #delta = torch.clamp(delta, -2, 2)
        return SimpleDeltaTransitionNetworkOutput(delta=delta)


class WorldModel(ParametricTransform):
    """Variational Autoencoder that learns the model (environment)

    "Habitual Prior": Q(s)
    Encoder: Q(s|o)
    Decoder: P(o|s)

    """

    def __init__(
        self,
        in_dim: int = 9,
        out_dim: Optional[int] = None,
        world_latent_dim: int = 4,  # s_dim
        policy_dim: int = 7,  # pi_dim,
        width: int = 32,
        dropout: float = 0.5,
        depth_increase: int = 0,
    ):
        super().__init__()

        self.out_dim = out_dim or in_dim
        # Sort of only needed in MTCS might as well use agent to query that
        self.policy_dim = policy_dim
        # just hardcode for now
        self.enc_linear = nn.Sequential(
            FullyConnectedBlock(
                in_features=in_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=dropout,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
            nn.Linear(
                in_features=width, out_features=world_latent_dim * 2
            ),  # loc, scale
        )

        self.dec_linear = nn.Sequential(
            FullyConnectedBlock(
                in_features=world_latent_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=0.5,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
            nn.Linear(in_features=width, out_features=out_dim),  # loc, scale
        )

        self.apply(self._init_weights)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, o, sample: bool = False) -> WorldModelEncoding:
        o = self.enc(o)
        o = o.flatten(start_dim=1)
        gauss_params = self.enc_linear(o)
        mu, logvar = gauss_params.chunk(chunks=2, dim=1)
        if sample:
            s = self.reparameterize(mu, logvar)
            return WorldModelEncoding(s=s, s_dist_params=(mu, logvar))
        return WorldModelEncoding(s=None, s_dist_params=(mu, logvar))

    def decode(self, s) -> MCDaciWorldModelDecOutput:
        o_t1 = self.dec_linear(s)
        return MCDaciWorldModelDecOutput(o_pred=o_t1)

    def forward(self, o) -> MCDaciWorldOutput:
        enc_out = self.encode(o, sample=True)
        s = enc_out["s"]
        dec_out = self.decode(s)
        return MCDaciWorldOutput(**dec_out, **enc_out)


class SimpleMCDaciWorldModel(ParametricTransform):
    """Like WorldModel but for simplified environments and objective

    Does not use habitual network
    Encoder: Q(s|o)
    Decoder: P(o|s)

    """

    def __init__(
        self,
        in_dim: int = 2 * 8,  # observe both service states
        out_dim: Optional[int] = None,
        world_latent_dim: int = 2 * 4,  # output joint latent
        width: int = 32,
        dropout: float = 0.5,
        depth_increase: int = 0,
    ):
        super().__init__()

        self.out_dim = out_dim or in_dim
        # Sort of only needed in MTCS might as well use agent to query that

        self.enc_transf = nn.Sequential(
            FullyConnectedBlock(
                in_features=in_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=dropout,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
        )
        self.enc_mu = nn.Linear(width, world_latent_dim)
        self.enc_logvar = nn.Linear(width, world_latent_dim)

        self.dec_transf = nn.Sequential(
            FullyConnectedBlock(
                in_features=world_latent_dim,
                out_features=width,
                non_linearity=nn.LeakyReLU,
                dropout=0.5,
            ),
            *[
                FullyConnectedBlock(
                    in_features=width,
                    out_features=width,
                    non_linearity=nn.LeakyReLU,
                    dropout=dropout,
                )
                for _ in range(depth_increase)
            ],
        )
        self.dec_mu = nn.Linear(width, self.out_dim)
        self.dec_logvar = nn.Linear(width, self.out_dim)

        self.apply(self._init_weights)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, obs: torch.Tensor, sample: bool = False) -> WorldModelEncoding:
        s = self.enc_transf(obs)
        mu, logvar = self.enc_mu(s), self.enc_logvar(s)
        logvar = torch.clamp(
            logvar, min=-10, max=1
        )  # Adjust max as needed, e.g., 2 or 4 might be more stable
        if sample:
            s = self.reparameterize(mu, logvar)
            return WorldModelEncoding(s=s, s_dist_params=(mu, logvar))
        return WorldModelEncoding(s=None, s_dist_params=(mu, logvar))

    def decode(
        self, s: torch.Tensor, sample: bool = False
    ) -> SimpleMCDaciWorldModelDecOutput:
        h = self.dec_transf(s)
        expectation, logvar = self.dec_mu(h), self.dec_logvar(h)

        expectation = torch.sigmoid(expectation)

        logvar = torch.clamp(logvar, min=-10, max=1)
        obs_pred = self.reparameterize(expectation, logvar) if sample else None

        return SimpleMCDaciWorldModelDecOutput(
            o_pred=obs_pred, o_dist_params=(expectation,logvar)
        )

    def forward(self, o) -> MCDaciWorldOutput:
        enc_out = self.encode(o, sample=True)
        s = enc_out["s"]
        dec_out = self.decode(s)
        return MCDaciWorldOutput(**dec_out, **enc_out)


if __name__ == "__main__":
    # todo: Some sanity checks
    pass
