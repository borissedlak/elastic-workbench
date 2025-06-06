import itertools
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import utils
from agent.daci.aif_utils import calculate_expected_free_energy, calculate_expected_free_energy_eh
from agent.daci_optim.hybrid_daci_agent import HybridMCDACIAgent


def _one_hot(idx: int, dim: int, device: torch.device) -> torch.Tensor:
    """Utility for fast one‑hot creation (B=1)."""
    return F.one_hot(
        torch.tensor([idx], device=device, dtype=torch.long), num_classes=dim
    ).float()


class _Node:
    """A single node for the joint‑action MCTS (CV × QR).

    * ``action`` holds the **index** inside ``MCTS.action_pairs``; use
      ``mcts.action_pairs[action]`` to recover the actual (cv, qr) tuple.
    """

    def __init__(
            self,
            action_space_size: int,
            c: float,
            obs: torch.Tensor,  # [1, obs_dim]
            action: int | None = None,
            parent: "_Node | None" = None,
    ) -> None:
        self.c = c
        self.obs = obs  # already on correct device, *normalised*
        self.parent = parent
        self.action = action

        self.visit_count: int = 0
        self.accum_value: float = 0.0  # lower = better (EFE)

        self.children: dict[int, _Node] = {}
        self.available_actions = list(
            range(action_space_size)
        )  # indices that are not expanded yet

    # ――― UCT helpers ――― ---------------------------------------------------------------------
    def _ucb(self, child: "_Node") -> float:
        if child.visit_count == 0:
            return float("inf")
        exploit = -child.accum_value / child.visit_count  # we *minimise* EFE
        explore = self.c * np.sqrt(np.log(self.visit_count + 1) / child.visit_count)
        return exploit + explore

    def select_child(self) -> "_Node":
        """Select the child with maximal UCB."""
        best_score = -float("inf")
        best_child: _Node | None = None
        for child in self.children.values():
            score = self._ucb(child)
            if score > best_score:
                best_score = score
                best_child = child
        # The tree policy only calls this when *some* child exists.
        assert best_child is not None
        return best_child

    def update(self, value: float):
        self.visit_count += 1
        self.accum_value += value

    def is_fully_expanded(self) -> bool:
        return len(self.available_actions) == 0


class MCTS:
    """Monte‑Carlo Tree Search for the *joint* CV/QR agent.

    The class assumes **two** independent transition models and therefore keeps them separate
    throughout planning, while still expanding over the joint action space (Cartesian product).
    """

    def __init__(
            self,
            action_dim_cv: int,
            action_dim_qr: int,
            agent: HybridMCDACIAgent,
            *,
            depth: int = 5,
            max_len: int = 10,
            iterations: int = 500,
            c: float = 0.5,
            use_eh: bool = False
    ):
        self.action_dim_cv = action_dim_cv
        self.action_dim_qr = action_dim_qr
        self.action_pairs = list(
            itertools.product(range(action_dim_cv), range(action_dim_qr))
        )

        self.depth = depth
        self.iterations = iterations
        self.c = c
        self.agent = agent
        self.device = getattr(agent, "device", torch.device("cpu"))
        self.max_len = max_len
        self.root: _Node | None = None
        self.use_eh = use_eh

    @utils.print_execution_time
    def run_mcts(self, start_state: np.ndarray | torch.Tensor):
        """Main entry – builds a search tree and returns the best trajectory of (cv, qr) actions."""
        start_obs = self._to_tensor(start_state)
        self.root = _Node(len(self.action_pairs), self.c, start_obs)

        for _ in range(self.iterations):
            node = self._tree_policy(self.root)
            value = self._rollout(node.obs.clone())
            self._backprop(node, value)

        best_traj = self._extract_best_trajectory(self.root)
        return best_traj, None, self.root  # placeholder for stats (None for now)

    def _tree_policy(self, node: _Node) -> _Node:
        """Select a node for rollout using the usual *selection → expansion* strategy."""
        while True:
            if not node.is_fully_expanded():
                return self._expand(node)
            # fully expanded – descend
            node = node.select_child()

    def _expand(self, node: _Node) -> _Node:
        action_idx = node.available_actions.pop(
            np.random.randint(len(node.available_actions))
        )
        next_obs = self._apply_action(node.obs, action_idx)
        child = _Node(len(self.action_pairs), self.c, next_obs, action_idx, parent=node)
        node.children[action_idx] = child
        return child

    def _rollout(self, obs: torch.Tensor) -> float:
        total_efe = 0.0
        for _ in range(self.depth):
            action_idx = np.random.randint(len(self.action_pairs))
            obs, efe_val = self._step_dynamics(obs, action_idx)
            total_efe += efe_val
        return total_efe

    def _step_dynamics(self, norm_obs: torch.Tensor, action_idx: int):
        """One predictive step *without* modifying any tree structures."""
        action_cv, action_qr = self.action_pairs[action_idx]

        # 1) encode current observation
        mu_joint, logvar_joint = self.agent.world_model.encode(norm_obs, sample=False)[
            "s_dist_params"
        ]
        mu_cv, mu_qr = torch.chunk(mu_joint, 2, dim=1)

        # 2) delta z predictions from the separate transition nets
        a_cv = _one_hot(action_cv, self.action_dim_cv, self.device)
        a_qr = _one_hot(action_qr, self.action_dim_qr, self.device)

        delta_cv = self.agent.transition_model_cv(mu_cv, a_cv)["delta"]
        delta_qr = self.agent.transition_model_qr(mu_qr, a_qr)["delta"]
        joint_delta = torch.cat([delta_cv, delta_qr], dim=1)
        joint_delta = joint_delta
        mu_prior = mu_joint + joint_delta

        # 3) project back to observation space
        recon_mu, _ = self.agent.world_model.decode(mu_prior, sample=False)[
            "o_dist_params"
        ]
        recon_norm_obs = recon_mu  # already sigmoid‑ed → within [0,1]

        # 4) posterior over z′
        mu_post, logvar_post = self.agent.world_model.encode(
            recon_norm_obs, sample=False
        )["s_dist_params"]

        # 5) EFE for both services
        if self.use_eh:
            efe_cv, efe_qr, *_ = calculate_expected_free_energy_eh(
                self.agent.vec_env.min_max_rescale(recon_norm_obs),
                self.agent.preferences_cv,
                self.agent.preferences_qr,
                mu_prior,
                mu_post,
                logvar_post,
                self.agent.transition_model_cv,
                self.agent.transition_model_qr,
            )
        else:
            efe_cv, efe_qr, *_ = calculate_expected_free_energy(
                recon_norm_obs,
                self.agent.preferences_cv,
                self.agent.preferences_qr,
                mu_prior,
                mu_post,
                logvar_post,
                self.agent.transition_model_cv,
                self.agent.transition_model_qr,
            )
        efe_val = (efe_cv + efe_qr).item()
        return recon_norm_obs.detach(), efe_val

    def _apply_action(self, norm_obs: torch.Tensor, action_idx: int) -> torch.Tensor:
        """Wrapper used during tree expansion – identical to the rollout step but returns only obs."""
        next_obs, _ = self._step_dynamics(norm_obs, action_idx)
        return next_obs

    def _backprop(self, node: _Node, value: float):
        while node is not None:
            node.update(value)
            node = node.parent

    def _to_tensor(self, arr: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(arr, torch.Tensor):
            return (
                arr.to(self.device).unsqueeze(0)
                if arr.dim() == 1
                else arr.to(self.device)
            )
        return torch.tensor(arr, device=self.device, dtype=torch.float32).unsqueeze(0)

    def _extract_best_trajectory(self, node: _Node):
        traj: list[tuple[int, int]] = []
        depth = 0
        while node.children and depth < self.max_len:
            # pick the *most‑visited* child
            best_child = max(node.children.values(), key=lambda n: n.visit_count)
            traj.append(self.action_pairs[best_child.action])
            node = best_child
            depth += 1
        return traj
