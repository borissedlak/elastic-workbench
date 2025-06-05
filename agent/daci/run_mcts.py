import argparse
import os.path
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from agent.daci_optim.hybrid_daci_agent import HybridMCDACIAgent

from mcts_utils import MCTS


def scale_joint(raw: torch.Tensor, vec_env) -> torch.Tensor:
    """Scale an 8‑D raw joint observation to 0‑1 using the env’s helper."""
    return vec_env.min_max_scale(raw)


def rescale_joint(scaled: torch.Tensor, vec_env) -> torch.Tensor:
    """Inverse of :pyfunc:`scale_joint`. Returns raw units."""
    return vec_env.min_max_rescale(scaled)


# -----------------------------------------------------------------------------
# Tiny debug visualiser (optional) ------------------------------------------------


def visualise_tree(root, max_depth: int = 2):
    """Draw a *tiny* slice of the tree so you can confirm expansion works."""

    G = nx.DiGraph()
    labels: dict[int, str] = {}

    def add_edges(node, depth: int):
        if depth > max_depth:
            return
        for action, child in node.children.items():
            G.add_edge(id(node), id(child))
            labels[id(child)] = f"a={action}\nV={child.accum_value:.2f}"
            add_edges(child, depth + 1)

    labels[id(root)] = "ROOT"
    add_edges(root, 0)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:  # pragma: no cover
        pos = nx.spring_layout(G, seed=42)
        print("Graphviz not found – falling back to spring layout.")

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=False, node_size=300, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("MCTS (depth ≤ %d)" % max_depth)
    plt.tight_layout()
    plt.show()


def build_default_boundaries() -> dict:
    """Returns a *tiny* set of reasonable boundaries for a quick demo."""

    return {
        "data_quality": {"min": 0, "max": 100},
        "model_size": {"min": 0, "max": 150},
        "cores": {"min": 0, "max": 64},
    }


def build_midpoint_state(boundaries: dict) -> torch.Tensor:
    """Produces an 8‑D tensor located roughly in the middle of each range."""

    def mid(key):
        return 0.5 * (boundaries[key]["min"] + boundaries[key]["max"])

    # [dq_cv, dq_qr, unused, unused, ms_cv, ms_qr, cores_cv, cores_qr]
    state = torch.tensor(
        [
            mid("data_quality"),
            mid("data_quality"),
            0.0,
            0.0,
            mid("model_size"),
            mid("model_size"),
            mid("cores"),
            mid("cores"),
        ],
        dtype=torch.float32,
    ).unsqueeze(
        0
    )  # (1, 8)
    return state


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick MCTS smoke‑test")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0 | cuda:1 …")
    p.add_argument("--iters", type=int, default=100, help="MCTS iterations")
    p.add_argument("--depth", type=int, default=5, help="Search depth")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--viz", action="store_true", help="Visualise a slice of the tree")
    return p.parse_args()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    device = "cpu"
    viz = True
    depth: int = 2
    max_length_trajectory = 5
    iterations: int = 300
    c: float = 0.5
    agent_file = "hybrid_agent_checkpoint__hybrid_adaptive.pth"
    boundaries = {
        "model_size": {"min": 1, "max": 5},
        "data_quality": {"min": 100, "max": 1000},
        "cores": {"min": 1, "max": 8},
        "throughput": {"min": 0, "max": 100},
    }
    cv_slo_targets = (
        {
            "data_quality": 288,
            "model_size": 4,
            "throughput": 5,
        },
    )
    qr_slo_targets = (
        {
            "data_quality": 900,
            "throughput": 75,
        },
    )
    test_iters = 10

    if os.path.isfile(agent_file):
        agent = torch.load(agent_file, weights_only=False, map_location=device)["agent"]
        agent.device = device
    else:
        agent = HybridMCDACIAgent(
            boundaries=boundaries,
            cv_slo_targets=cv_slo_targets,
            qr_slo_targets=qr_slo_targets,
            device=device,
        )

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
    # Initialize start state
    start_state_raw_cv = torch.tensor([500, 300, 2, 5, 1, 3, 2,  4])
    start_state_raw_qr = torch.tensor([500, 800, 2, 75, 1, 1, 2,  4])
    joint_state = torch.cat([start_state_raw_cv, start_state_raw_qr], dim=0).unsqueeze(0).to(dtype=torch.float32, device=device)
    joint_state = scale_joint(joint_state, agent.vec_env)
    print("Start state (raw):", joint_state.squeeze().tolist())
    # 1) Initialize mcts
    mcts = MCTS(
        action_dim_qr=agent.action_dim_qr,
        action_dim_cv=agent.action_dim_cv,
        agent=agent,
        depth=depth,
        iterations=iterations,
        max_len=max_length_trajectory,
        c=c,
    )

    # THIS WILL BE IN YOUR GAME LOOP
    #  Apply as follows:
    #   1) pass current (joint state, just concatenate on top of each other the 2x(1, 8) observations
    #     to split you can: obs_cv, obs_qr = torch.chunk(joint_state, chunks=2, dim=1)
    #     ATTENTION: State mut be min max scaled between 0 and 1 see below for example
    #   2) Extract your next action(s) from trajectory. Trajectory is a list of tuples (action_cv, action_qr)
    #   3) You can decide whether:
    #      a) to apply actions from the list of trajectories (more efficient)
    #      b) apply only trajectory[0]
    #   4) to get the next state call
    for _ in range(test_iters):
        trajectory, stats, root = mcts.run_mcts(joint_state)
        action_cv, action_qr = trajectory[0]
        state_cv, state_qr = torch.chunk(joint_state, chunks=2, dim=1)
        #  note: Tensors are always shape (B, *). Since we are not training, B=1
        next_cv, _ = agent.simple_probe_transition(
            state_cv.squeeze(), None, action=action_cv, service_type="cv"
        )
        next_qr, _ = agent.simple_probe_transition(
            state_qr.squeeze(), None, action=action_qr, service_type="qr"
        )
        joint_state = torch.tensor(
            np.concatenate([next_cv, next_qr]), dtype=torch.float32, device=device
        )[
            None,
        ]
        # Note: if you need the recsaled state (original value range):
        rescaled_joint_state =rescale_joint(scaled=joint_state, vec_env=agent.vec_env)
        print(rescaled_joint_state)

    if viz:
        visualise_tree(root)
