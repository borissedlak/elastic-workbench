import torch
import numpy as np
from agent.daci.aif_utils import (
    denormalize_obs,
    normalize_obs,
    calculate_expected_free_energy,
)
import copy


class MCTS:
    def __init__(self, actions_dim, agent) -> None:
        self.depth = 5  # 15
        self.iterations = 500
        self.c = 0.5
        self.action_space = actions_dim
        self.agent = agent
        self.root_node = None
        self.logs = []
        self.mu_target = torch.tensor([0.55, 0.07]).unsqueeze(0)
        self.gaus_nll = torch.nn.GaussianNLLLoss()

    def init_root_node(self, start_obs):
        self.root_node = Node(self.action_space, self.c, start_obs)

    def apply_action(self, obs, action):
        with torch.no_grad():
            norm_obs = normalize_obs(obs)
            mu, logvar = self.agent.encoder(norm_obs)
            action_oh = self.agent.transform_action(action)
            predicted_next_mu_delta = self.agent.transition_model_qr(
                mu, action_oh.unsqueeze(0)
            )
            unnorm_predicted_next_mu_delta = self.agent.denormalize_deltas(
                predicted_next_mu_delta
            )
            mu_d, logvar_d = self.agent.decoder(mu + unnorm_predicted_next_mu_delta)
            next_obs = denormalize_obs(mu_d)
        return next_obs

    def expand_node(self, node):
        if len(node.available_actions) > 0:
            action = node.available_actions.pop()
            new_obs = self.apply_action(node.obs, action)
            child_node = Node(
                self.action_space,
                self.c,
                new_obs,
                action,
                parent=node,
            )
            node.children.update({action: child_node})
        else:
            child_node = node.return_child()
        return child_node

    def rollout(self, obs):
        total_efe = 0
        for i in range(self.depth):
            action = np.random.choice(
                self.action_space, size=1
            )  # you policy network could be here
            with torch.no_grad():
                # reminder (Alireza): vq-VAE

                # 1) Encode current obs → posterior q(z|o)
                norm_obs = normalize_obs(obs)
                mu, logvar = self.agent.encoder(norm_obs)

                # 2) Predict prior over next latent: p(z'|z,a)
                action_oh = self.agent.transform_action(action)
                (predicted_next_delta) = self.agent.transition_model_qr(
                    mu, action_oh.unsqueeze(0)
                )
                unnorm_predicted_next_mu_delta = self.agent.denormalize_deltas(
                    predicted_next_delta
                )
                mu_prior = unnorm_predicted_next_mu_delta + mu

                # 3) Decode prior latent → predicted observation distribution p(o'|z')
                recon_state, recon_logvar = self.agent.decoder(mu_prior)
                recon_obs = denormalize_obs(recon_state)
                recon_norm_obs = normalize_obs(recon_obs)

                # 4) Re‐encode predicted obs → posterior q(z'|o')
                mu_post, logvar_post = self.agent.encoder(recon_norm_obs)

            efe, ig, pv = calculate_expected_free_energy(
                recon_norm_obs, self.mu_target, mu_prior, mu_post, logvar_post
            )
            self.logs.append((obs, action, recon_obs, pv, ig, efe))
            total_efe += efe.item()
            obs = recon_obs.detach()
        return total_efe

    def select_best_action(self, node):
        best_value = np.inf
        chosen_action = -1
        for action, child in node.children.items():
            if child.visit_count != 0:
                val = child.accumulated_value / child.visit_count
            else:
                val = 1e6
            if val < best_value:
                best_value = val
                chosen_action = action
        return chosen_action

    def generate_best_trajectory(self, root_node, max_depth=30):
        trajectory = []
        current_node = root_node
        depth = 0

        while current_node.children and depth < max_depth:
            best_value = np.inf
            chosen_action = None
            next_node = None

            for action, child in current_node.children.items():
                if child.visit_count > 0:
                    val = child.accumulated_value / child.visit_count
                else:
                    val = 1e6

                if val < best_value:
                    best_value = val
                    chosen_action = action
                    next_node = child

            if chosen_action is None or next_node is None:
                break  # No valid children to follow

            trajectory.append(chosen_action)
            current_node = next_node
            depth += 1

        return trajectory

    def transform_state(self, start_state):
        obs = torch.tensor(start_state).unsqueeze(0)
        return obs

    def tree_policy(self, node):
        while not node.has_available_actions() and node.children:
            node = node.return_child()
        return node

    def find_closest_node(self, node, target_obs, threshold=0.01):
        stack = [node]
        best_node = None
        min_dist = float("inf")

        while stack:
            current = stack.pop()
            dist = torch.norm(current.obs[0] - target_obs[0])
            if dist < min_dist:
                min_dist = dist
                best_node = current

            for child in current.children.values():
                stack.append(child)

        if min_dist < threshold:
            return best_node
        else:
            return None

    def advance_root_with_env_obs(self, new_obs):

        best_node = self.find_closest_node(self.root_node, new_obs)

        if best_node is not None:
            self.root_node = best_node
            self.root_node.parent = None
        else:
            self.root_node = Node(self.action_space, self.c, new_obs)

    def run_mcts(self, start_obs):
        efe_values = []
        if self.root_node:
            start_obs = self.transform_state(start_obs)
            if not torch.equal(self.root_node.obs, start_obs):
                self.advance_root_with_env_obs(start_obs)
        else:
            start_obs = self.transform_state(start_obs)
            self.init_root_node(start_obs)

        for i in range(self.iterations):
            # select
            node_to_expand = self.tree_policy(self.root_node)

            # expand
            node = self.expand_node(node_to_expand)

            # simulate
            reward = self.rollout(node.obs)
            efe_values.append(reward)

            # backprop
            while node is not None:
                node.visit_count += 1
                node.accumulated_value += reward
                node = node.parent

        if efe_values:
            print(f"\n[MCTS Summary]")
            print(f"- Min EFE: {np.min(efe_values):.4f}")
            print(f"- Max EFE: {np.max(efe_values):.4f}")
            print(f"- Mean EFE: {np.mean(efe_values):.4f}")
            print(f"- Std EFE: {np.std(efe_values):.4f}")
        trajectory = self.generate_best_trajectory(self.root_node)  # [best_action]
        print(f"- Best action: {trajectory[0]} (Trajectory length: {len(trajectory)})")
        stats = {
            "min_efe": float(np.min(efe_values)),
            "max_efe": float(np.max(efe_values)),
            "mean_efe": float(np.mean(efe_values)),
            "std_efe": float(np.std(efe_values)),
            "trajectory_length": len(trajectory),
            "best_action": trajectory[0] if trajectory else -1,
        }

        return trajectory, stats, self.root_node


#


class Node:
    def __init__(
        self, action_space, exploration_constant, obs, action=None, parent=None
    ):
        self.visit_count = 0
        self.accumulated_value = 0
        self.parent = parent
        self.available_actions = set(range(action_space))
        self.action_space = action_space
        self.children = {}
        self.action = action
        self.obs = obs
        self.exploration_constant = exploration_constant

    def has_available_actions(self):
        return len(self.available_actions) != 0

    def return_child(self):
        return max(
            self.children.values(),
            key=lambda child: child.ucb(self.exploration_constant, self.visit_count),
        )

    def ucb(self, exploration_constant, visit_count):
        if self.visit_count > 0:
            return (
                self.accumulated_value / self.visit_count
                + exploration_constant * np.sqrt(np.log(visit_count) / self.visit_count)
            )
        else:
            return np.inf
