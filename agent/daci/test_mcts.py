import torch
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from agent.daci.mcts_utils import MCTS


def visualize_tree(root, max_depth=3):
    G = nx.DiGraph()
    labels = {}

    def add_edges(node, depth):
        if depth > max_depth:
            return
        for action, child in node.children.items():
            G.add_edge(id(node), id(child))
            labels[id(child)] = (
                f"A:{action}\n"
                f"V:{child.accumulated_value:.2f}/N:{child.visit_count}\n"
                f"OBS:({child.obs[0][0].item():.4f}, {child.obs[0][1].item():.4f})"
            )
            add_edges(child, depth + 1)

    labels[id(root)] = (
        f"ROOT" f"({root.obs[0][0].item():.4f}, {root.obs[0][1].item():.4f})"
    )
    add_edges(root, depth=0)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G)
        print("Graphviz layout unavailable, using spring layout.")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_size=1000, arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title(f"MCTS Search Tree (max_depth={max_depth})")
    plt.tight_layout()
    plt.show()


agent_dict = torch.load("aif_agent_checkpoint__clamp.pth")
agent = agent_dict["agent"]
mcts = MCTS(3, agent)
# where we start
# set this value strategically bc if it denies the laws of physics the result will be meh
# but this one is reasonable and goal is reachable
state = [0.45, 0.00]
log_states = []
log_actions = []
log_rewards = []
log_next_states = []
for i in range(100):
    print(state)
    actions, stats, root = mcts.run_mcts(state)
    print(actions)
    visualize_tree(root, 2)
    for a in actions:
        (
            next_state,
            reward,
        ) = agent.actual_transition(state, None, a)
        log_states.append(state)
        log_actions.append(a)
        log_rewards.append(reward)
        log_next_states.append(next_state)
        if next_state[0] >= 0.55:
            print("finish")
            break
        state = next_state

df = pd.DataFrame([], columns=["X", "vel", "action"])
df["X"] = [row[0] for row in log_states]
df["vel"] = [row[1] for row in log_states]
df["action"] = log_actions
df.to_csv("./mcts_results_upd_efe.csv", sep="|")
print("done")
