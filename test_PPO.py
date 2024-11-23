import torch
from torch import nn

from torchrl.envs import Compose, ObservationNorm, DoubleToFloat, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import check_env_specs, set_exploration_type, ExplorationType
from torchrl.modules import ProbabilisticActor, OneHotCategorical, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from tensordict.nn import TensorDictModule

# Taken from https://www.reddit.com/r/reinforcementlearning/comments/1eer8iv/why_is_my_ppo_algorithm_not_learning/

torch.set_printoptions(threshold=16384)

device="cuda" if torch.cuda.is_available() else "cpu"

base_env = GymEnv('CartPole-v1', device=device)

env = TransformedEnv(
    base_env,
    Compose(
        ObservationNorm(in_keys=["observation"]),
        DoubleToFloat(),
        StepCounter()
    )
)

env.transform[0].init_stats(1024)

check_env_specs(env)

actor_net = nn.Sequential(
    nn.Linear(env.observation_spec["observation"].shape[-1], 32, device=device),
    nn.Sigmoid(),
    nn.Linear(32, 32, device=device),
    nn.Sigmoid(),
    nn.Linear(32, env.action_spec.shape[-1], device=device)
)

actor_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["logits"])

actor = ProbabilisticActor(
    module = actor_module,
    spec = env.action_spec,
    in_keys = ["logits"],
    distribution_class = OneHotCategorical,
    return_log_prob = True
)

value_net = nn.Sequential(
    nn.Linear(env.observation_spec["observation"].shape[-1], 16, device=device),
    nn.Sigmoid(),
    nn.Linear(16, 16, device=device),
    nn.Sigmoid(),
    nn.Linear(16, 16, device=device),
    nn.Sigmoid(),
    nn.Linear(16, 1, device=device)
)

value_module = ValueOperator(
    module = value_net,
    in_keys = ["observation"]
)

frames_per_batch = 1024
total_frames = 1048576

collector = SyncDataCollector(
    env,
    actor,
    frames_per_batch = frames_per_batch,
    total_frames = total_frames,
    split_trajs = True,
    reset_at_each_iter = True,
    device=device
)

replay_buffer = ReplayBuffer(
    storage = LazyTensorStorage(max_size=frames_per_batch),
    sampler = SamplerWithoutReplacement()
)

advantage_module = GAE(
    gamma = 0.99,
    lmbda = 0.95,
    value_network = value_module,
    average_gae = True
)

entropy_eps = 1e-4

loss_module = ClipPPOLoss(
    actor_network = actor,
    critic_network = value_module,
    clip_epsilon = 0.2,
    entropy_bonus = bool(entropy_eps),
    entropy_coef = entropy_eps
)

optim = torch.optim.Adam(loss_module.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch)

sub_batch_size = 64

for i, tensordict_data in enumerate(collector):
    for _ in range(8):
        advantage_module(tensordict_data)
        replay_buffer.extend(tensordict_data.reshape(-1).cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            data = replay_buffer.sample(sub_batch_size)
            loss = loss_module(data.to(device))
            loss_value = loss["loss_objective"] + loss["loss_critic"] + loss["loss_entropy"]
            loss_value.backward()
            optim.step()
            optim.zero_grad()
    scheduler.step()
    if i % 16 == 0:
        with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
            rollout = env.rollout(1024, actor)
            print(rollout["next","reward"].sum())
            del rollout


actor = ProbabilisticActor(
    module = actor_module,
    spec = env.action_spec,
    in_keys = ["logits"],
    distribution_class = OneHotCategorical,
    return_log_prob = True
)