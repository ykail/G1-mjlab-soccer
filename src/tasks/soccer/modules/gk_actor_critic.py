"""Goalkeeper ActorCritic with HIMPPO-style estimator heads.

The actor observes a 10-frame 960D history and learns two deployable latent
predictions from that history:
  - 6D ball target: interception/landing position plus ball velocity
  - 6-way goalkeeper region logits

Those estimates are concatenated into the policy input, so training can use
critic-only supervision while inference still needs only actor observations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _get_activation(act_name: str) -> nn.Module:
  if act_name == "elu":
    return nn.ELU()
  if act_name == "relu":
    return nn.ReLU()
  if act_name == "selu":
    return nn.SELU()
  if act_name == "lrelu":
    return nn.LeakyReLU()
  if act_name == "tanh":
    return nn.Tanh()
  if act_name == "sigmoid":
    return nn.Sigmoid()
  return nn.ELU()


class _GoalkeeperGaussianDistribution(nn.Module):
  """Small RSL-RL-compatible diagonal Gaussian.

  Older goalkeeper checkpoints used a root-level ``std`` parameter, while
  RSL-RL 5.x expects distribution parameters on a submodule.  This wrapper
  keeps the runtime API explicit and the state_dict migration simple.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float = 1.0,
    min_std: float | None = None,
    max_std: float | None = None,
  ) -> None:
    super().__init__()
    self.std_param = nn.Parameter(init_std * torch.ones(output_dim))
    self.min_std = min_std
    self.max_std = max_std
    self._distribution: Normal | None = None
    Normal.set_default_validate_args(False)

  def update(self, mean: torch.Tensor) -> None:
    std = self.std_param
    if self.min_std is not None or self.max_std is not None:
      std = torch.clamp(std, min=self.min_std, max=self.max_std)
    self._distribution = Normal(mean, std.expand_as(mean))

  @property
  def _dist(self) -> Normal:
    if self._distribution is None:
      raise RuntimeError("distribution was queried before update()")
    return self._distribution

  def sample(self) -> torch.Tensor:
    return self._dist.sample()

  @property
  def mean(self) -> torch.Tensor:
    return self._dist.mean

  @property
  def std(self) -> torch.Tensor:
    return self._dist.stddev

  @property
  def entropy(self) -> torch.Tensor:
    return self._dist.entropy().sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, torch.Tensor]:
    return (self.mean, self.std)

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self._dist.log_prob(outputs).sum(dim=-1)

  @staticmethod
  def kl_divergence(old_params, new_params) -> torch.Tensor:
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    old_dist = Normal(old_mean, old_std)
    new_dist = Normal(new_mean, new_std)
    return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class GoalkeeperActorCritic(nn.Module):
  """Actor-critic model matching the Humanoid-Goalkeeper layout."""

  is_recurrent = False

  _TERM_SIZES: tuple[int, ...] = (3, 3, 3, 29, 29, 29)
  _HISTORY_LEN: int = 10
  _ONE_STEP_DIM: int = sum(_TERM_SIZES)  # 96

  def __init__(
    self,
    obs,
    obs_groups=None,
    group_name="actor",
    num_actions=29,
    num_one_step_obs=96,
    num_critic_obs=113,
    num_actor_obs=960,
    actor_history_length=10,
    hidden_dims=None,
    actor_hidden_dims=(512, 256, 256),
    critic_hidden_dims=(512, 256, 256),
    activation="elu",
    obs_normalization=False,
    distribution_cfg=None,
    init_noise_std=1.0,
    verbose=False,
    **kwargs,
  ) -> None:
    if kwargs:
      print(
        "GoalkeeperActorCritic.__init__ got unexpected arguments: "
        + str(sorted(kwargs.keys()))
      )
    super().__init__()
    self.group_name = group_name
    self.obs_normalization = obs_normalization

    if hidden_dims is not None:
      if group_name == "critic":
        critic_hidden_dims = tuple(hidden_dims)
      else:
        actor_hidden_dims = tuple(hidden_dims)

    min_noise_std = None
    max_noise_std = None
    if distribution_cfg is not None:
      init_noise_std = distribution_cfg.get("init_std", init_noise_std)
      min_noise_std = distribution_cfg.get("min_std", None)
      max_noise_std = distribution_cfg.get("max_std", None)

    def _feature_dim(space_or_tensor, fallback: int) -> int:
      shape = getattr(space_or_tensor, "shape", None)
      if shape is None or len(shape) == 0:
        return fallback
      return int(shape[-1])

    if hasattr(obs, "spaces"):
      actor_space = obs.spaces.get("actor")
      critic_space = obs.spaces.get("critic")
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = _feature_dim(actor_space, num_actor_obs)
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = _feature_dim(critic_space, num_critic_obs)
    elif isinstance(obs, dict) or hasattr(obs, "get"):
      actor_space = obs.get("actor", None)
      critic_space = obs.get("critic", None)
      if actor_space is not None and hasattr(actor_space, "shape"):
        num_actor_obs = _feature_dim(actor_space, num_actor_obs)
        num_one_step_obs = num_actor_obs // actor_history_length
      if critic_space is not None and hasattr(critic_space, "shape"):
        num_critic_obs = _feature_dim(critic_space, num_critic_obs)

    self.num_actor_obs = num_actor_obs
    self.num_critic_obs = num_critic_obs
    self.num_one_step_obs = num_one_step_obs
    self.actor_history_length = actor_history_length
    self.num_actions = num_actions
    if (
      num_one_step_obs != self._ONE_STEP_DIM
      or actor_history_length != self._HISTORY_LEN
      or num_actor_obs != self._ONE_STEP_DIM * self._HISTORY_LEN
    ):
      raise ValueError(
        "goalkeeper actor history layout must be "
        f"{self._HISTORY_LEN}x{self._ONE_STEP_DIM} "
        f"({self._ONE_STEP_DIM * self._HISTORY_LEN}D); got "
        f"{actor_history_length}x{num_one_step_obs} ({num_actor_obs}D)"
      )

    self.history_latent_dim = 16
    self.estimate_ball_dim = 6
    self.num_regions = 6
    self.distribution = _GoalkeeperGaussianDistribution(
      num_actions,
      init_noise_std,
      min_std=min_noise_std,
      max_std=max_noise_std,
    )

    history_input_dim = num_one_step_obs * actor_history_length
    self.history_encoder = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, self.history_latent_dim),
    )
    self.ball_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.estimate_ball_dim),
    )
    self.region_estimator = nn.Sequential(
      nn.Linear(history_input_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, self.num_regions),
    )

    act_fn = _get_activation(activation)
    mlp_input_dim_a = (
      num_one_step_obs + self.history_latent_dim + self.estimate_ball_dim + 1
    )
    self.num_actor_input = mlp_input_dim_a

    actor_layers: list[nn.Module] = [nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]), act_fn]
    for i in range(len(actor_hidden_dims)):
      if i == len(actor_hidden_dims) - 1:
        actor_layers.append(nn.Linear(actor_hidden_dims[i], num_actions))
      else:
        actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
        actor_layers.append(act_fn)
    self.actor = nn.Sequential(*actor_layers)

    critic_layers: list[nn.Module] = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), act_fn]
    for i in range(len(critic_hidden_dims)):
      if i == len(critic_hidden_dims) - 1:
        critic_layers.append(nn.Linear(critic_hidden_dims[i], 1))
      else:
        critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
        critic_layers.append(act_fn)
    self.critic = nn.Sequential(*critic_layers)

    self.estimate_ball: torch.Tensor | None = None
    self.estimate_region: torch.Tensor | None = None

    if verbose:
      print(f"Actor MLP: {self.actor}")
      print(f"Critic MLP: {self.critic}")
      print(f"History MLP: {self.history_encoder}")
      print(f"Ball MLP: {self.ball_estimator}")
      print(f"Region MLP: {self.region_estimator}")

  def load_state_dict(self, state_dict, strict=True, assign=False):
    state_dict = state_dict.copy()
    if "std" in state_dict and "distribution.std_param" not in state_dict:
      state_dict["distribution.std_param"] = state_dict.pop("std")
    else:
      state_dict.pop("std", None)
    if "log_std" in state_dict:
      log_std = state_dict.pop("log_std")
      state_dict.setdefault("distribution.std_param", log_std.exp())
    if "distribution.log_std_param" in state_dict:
      log_std = state_dict.pop("distribution.log_std_param")
      state_dict.setdefault("distribution.std_param", log_std.exp())
    try:
      return super().load_state_dict(state_dict, strict=strict, assign=assign)
    except TypeError:
      return super().load_state_dict(state_dict, strict=strict)

  def reset(self, dones=None, hidden_state=None) -> None:
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones=None) -> None:
    del dones

  def update_normalization(self, obs) -> None:
    del obs

  def _extract_tensor(self, obs, group="actor") -> torch.Tensor:
    if isinstance(obs, (tuple, list)):
      return self._extract_tensor(obs[0], group)
    if isinstance(obs, dict):
      x = obs.get(group, obs)
    elif hasattr(obs, "get") and group in obs:
      x = obs[group]
    else:
      x = obs
    if hasattr(x, "get") and not isinstance(x, torch.Tensor):
      x = x[group] if group in x else x
    if isinstance(x, torch.Tensor) and x.dim() == 1:
      x = x.unsqueeze(0)
    return x

  def _reorder_obs_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    """Transpose mjlab term-major history to frame-major policy history."""
    batch = obs_history.shape[0]
    chunks = []
    offset = 0
    for size in self._TERM_SIZES:
      block = obs_history[:, offset : offset + self._HISTORY_LEN * size]
      chunks.append(block.reshape(batch, self._HISTORY_LEN, size))
      offset += self._HISTORY_LEN * size
    return torch.cat(chunks, dim=-1).reshape(batch, -1)

  def _actor_mean_from_history(self, obs_history: torch.Tensor) -> torch.Tensor:
    obs_history = self._reorder_obs_history(obs_history)
    history_latent = self.history_encoder(obs_history)
    self.estimate_ball = self.ball_estimator(obs_history)
    self.estimate_region = self.region_estimator(obs_history)
    estimated_region = torch.argmax(self.estimate_region, dim=-1, keepdim=True).to(
      dtype=obs_history.dtype
    ) / 3.0
    actor_input = torch.cat(
      (
        obs_history[:, -self.num_one_step_obs :],
        history_latent,
        self.estimate_ball,
        estimated_region,
      ),
      dim=-1,
    )
    return self.actor(actor_input)

  def forward(
    self,
    obs,
    masks=None,
    hidden_state=None,
    stochastic_output=False,
    **kwargs,
  ):
    del hidden_state, kwargs
    if self.group_name == "critic":
      return self.evaluate(obs)

    obs_history = self._extract_tensor(obs, group="actor")
    if masks is not None and not self.is_recurrent:
      from rsl_rl.utils import unpad_trajectories

      obs_history = unpad_trajectories(obs_history, masks)
    if stochastic_output:
      self.update_distribution(obs_history)
      return self.distribution.sample()
    return self.act_inference(obs_history)

  @property
  def action_mean(self) -> torch.Tensor:
    return self.distribution.mean

  @property
  def action_std(self) -> torch.Tensor:
    return self.distribution.std

  @property
  def entropy(self) -> torch.Tensor:
    return self.distribution.entropy

  @property
  def output_mean(self) -> torch.Tensor:
    return self.action_mean

  @property
  def output_std(self) -> torch.Tensor:
    return self.action_std

  @property
  def output_entropy(self) -> torch.Tensor:
    return self.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, torch.Tensor]:
    return self.distribution.params

  def update_distribution(self, obs_history: torch.Tensor) -> None:
    self.distribution.update(self._actor_mean_from_history(obs_history))

  def act(self, obs_history=None, **kwargs):
    del kwargs
    self.update_distribution(obs_history)
    return self.distribution.sample(), self.estimate_ball, self.estimate_region

  def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
    return self.distribution.log_prob(actions)

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    return self.get_actions_log_prob(outputs)

  def get_kl_divergence(self, old_params, new_params) -> torch.Tensor:
    return self.distribution.kl_divergence(old_params, new_params)

  def estimate_ball_and_region(
    self,
    obs_history: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    obs_history = self._reorder_obs_history(obs_history)
    return self.ball_estimator(obs_history), self.region_estimator(obs_history)

  def compute_estimator_loss(
    self,
    obs_history: torch.Tensor,
    ball_target: torch.Tensor,
    region_target: torch.Tensor,
    ball_loss_coef: float = 1.0,
    region_loss_coef: float = 1.0,
  ) -> dict[str, torch.Tensor]:
    estimate_ball, estimate_region = self.estimate_ball_and_region(obs_history)
    region_target = region_target.to(device=estimate_region.device, dtype=torch.long)
    ball_target = ball_target.to(device=estimate_ball.device, dtype=estimate_ball.dtype)
    ball_loss = F.mse_loss(estimate_ball, ball_target)
    region_loss = F.cross_entropy(estimate_region, region_target)
    total = ball_loss_coef * ball_loss + region_loss_coef * region_loss
    return {"ball": ball_loss, "region": region_loss, "total": total}

  def act_inference(self, obs_history, observations=None) -> torch.Tensor:
    del observations
    return self._actor_mean_from_history(obs_history)

  def evaluate(self, critic_observations, **kwargs) -> torch.Tensor:
    del kwargs
    x = self._extract_tensor(critic_observations, group="critic")
    return self.critic(x)
