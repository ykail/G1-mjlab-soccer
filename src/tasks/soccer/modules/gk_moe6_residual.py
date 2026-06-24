"""MoE6 frozen-base policy plus a deployable learned residual.

The base MoE6 bundle keeps the strong specialist/gate behavior intact.  This
wrapper adds a small zero-initialized residual head on top of the selected MoE
action, intended for offline distillation from repair-oracle demonstrations.
"""

from __future__ import annotations

import tempfile

import torch
import torch.nn as nn

from rsl_rl.modules import MLP
from src.tasks.soccer.modules.gk_moe6 import GoalkeeperMoE6Policy


_BASE_OBS_DIM = 960
_HISTORY_LEN = 10
_BALL_TERM_DIM = 3
_FEATURE_DIM = 8
_CONTROL_DT = 0.02
_GRAVITY = 9.81


class GoalkeeperMoE6ResidualPolicy(nn.Module):
  """Frozen MoE6 action plus a small residual head."""

  def __init__(
    self,
    env,
    bundle: dict,
    device: str,
    hidden_dims=(512, 256, 128),
    activation="elu",
    residual_scale: float = 0.18,
    residual_regions: tuple[int, ...] = (1, 2, 3, 5),
  ) -> None:
    super().__init__()
    self.device = device
    self.residual_scale = float(residual_scale)
    self._temp_dir = tempfile.mkdtemp(prefix="gk_moe6_residual_")
    # Use the exact deployable MoE6 implementation for the frozen base.  The
    # previous residual wrapper rebuilt the gate from observation history, which
    # did not match the original env-state gate and could destroy base behavior.
    self.base_policy = GoalkeeperMoE6Policy(bundle, env, device)
    self.residual = MLP(_BASE_OBS_DIM + _FEATURE_DIM, 29, hidden_dims, activation)
    for module in reversed([m for m in self.residual.modules() if isinstance(m, nn.Linear)]):
      nn.init.zeros_(module.weight)
      nn.init.zeros_(module.bias)
      break
    self.register_buffer("_marker", torch.ones(1), persistent=True)
    self.register_buffer(
      "_residual_regions",
      torch.tensor(tuple(residual_regions), dtype=torch.long),
      persistent=True,
    )

  def reset(self, dones=None) -> None:
    reset = getattr(self.base_policy, "reset", None)
    if reset is not None:
      try:
        reset(dones)
      except TypeError:
        reset()

  @staticmethod
  def ballistic_features_from_history(obs_history: torch.Tensor) -> torch.Tensor:
    if obs_history.shape[-1] < _HISTORY_LEN * _BALL_TERM_DIM:
      return torch.zeros(
        obs_history.shape[0],
        _FEATURE_DIM,
        dtype=obs_history.dtype,
        device=obs_history.device,
      )
    ball_hist = obs_history[:, : _HISTORY_LEN * _BALL_TERM_DIM].reshape(
      obs_history.shape[0], _HISTORY_LEN, _BALL_TERM_DIM
    )
    pos = ball_hist[:, -1]
    prev = ball_hist[:, -4]
    vel = (pos - prev) / (3.0 * _CONTROL_DT)
    vx = vel[:, 0]

    def plane_time(plane_x: float) -> torch.Tensor:
      safe_vx = torch.where(vx < -1.0e-3, vx, torch.full_like(vx, -1.0e-3))
      raw_t = (plane_x - pos[:, 0]) / safe_vx
      valid = (vx < -1.0e-3) & (raw_t >= 0.0)
      t = torch.where(valid, raw_t, torch.full_like(raw_t, 2.0))
      return torch.clamp(t, 0.0, 2.0)

    t_keeper = plane_time(0.0)
    t_goal = plane_time(-0.5)

    def cross(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
      y = pos[:, 1] + vel[:, 1] * t
      z = pos[:, 2] + vel[:, 2] * t - 0.5 * _GRAVITY * t * t
      return y, torch.clamp(z, min=0.0)

    y_keeper, z_keeper = cross(t_keeper)
    y_goal, z_goal = cross(t_goal)
    speed = torch.linalg.norm(vel, dim=-1)
    features = torch.stack(
      (
        t_keeper / 2.0,
        y_keeper / 1.5,
        z_keeper / 1.8,
        t_goal / 2.0,
        y_goal / 1.5,
        z_goal / 1.8,
        vx / 8.0,
        speed / 8.0,
      ),
      dim=-1,
    )
    return torch.nan_to_num(features, nan=0.0, posinf=3.0, neginf=-3.0).clamp(-3.0, 3.0)

  def base_action(
    self,
    obs: dict[str, torch.Tensor],
    *,
    use_latch: bool | None = None,
  ) -> torch.Tensor:
    del use_latch
    return self.base_policy(obs)

  def residual_delta(self, actor_obs: torch.Tensor) -> torch.Tensor:
    features = self.ballistic_features_from_history(actor_obs)
    residual = self.residual(torch.cat((actor_obs, features), dim=-1))
    return self.residual_scale * torch.tanh(residual)

  def _residual_mask(self, base: torch.Tensor) -> torch.Tensor:
    regions = getattr(self.base_policy, "latched", None)
    if (
      regions is None
      or regions.numel() != base.shape[0]
      or self._residual_regions.numel() == 0
    ):
      return torch.ones(base.shape[0], 1, dtype=base.dtype, device=base.device)
    active = regions >= 0
    selected = (regions[:, None] == self._residual_regions.to(regions.device)[None, :]).any(1)
    return (active & selected).to(dtype=base.dtype).unsqueeze(1)

  def forward(
    self,
    obs: dict[str, torch.Tensor],
    *,
    use_latch: bool | None = None,
  ) -> torch.Tensor:
    actor_obs = obs["actor"]
    with torch.no_grad():
      base = self.base_action(obs, use_latch=use_latch)
    residual = self.residual_delta(actor_obs) * self._residual_mask(base)
    return base + residual

  def forward_from_base_action(
    self,
    actor_obs: torch.Tensor,
    base_action: torch.Tensor,
  ) -> torch.Tensor:
    """Offline distillation path using the recorded MoE6 base action.

    Repair shards contain observation/action pairs from many forced scenarios,
    but they do not carry a live simulator state for the MoE6 env-state gate.
    Using the recorded base action keeps offline training aligned with the
    exact base behavior observed during repair collection.
    """
    return base_action + self.residual_delta(actor_obs)

  def __call__(self, obs):
    return self.forward(obs)
