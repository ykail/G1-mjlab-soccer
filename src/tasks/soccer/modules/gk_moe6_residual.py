"""MoE6 frozen-base policy plus a deployable learned residual.

The base MoE6 bundle keeps the strong specialist/gate behavior intact.  This
wrapper adds a small zero-initialized residual head on top of the selected MoE
action, intended for offline distillation from repair-oracle demonstrations.
"""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.modules import MLP
from src.tasks.soccer.modules.symmetry import mirror_action, mirror_obs


_BASE_OBS_DIM = 960
_HISTORY_LEN = 10
_BALL_TERM_DIM = 3
_FEATURE_DIM = 8
_CONTROL_DT = 0.02
_GRAVITY = 9.81


def _load_expert_actor(env, checkpoint: dict, device: str, temp_dir: str):
  from src.tasks.soccer.config.g1.gk_train_cfg import (
    goalkeeper_ballistic_residual_runner_cfg,
    goalkeeper_train_runner_cfg,
  )

  if checkpoint.get("ballistic_residual"):
    import src.tasks.soccer.modules.gk_ballistic_residual as gkbr

    meta = checkpoint["ballistic_residual"]
    gkbr.BASE_CKPT = meta.get("base")
    gkbr.BASE_HIDDEN = tuple(meta.get("base_hidden", (1024, 512, 256)))
    gkbr.RESIDUAL_SCALE = float(meta.get("residual_scale", 0.25))
    agent_cfg = goalkeeper_ballistic_residual_runner_cfg()
  else:
    agent_cfg = goalkeeper_train_runner_cfg()

  path = Path(temp_dir) / f"expert_{id(checkpoint)}.pt"
  torch.save(checkpoint, path)
  runner = MjlabOnPolicyRunner(env, asdict(agent_cfg), device=device)
  runner.load(str(path), load_cfg={"actor": True})
  actor = runner.alg.actor
  actor.eval()
  for param in actor.parameters():
    param.requires_grad_(False)
  return actor


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
  ) -> None:
    super().__init__()
    self.device = device
    self.residual_scale = float(residual_scale)
    self.z_low = float(bundle.get("z_low", 0.85))
    self.z_up = float(bundle.get("z_up", 1.35))
    self.vz_low = float(bundle.get("vz_low", -99.0))
    self.latch_hi = float(bundle.get("latch_hi", 5.0))
    self.land_x = float(bundle.get("land_x", 0.0))
    self.mirror_map = self._parse_mirror_map(str(bundle.get("mirror_map", "")))
    self._temp_dir = tempfile.mkdtemp(prefix="gk_moe6_residual_")

    experts = bundle.get("sr")
    if not isinstance(experts, (list, tuple)) or len(experts) != 6:
      raise ValueError("MoE6 residual base expects bundle['sr'] with six experts")
    self.experts = nn.ModuleList(
      [_load_expert_actor(env, expert, device, self._temp_dir) for expert in experts]
    )
    self.residual = MLP(_BASE_OBS_DIM + _FEATURE_DIM, 29, hidden_dims, activation)
    for module in reversed([m for m in self.residual.modules() if isinstance(m, nn.Linear)]):
      nn.init.zeros_(module.weight)
      nn.init.zeros_(module.bias)
      break
    self.register_buffer("_marker", torch.ones(1), persistent=True)
    self.register_buffer("_latched", torch.empty(0, dtype=torch.long), persistent=False)

  @staticmethod
  def _parse_mirror_map(raw: str) -> dict[int, int]:
    out: dict[int, int] = {}
    if not raw:
      return out
    for item in raw.split(","):
      if not item:
        continue
      dst, src = (int(x) for x in item.split(":"))
      if not (0 <= dst < 6 and 0 <= src < 6):
        raise ValueError(f"mirror_map entries must be in [0, 5], got {item}")
      out[dst] = src
    return out

  def reset(self, dones=None) -> None:
    if dones is None or self._latched.numel() == 0:
      self._latched = torch.empty(0, dtype=torch.long, device=self._marker.device)
    elif self._latched.numel() == dones.numel():
      self._latched = torch.where(
        dones.to(device=self._latched.device).bool().view(-1),
        torch.full_like(self._latched, -1),
        self._latched,
      )
    for expert in self.experts:
      reset = getattr(expert, "reset", None)
      if reset is not None:
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

  def gate_region(self, obs_history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ball_hist = obs_history[:, : _HISTORY_LEN * _BALL_TERM_DIM].reshape(
      obs_history.shape[0], _HISTORY_LEN, _BALL_TERM_DIM
    )
    pos = ball_hist[:, -1]
    prev = ball_hist[:, -4]
    vel = (pos - prev) / (3.0 * _CONTROL_DT)
    bx = pos[:, 0]
    vx = vel[:, 0]
    valid = (vx < -1.0) & (bx > 0.2) & (bx < self.latch_hi)
    t = torch.clamp(-(bx - self.land_x) / (vx - 1.0e-3), 0.0, 2.0)
    cy = pos[:, 1] + vel[:, 1] * t
    cz = pos[:, 2] + vel[:, 2] * t - 0.5 * _GRAVITY * t * t
    vz_cross = vel[:, 2] - _GRAVITY * t
    base = torch.zeros_like(bx, dtype=torch.long)
    base = torch.where(cz < self.z_low, torch.full_like(base, 4), base)
    base = torch.where(cz > self.z_up, torch.full_like(base, 2), base)
    base = torch.where(vz_cross < self.vz_low, torch.full_like(base, 4), base)
    return base + (cy < 0).long(), valid

  def _expert_action(self, region: int, actor_obs: torch.Tensor) -> torch.Tensor:
    if region in self.mirror_map:
      src = self.mirror_map[region]
      mirrored = mirror_obs(actor_obs)
      return mirror_action(self.experts[src]({"actor": mirrored}, stochastic_output=False))
    return self.experts[region]({"actor": actor_obs}, stochastic_output=False)

  def base_action(
    self,
    obs: dict[str, torch.Tensor],
    *,
    use_latch: bool | None = None,
  ) -> torch.Tensor:
    actor_obs = obs["actor"]
    region, valid = self.gate_region(actor_obs)
    if use_latch is None:
      use_latch = (not self.training) and (
        self._latched.numel() in (0, actor_obs.shape[0])
      )
    if use_latch:
      if self._latched.numel() != actor_obs.shape[0]:
        self._latched = torch.full(
          (actor_obs.shape[0],),
          -1,
          dtype=torch.long,
          device=actor_obs.device,
        )
      new_latch = valid & (self._latched < 0)
      self._latched = torch.where(new_latch, region, self._latched)
      use_region = torch.where(self._latched < 0, torch.zeros_like(self._latched), self._latched)
    else:
      use_region = torch.where(valid, region, torch.zeros_like(region))
    actions = torch.stack([self._expert_action(idx, actor_obs) for idx in range(6)], dim=0)
    return actions[use_region, torch.arange(actions.shape[1], device=actor_obs.device)]

  def forward(
    self,
    obs: dict[str, torch.Tensor],
    *,
    use_latch: bool | None = None,
  ) -> torch.Tensor:
    actor_obs = obs["actor"]
    with torch.no_grad():
      base = self.base_action(obs, use_latch=use_latch)
    features = self.ballistic_features_from_history(actor_obs)
    residual = self.residual(torch.cat((actor_obs, features), dim=-1))
    return base + self.residual_scale * torch.tanh(residual)

  def __call__(self, obs):
    return self.forward(obs)
