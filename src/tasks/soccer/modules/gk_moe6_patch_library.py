"""Policy wrapper that applies nearest-neighbor open-loop repair patches."""

from __future__ import annotations

import torch


class GoalkeeperPatchLibraryPolicy:
  """Base policy plus a strict nearest-neighbor residual sequence library."""

  def __init__(
    self,
    base_policy,
    library: dict,
    env,
    device: str,
    pos_scale: float = 0.05,
    vel_scale: float = 0.15,
    threshold: float = 1.0,
    residual_scale: float = 1.0,
    fade_steps: int = 0,
  ) -> None:
    self.base_policy = base_policy
    self.device = device
    self.ball = env.unwrapped.scene["ball"]
    self.origins = env.unwrapped.scene.env_origins
    self.pos_scale = float(pos_scale)
    self.vel_scale = float(vel_scale)
    self.threshold = float(threshold)
    self.residual_scale = float(residual_scale)
    self.fade_steps = int(fade_steps)
    self.env = env.unwrapped
    self.start = torch.as_tensor(library["start"], dtype=torch.float32, device=device)
    self.vel = torch.as_tensor(library["vel"], dtype=torch.float32, device=device)
    self.region = torch.as_tensor(library["region"], dtype=torch.long, device=device)
    self.residual_seq = torch.as_tensor(library["residual_seq"], dtype=torch.float32, device=device)
    self.reset()

  def reset(self, dones=None) -> None:
    reset = getattr(self.base_policy, "reset", None)
    if reset is not None:
      try:
        reset(dones)
      except TypeError:
        reset()
    n = self.origins.shape[0]
    self._patch_idx = torch.full((n,), -1, dtype=torch.long, device=self.device)
    self._patch_step = torch.zeros((n,), dtype=torch.long, device=self.device)
    self._armed = torch.zeros((n,), dtype=torch.bool, device=self.device)

  def _nearest_patch(self) -> torch.Tensor:
    n = self.origins.shape[0]
    if self.start.numel() == 0:
      return torch.full((n,), -1, dtype=torch.long, device=self.device)
    pos = self.ball.data.root_link_pos_w - self.origins
    vel = self.ball.data.root_link_lin_vel_w
    d_pos = (pos[:, None, :] - self.start[None, :, :]) / max(self.pos_scale, 1.0e-6)
    d_vel = (vel[:, None, :] - self.vel[None, :, :]) / max(self.vel_scale, 1.0e-6)
    dist = torch.sqrt(d_pos.pow(2).sum(-1) + d_vel.pow(2).sum(-1))
    current_region = getattr(self.env, "_gk_region", None)
    if current_region is not None and current_region.numel() == n:
      region_match = current_region.to(device=self.device).long()[:, None] == self.region[None, :]
      dist = torch.where(region_match, dist, torch.full_like(dist, float("inf")))
    best_dist, best_idx = dist.min(1)
    return torch.where(best_dist <= self.threshold, best_idx, torch.full_like(best_idx, -1))

  def __call__(self, obs):
    with torch.inference_mode():
      action = self.base_policy(obs)
      new_idx = self._nearest_patch()
      should_arm = (~self._armed) & (new_idx >= 0)
      self._patch_idx = torch.where(should_arm, new_idx, self._patch_idx)
      self._patch_step = torch.where(should_arm, torch.zeros_like(self._patch_step), self._patch_step)
      self._armed |= should_arm
      active = self._armed & (self._patch_idx >= 0)
      if active.any():
        t = torch.clamp(self._patch_step[active], max=self.residual_seq.shape[1] - 1)
        residual = self.residual_seq[self._patch_idx[active], t]
        if self.fade_steps > 0:
          fade = torch.clamp(
            1.0 - (self._patch_step[active].float() - self.residual_seq.shape[1]) / self.fade_steps,
            min=0.0,
            max=1.0,
          ).unsqueeze(1)
          residual = residual * fade
        action[active] = action[active] + self.residual_scale * residual
        self._patch_step[active] += 1
        done = self._patch_step >= (self.residual_seq.shape[1] + max(0, self.fade_steps))
        self._armed &= ~done
        self._patch_idx = torch.where(done, torch.full_like(self._patch_idx, -1), self._patch_idx)
        self._patch_step = torch.where(done, torch.zeros_like(self._patch_step), self._patch_step)
    return action


def build_patch_library_policy(checkpoint: dict, env, device: str):
  from scripts.eval_naive_goalkeeper import _load_policy

  base_policy = _load_policy(checkpoint["base_checkpoint"], env, device)
  library = checkpoint["library"]
  return GoalkeeperPatchLibraryPolicy(
    base_policy,
    library,
    env,
    device,
    pos_scale=float(checkpoint.get("pos_scale", 0.05)),
    vel_scale=float(checkpoint.get("vel_scale", 0.15)),
    threshold=float(checkpoint.get("threshold", 1.0)),
    residual_scale=float(checkpoint.get("residual_scale", 1.0)),
    fade_steps=int(checkpoint.get("fade_steps", 0)),
  )
