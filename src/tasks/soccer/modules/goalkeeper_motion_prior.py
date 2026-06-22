"""Goalkeeper motion-prior utilities for AMP training.

The released goalkeeper motion clips are six region-specific G1 motions with a
21-joint subset.  AMP compares policy and expert transitions in the same
feature order: absolute joint positions at t and t+1 for those 21 joints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from src.tasks.soccer.mdp.goalkeeper_obs import _GK_JOINT_NAMES, _REF_DEFAULT_DOF_POS


GOALKEEPER_REGION_MOTION_NAMES: tuple[str, ...] = (
  "lefthand",
  "righthand",
  "leftjump",
  "rightjump",
  "leftstep",
  "rightstep",
)

_CRITIC_JOINT_POS = slice(9, 38)
_CRITIC_REGION = 112


def _extract_tensor(obs, group: str) -> torch.Tensor:
  if isinstance(obs, dict):
    return obs[group]
  if hasattr(obs, "get"):
    return obs.get(group)
  return getattr(obs, group)


def _region_from_observations(observations) -> torch.Tensor:
  """Extract integer region labels from the current 113D critic observation.

  Current critic order after the 96D actor slice is:
    base_lin_vel 96:99, ball_vel 99:102, ee 102:108, dist 108,
    end_target 109:112, end_region 112.
  """
  critic_obs = _extract_tensor(observations, "critic")
  return torch.clamp(torch.round(critic_obs[:, _CRITIC_REGION] * 3.0), 0, 5).long()


def build_amp_state_from_observations(
  observations,
  next_observations,
  joint_indices: torch.Tensor,
  default_joint_pos: torch.Tensor,
) -> torch.Tensor:
  """Build a policy AMP transition ``[joint_pos_t, joint_pos_t+1]``."""
  critic_obs = _extract_tensor(observations, "critic")
  next_critic_obs = _extract_tensor(next_observations, "critic")

  joint_indices = joint_indices.to(device=critic_obs.device, dtype=torch.long)
  default_joint_pos = default_joint_pos.to(
    device=critic_obs.device,
    dtype=critic_obs.dtype,
  )

  def _frame_state(critic: torch.Tensor) -> torch.Tensor:
    joint_rel = critic[:, _CRITIC_JOINT_POS]
    return joint_rel[:, joint_indices] + default_joint_pos[joint_indices]

  return torch.cat([_frame_state(critic_obs), _frame_state(next_critic_obs)], dim=-1)


class GoalkeeperMotionPrior:
  """Region-conditioned expert transition sampler."""

  def __init__(
    self,
    motion_dir: str | Path,
    motion_names: Iterable[str] = GOALKEEPER_REGION_MOTION_NAMES,
    fps: float = 30.0,
    env_fps: float = 50.0,
    num_steps: int = 2,
    device: str | torch.device = "cpu",
  ) -> None:
    self.motion_dir = self._resolve_motion_dir(Path(motion_dir))
    self.device = torch.device(device)
    self.motion_names = tuple(motion_names)
    self.fps = float(fps)
    self.env_fps = float(env_fps)
    self.num_steps = int(num_steps)
    if len(self.motion_names) != 6:
      raise ValueError("goalkeeper AMP expects six region motion slots")

    joint_names = self._load_joint_names(self.motion_dir / "joint_id.txt")
    full_name_to_index = {name: idx for idx, name in enumerate(_GK_JOINT_NAMES)}
    try:
      full_indices = [full_name_to_index[name] for name in joint_names]
    except KeyError as exc:
      raise ValueError(f"unknown goalkeeper prior joint name: {exc}") from exc

    self.joint_names = tuple(joint_names)
    self.full_joint_indices = torch.as_tensor(
      full_indices,
      dtype=torch.long,
      device=self.device,
    )
    self.full_default_joint_pos = torch.as_tensor(
      _REF_DEFAULT_DOF_POS,
      dtype=torch.float32,
      device=self.device,
    )

    lengths = []
    motions = []
    for name in self.motion_names:
      path = self.motion_dir / f"{name}.pt"
      if not path.exists():
        raise FileNotFoundError(f"goalkeeper motion prior not found: {path}")
      data = torch.load(path, map_location=self.device, weights_only=False)
      if not isinstance(data, dict) or "joint_position" not in data:
        raise ValueError(f"{path} missing motion prior tensor: joint_position")
      joint_pos = data["joint_position"].to(device=self.device, dtype=torch.float32)
      if joint_pos.ndim != 2 or joint_pos.shape[1] != len(joint_names):
        raise ValueError(
          f"{path} joint_position must have shape (T, {len(joint_names)}); "
          f"got {tuple(joint_pos.shape)}"
        )
      if joint_pos.shape[0] < 2:
        raise ValueError(f"{path} must contain at least two frames")
      motions.append(joint_pos)
      lengths.append(joint_pos.shape[0])

    self._motions = motions
    self.lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
    self.num_regions = len(self.motion_names)
    self.joint_dim = len(joint_names)
    self.frame_dim = self.joint_dim
    self.transition_dim = self.frame_dim * 2

  @staticmethod
  def _resolve_motion_dir(motion_dir: Path) -> Path:
    if motion_dir.exists():
      return motion_dir
    if not motion_dir.is_absolute():
      repo_root = Path(__file__).resolve().parents[4]
      repo_relative = repo_root / motion_dir
      if repo_relative.exists():
        return repo_relative
    return motion_dir

  @staticmethod
  def _load_joint_names(path: Path) -> list[str]:
    if not path.exists():
      raise FileNotFoundError(f"goalkeeper joint-id file not found: {path}")
    joint_names: list[str] = []
    for line in path.read_text().splitlines():
      stripped = line.strip()
      if not stripped:
        continue
      parts = stripped.split(maxsplit=1)
      if len(parts) != 2:
        raise ValueError(f"invalid joint_id line: {line!r}")
      joint_names.append(parts[1])
    return joint_names

  def sample_expert_transitions(self, regions: torch.Tensor) -> torch.Tensor:
    """Sample random-ratio expert ``[s_t, s_t+1]`` transitions."""
    regions = torch.clamp(regions.to(device=self.device, dtype=torch.long), 0, 5)
    samples = torch.empty(
      regions.shape[0],
      self.transition_dim,
      dtype=torch.float32,
      device=self.device,
    )
    for region in torch.unique(regions):
      mask = regions == region
      count = int(mask.sum().item())
      motion = self._motions[int(region.item())]
      max_start = max(motion.shape[0] - self.num_steps, 1)
      frame_ids = torch.randint(0, max_start, (count,), device=self.device)
      first = motion[frame_ids]
      ratio = self.fps / self.env_fps
      ratio = ratio * (torch.rand(count, device=self.device) + 0.25)
      next_pos = frame_ids.to(dtype=torch.float32) + ratio
      floor = torch.floor(next_pos).long().clamp(0, motion.shape[0] - 1)
      ceil = (floor + 1).clamp(0, motion.shape[0] - 1)
      linear_ratio = (next_pos - floor.to(dtype=torch.float32)).unsqueeze(-1)
      second = motion[floor] * (1.0 - linear_ratio) + motion[ceil] * linear_ratio
      samples[mask] = torch.cat([first, second], dim=-1)
    return samples
