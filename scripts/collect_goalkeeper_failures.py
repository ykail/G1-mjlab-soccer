"""Collect eval failures for any loadable goalkeeper checkpoint.

The output CSV is intentionally compatible with ``train_ballistic_residual.py``:
it contains ``true_region``, ``start_x/y/z`` and ``vel_x/y/z`` columns in the
goalkeeper-local frame.  A later residual-RL run can replay these exact failed
ball starts with small jitter.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from scripts.eval_naive_goalkeeper import _load_policy, _make_zero_policy


_REGION_NAMES = [
  "Right-Mid",
  "Left-Mid",
  "Right-Up",
  "Left-Up",
  "Right-Low",
  "Left-Low",
]
_GOAL_X = -0.5
_GOAL_HALF_WIDTH = 1.5
_GOAL_HEIGHT = 1.8
_GRAVITY = 9.81


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/model_repaired_lyk.pt"
  out_csv: str = "logs/keeper_failure_residual/failures.csv"
  task_id: str = "Eval-Goalkeeper"
  num_envs: int = 2048
  batches: int = 16
  steps: int = 150
  max_failures: int = 0
  seed: int = 2810
  device: str = "cuda:0"


def _reset_policy(policy) -> None:
  reset = getattr(policy, "reset", None)
  if reset is None:
    return
  try:
    reset()
  except TypeError:
    reset(None)


def _crossing_features(start_local: torch.Tensor, vel: torch.Tensor) -> tuple[torch.Tensor, ...]:
  vx = vel[:, 0]
  safe_vx = torch.where(vx < -1.0e-3, vx, torch.full_like(vx, -1.0e-3))
  t_goal = (_GOAL_X - start_local[:, 0]) / safe_vx
  t_goal = torch.clamp(torch.where(t_goal >= 0.0, t_goal, torch.full_like(t_goal, 2.0)), 0.0, 2.0)
  y_goal = start_local[:, 1] + vel[:, 1] * t_goal
  z_goal = start_local[:, 2] + vel[:, 2] * t_goal - 0.5 * _GRAVITY * t_goal * t_goal
  return t_goal, y_goal, torch.clamp(z_goal, min=0.0)


def _default_fieldnames() -> list[str]:
  return [
    "batch",
    "env",
    "true_region",
    "true_region_name",
    "enter_step",
    "min_ball_x",
    "start_x",
    "start_y",
    "start_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "goal_t",
    "goal_y",
    "goal_z",
    "final_x",
    "final_y",
    "final_z",
    "final_vx",
    "final_vy",
    "final_vz",
  ]


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  device = cfg.device

  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = cfg.num_envs
  env_cfg.seed = cfg.seed
  env_cfg.terminations.pop("fell_over", None)
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device), clip_actions=100.0)

  policy = (
    _load_policy(cfg.checkpoint, env, device)
    if cfg.checkpoint
    else _make_zero_policy(env, device)
  )
  ball = env.unwrapped.scene["ball"]
  origins = env.unwrapped.scene.env_origins

  rows: list[dict[str, int | float | str]] = []
  region_total = [0] * len(_REGION_NAMES)
  region_fail = [0] * len(_REGION_NAMES)

  with torch.inference_mode():
    for batch in range(cfg.batches):
      obs = env.reset()
      if isinstance(obs, tuple):
        obs = obs[0]
      _reset_policy(policy)

      true_region = getattr(env.unwrapped, "_gk_region").clone().long()
      start_pos = ball.data.root_link_pos_w.clone()
      start_vel = ball.data.root_link_lin_vel_w.clone()
      entered = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)
      enter_step = torch.full((cfg.num_envs,), -1, dtype=torch.long, device=device)
      min_goal_x = torch.full((cfg.num_envs,), 99.0, dtype=torch.float32, device=device)
      final_pos = start_pos.clone()
      final_vel = start_vel.clone()

      for step in range(cfg.steps):
        obs = env.step(policy(obs))[0]
        pos = ball.data.root_link_pos_w
        vel = ball.data.root_link_lin_vel_w
        local_pos = pos - origins
        in_goal = (
          (local_pos[:, 0] <= _GOAL_X)
          & (local_pos[:, 1].abs() <= _GOAL_HALF_WIDTH)
          & (pos[:, 2] <= _GOAL_HEIGHT)
        )
        new_enter = in_goal & ~entered
        enter_step = torch.where(new_enter, torch.full_like(enter_step, step), enter_step)
        entered |= in_goal
        min_goal_x = torch.minimum(min_goal_x, local_pos[:, 0])
        final_pos = pos.clone()
        final_vel = vel.clone()

      start_local = start_pos - origins
      final_local = final_pos - origins
      goal_t, goal_y, goal_z = _crossing_features(start_local, start_vel)
      for idx in range(cfg.num_envs):
        region = int(true_region[idx])
        fail = bool(entered[idx])
        region_total[region] += 1
        region_fail[region] += int(fail)
        if not fail:
          continue
        rows.append(
          {
            "batch": batch,
            "env": idx,
            "true_region": region,
            "true_region_name": _REGION_NAMES[region],
            "enter_step": int(enter_step[idx]),
            "min_ball_x": float(min_goal_x[idx]),
            "start_x": float(start_local[idx, 0]),
            "start_y": float(start_local[idx, 1]),
            "start_z": float(start_local[idx, 2]),
            "vel_x": float(start_vel[idx, 0]),
            "vel_y": float(start_vel[idx, 1]),
            "vel_z": float(start_vel[idx, 2]),
            "goal_t": float(goal_t[idx]),
            "goal_y": float(goal_y[idx]),
            "goal_z": float(goal_z[idx]),
            "final_x": float(final_local[idx, 0]),
            "final_y": float(final_local[idx, 1]),
            "final_z": float(final_local[idx, 2]),
            "final_vx": float(final_vel[idx, 0]),
            "final_vy": float(final_vel[idx, 1]),
            "final_vz": float(final_vel[idx, 2]),
          }
        )

      total_done = (batch + 1) * cfg.num_envs
      print(
        f"batch {batch + 1}/{cfg.batches}: failures={len(rows)} total={total_done}",
        flush=True,
      )
      if cfg.max_failures > 0 and len(rows) >= cfg.max_failures:
        rows = rows[: cfg.max_failures]
        print(f"[INFO] reached max_failures={cfg.max_failures}", flush=True)
        break

  out = Path(cfg.out_csv)
  out.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else _default_fieldnames()
  with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  total = sum(region_total)
  fails = sum(region_fail)
  print(f"\nFailures: {fails}/{total} = {100.0 * fails / max(1, total):.2f}%")
  for region, name in enumerate(_REGION_NAMES):
    n = region_total[region]
    f = region_fail[region]
    print(f"{name:<11}: fail {f}/{n} = {100.0 * f / max(1, n):.2f}%")
  print(f"\n[INFO] wrote {len(rows)} failures to {out}")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="collect_goalkeeper_failures"))
