"""Evaluate goalkeeper with the fixed multi-seed protocol.

This keeps the standalone official goalkeeper metric: a trial is successful if
the ball never crosses the goal plane inside the frame.  It differs from the
template script only by running a fixed list of seeds and averaging the block
rate before applying the phase-1 linear score formula.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

try:
  import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is present in training envs.
  np = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from scripts.eval_naive_goalkeeper import _load_policy, _make_zero_policy, run_trial


@dataclass
class Cfg:
  checkpoint: str | None = None
  seeds: tuple[int, ...] = (42, 2810, 202686)
  trials_per_seed: int = 58
  max_steps: int = 150
  score_points: float = 30.0
  score_threshold: float = 0.8
  task_id: str = "Eval-Goalkeeper"
  device: str | None = None
  out: str = ""


def _set_seed(seed: int) -> None:
  random.seed(seed)
  if np is not None:
    np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _make_env(cfg: Cfg, seed: int, device: str):
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = 1
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=100.0)


def _eval_seed(cfg: Cfg, seed: int, device: str) -> dict:
  _set_seed(seed)
  env = _make_env(cfg, seed, device)
  policy = (
    _load_policy(cfg.checkpoint, env, device)
    if cfg.checkpoint
    else _make_zero_policy(env, device)
  )

  blocked = 0
  try:
    for trial in range(cfg.trials_per_seed):
      reset = getattr(policy, "reset", None)
      if reset is not None:
        reset()
      stats = run_trial(env, policy, max_steps=cfg.max_steps)
      blocked += int(not stats["ball_entered_goal"])
      interval = 1 if cfg.trials_per_seed <= 10 else max(1, cfg.trials_per_seed // 4)
      if (trial + 1) % interval == 0 or trial == 0:
        print(
          f"  seed {seed} trial {trial + 1:3d}/{cfg.trials_per_seed}: "
          f"blocked={not stats['ball_entered_goal']} steps={stats['steps']}",
          flush=True,
        )
  finally:
    env.close()

  rate = blocked / max(1, cfg.trials_per_seed)
  print(f"[SEED] {seed}: {blocked}/{cfg.trials_per_seed} = {100.0 * rate:.2f}%", flush=True)
  return {"seed": seed, "blocked": blocked, "trials": cfg.trials_per_seed, "rate": rate}


def _score(mean_rate: float, threshold: float, points: float) -> float:
  if threshold >= 1.0:
    raise ValueError("score_threshold must be < 1.0")
  frac = (mean_rate - threshold) / (1.0 - threshold)
  return max(0.0, min(1.0, frac)) * points


def main(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(
    f"[INFO] checkpoint={cfg.checkpoint or '<zero policy>'} "
    f"seeds={cfg.seeds} trials_per_seed={cfg.trials_per_seed}",
    flush=True,
  )

  results = [_eval_seed(cfg, seed, device) for seed in cfg.seeds]
  total_blocked = sum(row["blocked"] for row in results)
  total_trials = sum(row["trials"] for row in results)
  mean_rate = sum(row["rate"] for row in results) / max(1, len(results))
  pooled_rate = total_blocked / max(1, total_trials)
  score = _score(mean_rate, cfg.score_threshold, cfg.score_points)

  summary = {
    "checkpoint": cfg.checkpoint,
    "seeds": [row["seed"] for row in results],
    "trials_per_seed": cfg.trials_per_seed,
    "per_seed": results,
    "mean_block_rate": mean_rate,
    "pooled_block_rate": pooled_rate,
    "score": score,
    "score_points": cfg.score_points,
    "score_threshold": cfg.score_threshold,
  }

  print("\n" + "=" * 60)
  print(f"Mean Block Rate:   {100.0 * mean_rate:.2f}%")
  print(f"Pooled Block Rate: {total_blocked}/{total_trials} = {100.0 * pooled_rate:.2f}%")
  print(f"Score:             {score:.2f}/{cfg.score_points:.1f}")
  print("=" * 60)

  if cfg.out:
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[INFO] wrote {out}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="eval_goalkeeper_official_seeds"))
