"""Fast batched goalkeeper evaluation for the fixed official seed protocol.

The original fixed-seed evaluator runs one environment per process and repeats
``env.reset()`` for every trial.  This version evaluates many copied
environments at once on a single GPU, which is much faster on large-memory
cards.  The sampled trajectories are distribution-equivalent to the official
parabolic goalkeeper evaluation, but the exact random sequence is batched, so
compare checkpoints with the same ``--batch-size`` and use the serial evaluator
only for final spot checks.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

try:
  import numpy as np
except ModuleNotFoundError:  # pragma: no cover
  np = None

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

from scripts.eval_naive_goalkeeper import _load_policy


_GOAL_X = -0.5
_GOAL_HALF_WIDTH = 1.5
_GOAL_HEIGHT = 1.8
_REGION_NAMES = (
  "Right-Mid",
  "Left-Mid",
  "Right-Up",
  "Left-Up",
  "Right-Low",
  "Left-Low",
)


@dataclass
class Cfg:
  checkpoint: str | None = None
  seeds: tuple[int, ...] = (42, 2810, 202686)
  trials_per_seed: int = 200
  batch_size: int = 256
  max_steps: int = 150
  score_points: float = 30.0
  score_threshold: float = 0.8
  task_id: str = "Eval-Goalkeeper"
  device: str | None = None
  out: str = ""
  log: str = ""
  parallel_seeds: bool = False
  seed_gpus: tuple[int, ...] = ()
  seed_log_dir: str = ""
  single_seed_mode: bool = False


class _Tee:
  def __init__(self, *streams):
    self._streams = streams

  def write(self, data: str) -> int:
    for stream in self._streams:
      stream.write(data)
    return len(data)

  def flush(self) -> None:
    for stream in self._streams:
      stream.flush()

  def isatty(self) -> bool:
    return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

  def fileno(self) -> int:
    return self._streams[0].fileno()


class _ZeroPolicy:
  def __init__(self, num_envs: int, act_dim: int, device: str):
    self.num_envs = int(num_envs)
    self.act_dim = int(act_dim)
    self.device = device

  def __call__(self, obs):
    del obs
    return torch.zeros(self.num_envs, self.act_dim, device=self.device)

  def reset(self, dones=None):
    del dones


def _set_seed(seed: int) -> None:
  random.seed(seed)
  if np is not None:
    np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _make_env(cfg: Cfg, seed: int, device: str, num_envs: int):
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=100.0)


def _reset_policy(policy, dones=None) -> None:
  reset = getattr(policy, "reset", None)
  if reset is None:
    return
  try:
    reset(dones)
  except TypeError:
    reset()


def _entered_goal(ball_pos_w: torch.Tensor, origins: torch.Tensor, active_n: int) -> torch.Tensor:
  local = ball_pos_w[:active_n] - origins[:active_n]
  return (
    (local[:, 0] <= _GOAL_X)
    & (local[:, 1].abs() <= _GOAL_HALF_WIDTH)
    & (local[:, 2] <= _GOAL_HEIGHT)
  )


def _region_counts(regions: torch.Tensor, blocked: torch.Tensor) -> dict[str, dict[str, int | float]]:
  out = {}
  for idx, name in enumerate(_REGION_NAMES):
    mask = regions == idx
    total = int(mask.sum().item())
    if total <= 0:
      out[name] = {"blocked": 0, "trials": 0, "rate": 0.0}
      continue
    count = int(blocked[mask].sum().item())
    out[name] = {"blocked": count, "trials": total, "rate": count / total}
  return out


def _merge_region_counts(results: list[dict]) -> dict[str, dict[str, int | float]]:
  merged = {name: {"blocked": 0, "trials": 0, "rate": 0.0} for name in _REGION_NAMES}
  for result in results:
    for name, stats in result.get("by_region", {}).items():
      merged[name]["blocked"] += int(stats.get("blocked", 0))
      merged[name]["trials"] += int(stats.get("trials", 0))
  for stats in merged.values():
    trials = max(1, int(stats["trials"]))
    stats["rate"] = int(stats["blocked"]) / trials
  return merged


def _run_batch(env, policy, active_n: int, max_steps: int) -> tuple[torch.Tensor, torch.Tensor, int]:
  _reset_policy(policy)
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  base_env = env.unwrapped
  ball = base_env.scene["ball"]
  origins = base_env.scene.env_origins
  entered = torch.zeros(active_n, dtype=torch.bool, device=origins.device)
  steps = 0

  for _ in range(max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    dones = result[2]
    steps += 1
    entered |= _entered_goal(ball.data.root_link_pos_w, origins, active_n)
    if torch.is_tensor(dones) and bool(dones[:active_n].all().item()):
      break

  regions = getattr(base_env, "_gk_region", None)
  if regions is None or regions.numel() < active_n:
    regions = torch.full((active_n,), -1, dtype=torch.long, device=origins.device)
  else:
    regions = regions[:active_n].long()
  return ~entered, regions, steps


def _eval_seed(cfg: Cfg, seed: int, device: str) -> dict:
  _set_seed(seed)
  batch_size = max(1, min(int(cfg.batch_size), int(cfg.trials_per_seed)))
  env = _make_env(cfg, seed, device, batch_size)
  policy = (
    _load_policy(cfg.checkpoint, env, device)
    if cfg.checkpoint
    else _ZeroPolicy(batch_size, env.num_actions, device)
  )

  all_blocked = []
  all_regions = []
  batch_rows = []
  done_trials = 0
  batch_idx = 0
  try:
    while done_trials < cfg.trials_per_seed:
      active_n = min(batch_size, cfg.trials_per_seed - done_trials)
      blocked, regions, steps = _run_batch(env, policy, active_n, cfg.max_steps)
      blocked_cpu = blocked.detach().cpu()
      regions_cpu = regions.detach().cpu()
      all_blocked.append(blocked_cpu)
      all_regions.append(regions_cpu)
      blocked_count = int(blocked_cpu.sum().item())
      row = {
        "batch": batch_idx,
        "active_envs": active_n,
        "blocked": blocked_count,
        "trials": active_n,
        "steps": steps,
        "rate": blocked_count / max(1, active_n),
      }
      batch_rows.append(row)
      done_trials += active_n
      batch_idx += 1
      print(
        f"  seed {seed} batch {batch_idx:3d}: "
        f"blocked={blocked_count}/{active_n} "
        f"done={done_trials}/{cfg.trials_per_seed}",
        flush=True,
      )
  finally:
    env.close()

  blocked = torch.cat(all_blocked) if all_blocked else torch.empty(0, dtype=torch.bool)
  regions = torch.cat(all_regions) if all_regions else torch.empty(0, dtype=torch.long)
  count = int(blocked.sum().item())
  total = int(blocked.numel())
  rate = count / max(1, total)
  print(f"[SEED] {seed}: {count}/{total} = {100.0 * rate:.2f}%", flush=True)
  return {
    "seed": seed,
    "blocked": count,
    "trials": total,
    "rate": rate,
    "batch_size": batch_size,
    "batches": batch_rows,
    "by_region": _region_counts(regions, blocked),
  }


def _score(mean_rate: float, threshold: float, points: float) -> float:
  if threshold >= 1.0:
    raise ValueError("score_threshold must be < 1.0")
  frac = (mean_rate - threshold) / (1.0 - threshold)
  return max(0.0, min(1.0, frac)) * points


def _summarize(cfg: Cfg, results: list[dict]) -> dict:
  total_blocked = sum(int(row["blocked"]) for row in results)
  total_trials = sum(int(row["trials"]) for row in results)
  mean_rate = sum(float(row["rate"]) for row in results) / max(1, len(results))
  pooled_rate = total_blocked / max(1, total_trials)
  score = _score(mean_rate, cfg.score_threshold, cfg.score_points)

  summary = {
    "checkpoint": cfg.checkpoint,
    "seeds": [row["seed"] for row in results],
    "trials_per_seed": cfg.trials_per_seed,
    "batch_size": cfg.batch_size,
    "max_steps": cfg.max_steps,
    "per_seed": results,
    "by_region": _merge_region_counts(results),
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
  return summary


def _run_single_seed_cli(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  if len(cfg.seeds) != 1:
    raise ValueError("--single-seed-mode requires exactly one seed")
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  result = _eval_seed(cfg, cfg.seeds[0], device)
  if not cfg.out:
    raise ValueError("--out is required in --single-seed-mode")
  out = Path(cfg.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2) + "\n")
  print(f"[INFO] wrote {out}", flush=True)


def _run_parallel(cfg: Cfg) -> list[dict]:
  out_root = Path(cfg.out).with_suffix("") if cfg.out else Path("logs/lyk/official_batched_eval")
  seed_log_dir = Path(cfg.seed_log_dir) if cfg.seed_log_dir else out_root.parent / f"{out_root.name}_seed_logs"
  seed_log_dir.mkdir(parents=True, exist_ok=True)

  procs = []
  for idx, seed in enumerate(cfg.seeds):
    seed_json = seed_log_dir / f"seed_{seed}.json"
    seed_log = seed_log_dir / f"seed_{seed}.log"
    cmd = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--seeds",
      str(seed),
      "--trials-per-seed",
      str(cfg.trials_per_seed),
      "--batch-size",
      str(cfg.batch_size),
      "--max-steps",
      str(cfg.max_steps),
      "--score-points",
      str(cfg.score_points),
      "--score-threshold",
      str(cfg.score_threshold),
      "--task-id",
      cfg.task_id,
      "--out",
      str(seed_json),
      "--log",
      str(seed_log),
      "--single-seed-mode",
    ]
    if cfg.checkpoint:
      cmd.extend(["--checkpoint", cfg.checkpoint])
    if cfg.device:
      cmd.extend(["--device", cfg.device])
    env = os.environ.copy()
    if cfg.seed_gpus:
      env["CUDA_VISIBLE_DEVICES"] = str(cfg.seed_gpus[idx % len(cfg.seed_gpus)])
      env["MUJOCO_EGL_DEVICE_ID"] = "0"
      env.setdefault("PYOPENGL_PLATFORM", "egl")
      env.setdefault("MUJOCO_GL", "egl")
      cmd.extend(["--device", "cuda:0"])
    print(f"[LAUNCH] seed {seed}: log={seed_log}", flush=True)
    procs.append((seed, seed_json, seed_log, subprocess.Popen(cmd, env=env)))

  results = []
  failed = []
  for seed, seed_json, seed_log, proc in procs:
    code = proc.wait()
    print(f"\n===== seed {seed} log tail =====")
    if seed_log.exists():
      lines = seed_log.read_text(errors="replace").splitlines()
      print("\n".join(lines[-40:]))
    if code != 0:
      failed.append((seed, code, seed_log))
      continue
    results.append(json.loads(seed_json.read_text()))

  if failed:
    raise RuntimeError(f"batched seed eval failed: {failed}")
  results.sort(key=lambda row: cfg.seeds.index(row["seed"]))
  return results


def _run(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  if cfg.single_seed_mode:
    _run_single_seed_cli(cfg)
    return

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(
    f"[INFO] checkpoint={cfg.checkpoint or '<zero policy>'} "
    f"seeds={cfg.seeds} trials_per_seed={cfg.trials_per_seed} "
    f"batch_size={cfg.batch_size}",
    flush=True,
  )

  results = _run_parallel(cfg) if cfg.parallel_seeds else [
    _eval_seed(cfg, seed, device) for seed in cfg.seeds
  ]
  summary = _summarize(cfg, results)

  if cfg.out:
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[INFO] wrote {out}", flush=True)


def main(cfg: Cfg) -> None:
  if cfg.batch_size <= 0:
    raise ValueError("--batch-size must be positive")
  if not cfg.log:
    _run(cfg)
    return

  log = Path(cfg.log)
  log.parent.mkdir(parents=True, exist_ok=True)
  with log.open("w") as f:
    tee_out = _Tee(sys.stdout, f)
    tee_err = _Tee(sys.stderr, f)
    with redirect_stdout(tee_out), redirect_stderr(tee_err):
      print(f"[INFO] logging to {log}", flush=True)
      _run(cfg)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="eval_goalkeeper_official_batched"))
