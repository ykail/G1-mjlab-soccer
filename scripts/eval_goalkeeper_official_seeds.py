"""Evaluate goalkeeper with the fixed multi-seed protocol.

This keeps the standalone official goalkeeper metric: a trial is successful if
the ball never crosses the goal plane inside the frame.  It differs from the
template script only by running a fixed list of seeds and averaging the block
rate before applying the phase-1 linear score formula.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
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


def _summarize(cfg: Cfg, results: list[dict]) -> dict:
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
  out_root = Path(cfg.out).with_suffix("") if cfg.out else Path("logs/lyk/official_seed_eval")
  seed_log_dir = Path(cfg.seed_log_dir) if cfg.seed_log_dir else out_root.parent / f"{out_root.name}_seed_logs"
  seed_log_dir.mkdir(parents=True, exist_ok=True)

  procs = []
  for idx, seed in enumerate(cfg.seeds):
    seed_json = seed_log_dir / f"seed_{seed}.json"
    seed_log = seed_log_dir / f"seed_{seed}.log"
    cmd = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--seeds", str(seed),
      "--trials-per-seed", str(cfg.trials_per_seed),
      "--max-steps", str(cfg.max_steps),
      "--score-points", str(cfg.score_points),
      "--score-threshold", str(cfg.score_threshold),
      "--task-id", cfg.task_id,
      "--out", str(seed_json),
      "--log", str(seed_log),
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
    raise RuntimeError(f"seed eval failed: {failed}")
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
    f"seeds={cfg.seeds} trials_per_seed={cfg.trials_per_seed}",
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
  main(tyro.cli(Cfg, prog="eval_goalkeeper_official_seeds"))
