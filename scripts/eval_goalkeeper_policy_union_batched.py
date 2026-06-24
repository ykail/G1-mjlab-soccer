"""Batched paired evaluation and oracle-union analysis for keeper policies.

This script samples each goalkeeper trajectory once, then replays the exact
same batched scenarios for a base checkpoint and any number of candidates.  It
answers the question aggregate eval cannot answer:

* do candidates fix different balls than the base?
* how many base successes do they regress?
* what is the upper bound if an oracle could choose any policy per scenario?

If the oracle-union rate is below the target, no router/gate over these
checkpoints can reach that target.  If it is high, the next useful step is a
small trajectory-feature gate rather than more blind checkpoint sweeps.
"""

from __future__ import annotations

import csv
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
  base: str = "checkpoints/keeper_93_moe6.pt"
  candidates: tuple[str, ...] = ()
  names: tuple[str, ...] = ()
  seeds: tuple[int, ...] = (42, 2810, 202686)
  trials_per_seed: int = 200
  batch_size: int = 256
  max_steps: int = 150
  task_id: str = "Eval-Goalkeeper"
  device: str | None = None
  out_json: str = "logs/keeper_union_batched/union.json"
  out_csv: str = "logs/keeper_union_batched/trials.csv"
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


def _make_env(cfg: Cfg, seed: int, device: str, num_envs: int):
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.seed = seed
  env_cfg.scene.num_envs = num_envs
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  return RslRlVecEnvWrapper(env, clip_actions=100.0)


def _candidate_names(cfg: Cfg) -> list[str]:
  if cfg.names:
    if len(cfg.names) != len(cfg.candidates):
      raise ValueError("--names length must match --candidates length")
    return list(cfg.names)
  names = []
  for ckpt in cfg.candidates:
    stem = Path(ckpt).stem
    for prefix in ("moe6_residual_", "targeted_moe6_residual_"):
      if stem.startswith(prefix):
        stem = stem[len(prefix) :]
    names.append(stem.replace(".", "p"))
  return names


def _reset_policy(policy, dones=None) -> None:
  reset = getattr(policy, "reset", None)
  if reset is None:
    return
  try:
    reset(dones)
  except TypeError:
    reset()


def _sample_scenarios(env) -> dict[str, torch.Tensor]:
  base_env = env.unwrapped
  if hasattr(base_env, "_gk_forced"):
    delattr(base_env, "_gk_forced")
  reset = env.reset()
  del reset
  ball = base_env.scene["ball"]
  origins = base_env.scene.env_origins
  region = getattr(base_env, "_gk_region", None)
  if region is None or region.shape[0] != base_env.num_envs:
    region = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=origins.device)
  return {
    "start_w": ball.data.root_link_pos_w.detach().clone(),
    "start_local": (ball.data.root_link_pos_w - origins).detach().clone(),
    "vel": ball.data.root_link_lin_vel_w.detach().clone(),
    "region": region.detach().long().clone(),
  }


def _force_scenarios(env, scenarios: dict[str, torch.Tensor]) -> None:
  env.unwrapped._gk_forced = {
    "start": scenarios["start_w"],
    "vel": scenarios["vel"],
    "region": scenarios["region"],
  }


def _entered_goal(ball_pos_w: torch.Tensor, origins: torch.Tensor, active_n: int) -> torch.Tensor:
  local = ball_pos_w[:active_n] - origins[:active_n]
  return (
    (local[:, 0] <= _GOAL_X)
    & (local[:, 1].abs() <= _GOAL_HALF_WIDTH)
    & (local[:, 2] <= _GOAL_HEIGHT)
  )


def _run_policy_batch(env, policy, scenarios: dict[str, torch.Tensor], active_n: int, max_steps: int) -> torch.Tensor:
  _force_scenarios(env, scenarios)
  _reset_policy(policy)
  obs = env.reset()
  if isinstance(obs, tuple):
    obs = obs[0]

  base_env = env.unwrapped
  ball = base_env.scene["ball"]
  origins = base_env.scene.env_origins
  entered = torch.zeros(active_n, dtype=torch.bool, device=origins.device)

  for _ in range(max_steps):
    with torch.inference_mode():
      action = policy(obs)
    result = env.step(action)
    obs = result[0]
    dones = result[2]
    entered |= _entered_goal(ball.data.root_link_pos_w, origins, active_n)
    if torch.is_tensor(dones) and bool(dones[:active_n].all().item()):
      break
  return ~entered


def _empty_policy_stats() -> dict:
  return {
    "blocked": 0,
    "trials": 0,
    "rate": 0.0,
    "fixed_base_failure": 0,
    "regressed_base_success": 0,
    "net_vs_base": 0,
    "by_region": {
      name: {"blocked": 0, "trials": 0, "rate": 0.0}
      for name in _REGION_NAMES
    },
  }


def _empty_oracle_stats() -> dict:
  return {
    "blocked": 0,
    "trials": 0,
    "rate": 0.0,
    "extra_vs_base": 0,
    "missed_by_all": 0,
    "by_region": {
      name: {"blocked": 0, "trials": 0, "rate": 0.0}
      for name in _REGION_NAMES
    },
  }


def _add_region_counts(stats: dict, regions: torch.Tensor, blocked: torch.Tensor) -> None:
  for idx, name in enumerate(_REGION_NAMES):
    mask = regions == idx
    trials = int(mask.sum().item())
    if trials <= 0:
      continue
    stats["by_region"][name]["trials"] += trials
    stats["by_region"][name]["blocked"] += int(blocked[mask].sum().item())


def _finalize_rates(stats: dict) -> None:
  stats["rate"] = int(stats["blocked"]) / max(1, int(stats["trials"]))
  for row in stats.get("by_region", {}).values():
    row["rate"] = int(row["blocked"]) / max(1, int(row["trials"]))


def _eval_seed(cfg: Cfg, seed: int, device: str) -> tuple[dict, list[dict]]:
  _set_seed(seed)
  batch_size = max(1, min(int(cfg.batch_size), int(cfg.trials_per_seed)))
  env = _make_env(cfg, seed, device, batch_size)
  names = ["base"] + _candidate_names(cfg)
  checkpoints = [cfg.base] + list(cfg.candidates)
  policies = []
  for name, checkpoint in zip(names, checkpoints):
    print(f"[LOAD] {name}: {checkpoint}", flush=True)
    policies.append(_load_policy(checkpoint, env, device))

  policy_stats = {name: _empty_policy_stats() for name in names}
  oracle_stats = _empty_oracle_stats()
  rows: list[dict] = []
  done_trials = 0
  batch_idx = 0

  try:
    while done_trials < cfg.trials_per_seed:
      active_n = min(batch_size, cfg.trials_per_seed - done_trials)
      scenarios = _sample_scenarios(env)
      regions = scenarios["region"][:active_n].detach().cpu()
      start = scenarios["start_local"][:active_n].detach().cpu()
      vel = scenarios["vel"][:active_n].detach().cpu()
      outcomes = {}

      for name, policy in zip(names, policies):
        blocked = _run_policy_batch(env, policy, scenarios, active_n, cfg.max_steps)
        blocked_cpu = blocked.detach().cpu()
        outcomes[name] = blocked_cpu
        stats = policy_stats[name]
        stats["blocked"] += int(blocked_cpu.sum().item())
        stats["trials"] += active_n
        _add_region_counts(stats, regions, blocked_cpu)

      base_blocked = outcomes["base"]
      stacked = torch.stack([outcomes[name] for name in names], dim=0)
      oracle_blocked = stacked.any(dim=0)
      oracle_stats["blocked"] += int(oracle_blocked.sum().item())
      oracle_stats["trials"] += active_n
      oracle_stats["extra_vs_base"] += int((oracle_blocked & ~base_blocked).sum().item())
      oracle_stats["missed_by_all"] += int((~oracle_blocked).sum().item())
      _add_region_counts(oracle_stats, regions, oracle_blocked)

      for name in names:
        if name == "base":
          continue
        stats = policy_stats[name]
        cand = outcomes[name]
        stats["fixed_base_failure"] += int((~base_blocked & cand).sum().item())
        stats["regressed_base_success"] += int((base_blocked & ~cand).sum().item())

      for i in range(active_n):
        row = {
          "seed": seed,
          "trial": done_trials + i,
          "batch": batch_idx,
          "env": i,
          "region": int(regions[i].item()),
          "region_name": _REGION_NAMES[int(regions[i].item())] if 0 <= int(regions[i].item()) < len(_REGION_NAMES) else "unknown",
          "start_x": float(start[i, 0].item()),
          "start_y": float(start[i, 1].item()),
          "start_z": float(start[i, 2].item()),
          "vel_x": float(vel[i, 0].item()),
          "vel_y": float(vel[i, 1].item()),
          "vel_z": float(vel[i, 2].item()),
          "oracle_any_blocked": int(oracle_blocked[i].item()),
        }
        for name in names:
          row[f"{name}_blocked"] = int(outcomes[name][i].item())
        rows.append(row)

      print(
        f"  seed {seed} batch {batch_idx + 1:3d}: "
        f"base={int(base_blocked.sum().item())}/{active_n} "
        f"oracle={int(oracle_blocked.sum().item())}/{active_n} "
        f"done={done_trials + active_n}/{cfg.trials_per_seed}",
        flush=True,
      )
      done_trials += active_n
      batch_idx += 1
  finally:
    if hasattr(env.unwrapped, "_gk_forced"):
      delattr(env.unwrapped, "_gk_forced")
    env.close()

  for name, stats in policy_stats.items():
    if name != "base":
      stats["net_vs_base"] = int(stats["fixed_base_failure"]) - int(stats["regressed_base_success"])
    _finalize_rates(stats)
  _finalize_rates(oracle_stats)

  result = {
    "seed": seed,
    "base": cfg.base,
    "candidates": dict(zip(names[1:], cfg.candidates)),
    "names": names,
    "trials": cfg.trials_per_seed,
    "batch_size": batch_size,
    "policy_stats": policy_stats,
    "oracle_any": oracle_stats,
  }
  return result, rows


def _merge_results(results: list[dict], cfg: Cfg) -> dict:
  if not results:
    raise RuntimeError("no seed results")
  names = results[0]["names"]
  merged_policy = {name: _empty_policy_stats() for name in names}
  merged_oracle = _empty_oracle_stats()

  for result in results:
    for name in names:
      src = result["policy_stats"][name]
      dst = merged_policy[name]
      for key in ("blocked", "trials", "fixed_base_failure", "regressed_base_success", "net_vs_base"):
        dst[key] += int(src.get(key, 0))
      for region_name in _REGION_NAMES:
        dst_r = dst["by_region"][region_name]
        src_r = src["by_region"][region_name]
        dst_r["blocked"] += int(src_r.get("blocked", 0))
        dst_r["trials"] += int(src_r.get("trials", 0))
    src_o = result["oracle_any"]
    for key in ("blocked", "trials", "extra_vs_base", "missed_by_all"):
      merged_oracle[key] += int(src_o.get(key, 0))
    for region_name in _REGION_NAMES:
      dst_r = merged_oracle["by_region"][region_name]
      src_r = src_o["by_region"][region_name]
      dst_r["blocked"] += int(src_r.get("blocked", 0))
      dst_r["trials"] += int(src_r.get("trials", 0))

  for stats in merged_policy.values():
    _finalize_rates(stats)
  _finalize_rates(merged_oracle)

  return {
    "base": cfg.base,
    "candidates": dict(zip(names[1:], cfg.candidates)),
    "seeds": [row["seed"] for row in results],
    "trials_per_seed": cfg.trials_per_seed,
    "batch_size": cfg.batch_size,
    "max_steps": cfg.max_steps,
    "per_seed": results,
    "policy_stats": merged_policy,
    "oracle_any": merged_oracle,
  }


def _write_csv(path: Path, rows: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    path.write_text("")
    return
  fieldnames = list(rows[0].keys())
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _run_single_seed_cli(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  if len(cfg.seeds) != 1:
    raise ValueError("--single-seed-mode requires exactly one seed")
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  result, rows = _eval_seed(cfg, cfg.seeds[0], device)

  out_json = Path(cfg.out_json)
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(result, indent=2) + "\n")
  _write_csv(Path(cfg.out_csv), rows)
  print(f"[INFO] wrote {out_json}", flush=True)
  print(f"[INFO] wrote {cfg.out_csv}", flush=True)


def _run_parallel(cfg: Cfg) -> tuple[list[dict], list[Path]]:
  out_root = Path(cfg.out_json).with_suffix("")
  seed_log_dir = Path(cfg.seed_log_dir) if cfg.seed_log_dir else out_root.parent / f"{out_root.name}_seed_logs"
  seed_log_dir.mkdir(parents=True, exist_ok=True)
  procs = []
  for idx, seed in enumerate(cfg.seeds):
    seed_json = seed_log_dir / f"seed_{seed}.json"
    seed_csv = seed_log_dir / f"seed_{seed}.csv"
    seed_log = seed_log_dir / f"seed_{seed}.log"
    cmd = [
      sys.executable,
      str(Path(__file__).resolve()),
      "--base",
      cfg.base,
      "--seeds",
      str(seed),
      "--trials-per-seed",
      str(cfg.trials_per_seed),
      "--batch-size",
      str(cfg.batch_size),
      "--max-steps",
      str(cfg.max_steps),
      "--task-id",
      cfg.task_id,
      "--out-json",
      str(seed_json),
      "--out-csv",
      str(seed_csv),
      "--single-seed-mode",
    ]
    if cfg.candidates:
      cmd.extend(["--candidates", *cfg.candidates])
    if cfg.names:
      cmd.extend(["--names", *cfg.names])
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
    log = seed_log.open("w")
    log.write("[CMD] " + " ".join(cmd) + "\n")
    log.flush()
    proc = subprocess.Popen(cmd, env=env, cwd=_REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    procs.append((seed, seed_json, seed_csv, seed_log, proc, log))

  results = []
  csv_paths = []
  failed = []
  for seed, seed_json, seed_csv, seed_log, proc, log in procs:
    code = proc.wait()
    log.close()
    print(f"\n===== seed {seed} log tail =====")
    if seed_log.exists():
      print("\n".join(seed_log.read_text(errors="replace").splitlines()[-60:]))
    if code != 0:
      failed.append((seed, code, seed_log))
      continue
    results.append(json.loads(seed_json.read_text()))
    csv_paths.append(seed_csv)
  if failed:
    raise RuntimeError(f"seed union eval failed: {failed}")
  results.sort(key=lambda row: cfg.seeds.index(row["seed"]))
  csv_paths.sort(key=lambda path: cfg.seeds.index(int(path.stem.split("_")[1])))
  return results, csv_paths


def _merge_csvs(paths: list[Path], out_csv: Path) -> None:
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  wrote_header = False
  with out_csv.open("w", newline="") as fout:
    writer = csv.writer(fout)
    for path in paths:
      with path.open(newline="") as fin:
        reader = csv.reader(fin)
        try:
          header = next(reader)
        except StopIteration:
          continue
        if not wrote_header:
          writer.writerow(header)
          wrote_header = True
        writer.writerows(reader)


def _run(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  if cfg.single_seed_mode:
    _run_single_seed_cli(cfg)
    return
  if not cfg.candidates:
    raise ValueError("--candidates is required")

  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  print(
    f"[INFO] base={cfg.base} candidates={len(cfg.candidates)} "
    f"seeds={cfg.seeds} trials_per_seed={cfg.trials_per_seed} "
    f"batch_size={cfg.batch_size}",
    flush=True,
  )

  if cfg.parallel_seeds:
    results, csv_paths = _run_parallel(cfg)
    _merge_csvs(csv_paths, Path(cfg.out_csv))
  else:
    results = []
    rows = []
    for seed in cfg.seeds:
      result, seed_rows = _eval_seed(cfg, seed, device)
      results.append(result)
      rows.extend(seed_rows)
    _write_csv(Path(cfg.out_csv), rows)

  summary = _merge_results(results, cfg)
  out_json = Path(cfg.out_json)
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(summary, indent=2) + "\n")
  print(f"\n[INFO] wrote {out_json}", flush=True)
  print(f"[INFO] wrote {cfg.out_csv}", flush=True)
  for name, stats in summary["policy_stats"].items():
    print(
      f"[POLICY] {name}: {stats['blocked']}/{stats['trials']} "
      f"= {100.0 * stats['rate']:.2f}% "
      f"fixed={stats['fixed_base_failure']} "
      f"regressed={stats['regressed_base_success']} "
      f"net={stats['net_vs_base']:+d}",
      flush=True,
    )
  oracle = summary["oracle_any"]
  print(
    f"[ORACLE_ANY] {oracle['blocked']}/{oracle['trials']} "
    f"= {100.0 * oracle['rate']:.2f}% "
    f"extra_vs_base={oracle['extra_vs_base']} missed_by_all={oracle['missed_by_all']}",
    flush=True,
  )


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
  main(tyro.cli(Cfg, prog="eval_goalkeeper_policy_union_batched"))
