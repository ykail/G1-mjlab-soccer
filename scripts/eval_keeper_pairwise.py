"""Paired goalkeeper evaluation on identical ball trajectories.

The normal official evaluator gives an aggregate score per checkpoint. This
script answers the more useful repair question: on the same sampled and
forced-replayed trials, which balls did a candidate fix, and which previously
saved balls did it lose? It samples each trial once, then replays the exact ball
state for the base and every candidate using the env's forced-reset path.

The aggregate rate can differ slightly from the official evaluator because the
forced replay path resets the simulator again for each policy. Use this script
for relative fixed/regressed diagnostics; use eval_goalkeeper_official_seeds.py
for the final score.
"""

from __future__ import annotations

import csv
import json
import random
import sys
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

from scripts.eval_naive_goalkeeper import _load_policy, run_trial


_REGION_NAMES = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]


@dataclass
class Cfg:
  base: str = "checkpoints/keeper_93_moe6.pt"
  candidates: tuple[str, ...] = ()
  names: tuple[str, ...] = ()
  seeds: tuple[int, ...] = (42, 2810, 202686)
  trials_per_seed: int = 50
  max_steps: int = 150
  task_id: str = "Eval-Goalkeeper"
  device: str = "cuda:0"
  out_json: str = "logs/keeper_pairwise/pairwise.json"
  out_csv: str = "logs/keeper_pairwise/pairwise_trials.csv"


def _set_seed(seed: int) -> None:
  random.seed(seed)
  if np is not None:
    np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def _make_env(cfg: Cfg):
  env_cfg = load_env_cfg(cfg.task_id, play=False)
  env_cfg.scene.num_envs = 1
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  return RslRlVecEnvWrapper(env, clip_actions=100.0)


def _candidate_names(cfg: Cfg) -> list[str]:
  if cfg.names:
    if len(cfg.names) != len(cfg.candidates):
      raise ValueError("--names length must match --candidates length")
    return list(cfg.names)
  names = []
  for ckpt in cfg.candidates:
    stem = Path(ckpt).stem
    if stem.startswith("moe6_residual_"):
      stem = stem[len("moe6_residual_") :]
    names.append(stem)
  return names


def _sample_trial(env, cfg: Cfg) -> dict:
  if hasattr(env.unwrapped, "_gk_forced"):
    delattr(env.unwrapped, "_gk_forced")
  obs = env.reset()
  del obs
  ball = env.unwrapped.scene["ball"]
  origins = env.unwrapped.scene.env_origins
  start_w = ball.data.root_link_pos_w[0].clone()
  vel = ball.data.root_link_lin_vel_w[0].clone()
  region = int(getattr(env.unwrapped, "_gk_region")[0].item())
  start_local = start_w - origins[0]
  return {
    "start_w": start_w,
    "vel": vel,
    "region": region,
    "start_local": start_local,
  }


def _force_trial(env, scenario: dict) -> None:
  dev = scenario["start_w"].device
  env.unwrapped._gk_forced = {
    "start": scenario["start_w"].reshape(1, 3),
    "vel": scenario["vel"].reshape(1, 3),
    "region": torch.tensor([scenario["region"]], dtype=torch.long, device=dev),
  }


def _run_one(env, policy, scenario: dict, cfg: Cfg) -> bool:
  reset = getattr(policy, "reset", None)
  if reset is not None:
    reset()
  _force_trial(env, scenario)
  stats = run_trial(env, policy, max_steps=cfg.max_steps)
  return not bool(stats["ball_entered_goal"])


def _empty_counts() -> dict:
  return {
    "base_blocked": 0,
    "candidate_blocked": 0,
    "both_blocked": 0,
    "both_failed": 0,
    "fixed_base_failure": 0,
    "regressed_base_success": 0,
    "trials": 0,
  }


def main(cfg: Cfg) -> None:
  if not cfg.candidates:
    raise ValueError("--candidates is required")
  configure_torch_backends()
  env = _make_env(cfg)
  base_policy = _load_policy(cfg.base, env, cfg.device)
  names = _candidate_names(cfg)
  candidate_policies = [
    _load_policy(checkpoint, env, cfg.device) for checkpoint in cfg.candidates
  ]

  rows = []
  totals = {name: _empty_counts() for name in names}
  by_region = {
    name: {region: _empty_counts() for region in range(len(_REGION_NAMES))}
    for name in names
  }
  by_seed = {
    name: {seed: _empty_counts() for seed in cfg.seeds}
    for name in names
  }

  try:
    for seed in cfg.seeds:
      _set_seed(seed)
      env.unwrapped.cfg.seed = seed
      for trial in range(cfg.trials_per_seed):
        scenario = _sample_trial(env, cfg)
        base_blocked = _run_one(env, base_policy, scenario, cfg)
        row_base = {
          "seed": seed,
          "trial": trial,
          "region": scenario["region"],
          "region_name": _REGION_NAMES[scenario["region"]],
          "start_x": float(scenario["start_local"][0]),
          "start_y": float(scenario["start_local"][1]),
          "start_z": float(scenario["start_local"][2]),
          "vel_x": float(scenario["vel"][0]),
          "vel_y": float(scenario["vel"][1]),
          "vel_z": float(scenario["vel"][2]),
          "base_blocked": int(base_blocked),
        }
        for name, policy in zip(names, candidate_policies):
          cand_blocked = _run_one(env, policy, scenario, cfg)
          fixed = (not base_blocked) and cand_blocked
          regressed = base_blocked and (not cand_blocked)
          row = {
            **row_base,
            "candidate": name,
            "candidate_blocked": int(cand_blocked),
            "fixed_base_failure": int(fixed),
            "regressed_base_success": int(regressed),
          }
          rows.append(row)
          for bucket in (totals[name], by_region[name][scenario["region"]], by_seed[name][seed]):
            bucket["trials"] += 1
            bucket["base_blocked"] += int(base_blocked)
            bucket["candidate_blocked"] += int(cand_blocked)
            bucket["both_blocked"] += int(base_blocked and cand_blocked)
            bucket["both_failed"] += int((not base_blocked) and (not cand_blocked))
            bucket["fixed_base_failure"] += int(fixed)
            bucket["regressed_base_success"] += int(regressed)
        if (trial + 1) % max(1, cfg.trials_per_seed // 5) == 0:
          print(f"[PAIR] seed {seed} trial {trial + 1}/{cfg.trials_per_seed}", flush=True)
  finally:
    if hasattr(env.unwrapped, "_gk_forced"):
      delattr(env.unwrapped, "_gk_forced")
    env.close()

  for name, stats in totals.items():
    stats["base_rate"] = stats["base_blocked"] / max(1, stats["trials"])
    stats["candidate_rate"] = stats["candidate_blocked"] / max(1, stats["trials"])
    stats["net_gain"] = stats["candidate_blocked"] - stats["base_blocked"]

  summary = {
    "base": cfg.base,
    "candidates": dict(zip(names, cfg.candidates)),
    "seeds": list(cfg.seeds),
    "trials_per_seed": cfg.trials_per_seed,
    "totals": totals,
    "by_seed": by_seed,
    "by_region": by_region,
  }

  out_json = Path(cfg.out_json)
  out_json.parent.mkdir(parents=True, exist_ok=True)
  out_json.write_text(json.dumps(summary, indent=2) + "\n")

  out_csv = Path(cfg.out_csv)
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  fieldnames = list(rows[0].keys()) if rows else []
  with out_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

  print(f"[INFO] wrote {out_json}", flush=True)
  print(f"[INFO] wrote {out_csv}", flush=True)
  for name, stats in totals.items():
    print(
      f"[{name}] base={stats['base_blocked']}/{stats['trials']} "
      f"candidate={stats['candidate_blocked']}/{stats['trials']} "
      f"fixed={stats['fixed_base_failure']} regressed={stats['regressed_base_success']} "
      f"net={stats['net_gain']:+d}",
      flush=True,
    )


if __name__ == "__main__":
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  import src.tasks.soccer.config.eval  # noqa: F401

  main(tyro.cli(Cfg, prog="eval_keeper_pairwise"))
