"""Evaluate already-distilled big-repair checkpoints.

Use this after ``run_keeper_big_repair.py`` has produced checkpoints but
``summary.json`` contains ``score: -1`` rows.  It does not train or collect
anything; it only reruns the fixed official-seed evaluation and writes a
diagnostic ranked summary.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  out_root: str = "logs/keeper_big_repair"
  checkpoints: tuple[str, ...] = ()
  include_base: str = ""
  devices: tuple[int, ...] = (0, 1, 2, 3)
  seeds: tuple[int, ...] = (42, 2810, 202686)
  trials_per_seed: int = 50
  max_steps: int = 150
  score_points: float = 30.0
  score_threshold: float = 0.8
  force: bool = False
  out: str = ""


def _env(gpu: int | None) -> dict[str, str]:
  env = os.environ.copy()
  if gpu is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  return env


def _tail_text(path: Path, lines: int = 80) -> str:
  if not path.exists():
    return ""
  return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def _checkpoint_name(path: str) -> str:
  stem = Path(path).stem
  if stem.startswith("moe6_residual_"):
    stem = stem[len("moe6_residual_") :]
  return stem.replace(".", "p")


def _discover(cfg: Cfg, out_root: Path) -> list[tuple[str, str]]:
  items: list[tuple[str, str]] = []
  if cfg.include_base:
    items.append(("base", cfg.include_base))
  paths = list(cfg.checkpoints)
  if not paths:
    paths = sorted(str(path) for path in (out_root / "distilled").glob("*.pt"))
  for path in paths:
    items.append((_checkpoint_name(path), path))
  if not items:
    raise RuntimeError(
      f"no checkpoints found; pass --checkpoints or put .pt files under {out_root / 'distilled'}"
    )
  return items


def _load_existing(eval_json: Path) -> dict | None:
  if not eval_json.exists():
    return None
  try:
    data = json.loads(eval_json.read_text())
  except json.JSONDecodeError:
    return None
  if "mean_block_rate" not in data:
    return None
  return data


def _eval_one(cfg: Cfg, out_root: Path, name: str, checkpoint: str) -> dict:
  eval_dir = out_root / "eval"
  log_dir = out_root / "logs"
  eval_dir.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)
  eval_json = eval_dir / f"eval_{name}.json"
  eval_log = log_dir / f"eval_{name}.log"

  existing = None if cfg.force else _load_existing(eval_json)
  if existing is not None:
    return {
      "name": name,
      "checkpoint": checkpoint,
      "score": float(existing["mean_block_rate"]),
      "eval": str(eval_json),
      "eval_log": str(eval_log),
      "status": "ok_existing",
      "eval_returncode": 0,
    }

  cmd = [
    sys.executable,
    "scripts/eval_goalkeeper_official_seeds.py",
    "--checkpoint",
    checkpoint,
    "--seeds",
    *[str(seed) for seed in cfg.seeds],
    "--trials-per-seed",
    str(cfg.trials_per_seed),
    "--max-steps",
    str(cfg.max_steps),
    "--score-points",
    str(cfg.score_points),
    "--score-threshold",
    str(cfg.score_threshold),
    "--parallel-seeds",
    "--seed-gpus",
    *[str(gpu) for gpu in cfg.devices[: max(1, min(len(cfg.devices), len(cfg.seeds)))]],
    "--out",
    str(eval_json),
  ]

  gpu = cfg.devices[0] if cfg.devices else None
  print(f"\n[EVAL {name}] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
  print(f"[EVAL {name}] log -> {eval_log}", flush=True)
  with eval_log.open("w") as log:
    log.write("[CMD] " + " ".join(shlex.quote(part) for part in cmd) + "\n")
    proc = subprocess.run(
      cmd,
      cwd=_REPO_ROOT,
      env=_env(gpu),
      stdout=log,
      stderr=subprocess.STDOUT,
      check=False,
    )

  row = {
    "name": name,
    "checkpoint": checkpoint,
    "score": -1.0,
    "eval": str(eval_json),
    "eval_log": str(eval_log),
    "eval_returncode": proc.returncode,
  }
  data = _load_existing(eval_json)
  if proc.returncode == 0 and data is not None:
    row["score"] = float(data["mean_block_rate"])
    row["status"] = "ok"
    row["pooled_block_rate"] = float(data.get("pooled_block_rate", -1.0))
    row["official_score"] = float(data.get("score", -1.0))
  else:
    row["status"] = "eval_failed"
    row["log_tail"] = _tail_text(eval_log)
  return row


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  rows = [_eval_one(cfg, out_root, name, ckpt) for name, ckpt in _discover(cfg, out_root)]
  rows.sort(key=lambda row: (row["score"] >= 0.0, row["score"]), reverse=True)

  out = Path(cfg.out) if cfg.out else out_root / "eval_summary.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(rows, indent=2) + "\n")
  print(f"\n[INFO] wrote {out}", flush=True)
  for row in rows:
    if row["score"] >= 0.0:
      print(f"[OK]   {row['name']}: {100.0 * row['score']:.2f}%  {row['checkpoint']}", flush=True)
    else:
      print(f"[FAIL] {row['name']}: see {row['eval_log']}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="eval_keeper_big_repair"))
