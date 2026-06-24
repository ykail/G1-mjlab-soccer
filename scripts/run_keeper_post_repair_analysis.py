"""Fully automated post-repair analysis for keeper residual checkpoints.

Pipeline:
1. read a completed big-repair ``summary.json`` and select top candidates;
2. run paired base-vs-candidate replay on identical ball trajectories;
3. generate region-masked variants of the best checkpoint;
4. evaluate all masked variants plus the base;
5. write a final recommendation JSON.

All child Python processes are launched with ``--python-bin``.  On the server,
use ``/data/mjlab-cu126/bin/python`` so subprocesses run in the MuJoCo env even
if the current shell points elsewhere.
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
  source_root: str = "logs/keeper_big_repair_fixed_long"
  out_root: str = "logs/keeper_post_repair_auto"
  base: str = "checkpoints/keeper_93_moe6.pt"
  python_bin: str = "/data/mjlab-cu126/bin/python"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  top_k: int = 2
  pairwise_trials_per_seed: int = 50
  mask_trials_per_seed: int = 50
  seeds: tuple[int, ...] = (42, 2810, 202686)
  regions: tuple[int, ...] = (1, 2, 3, 5)
  force: bool = True


def _env() -> dict[str, str]:
  env = os.environ.copy()
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  return env


def _quote(cmd: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in cmd)


def _run(label: str, cmd: list[str], log_path: Path, check: bool = True) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] {_quote(cmd)}", flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  with log_path.open("a") as log:
    log.write(f"\n===== {label} =====\n")
    log.write("[CMD] " + _quote(cmd) + "\n")
    log.flush()
    proc = subprocess.run(
      cmd,
      cwd=_REPO_ROOT,
      env=_env(),
      stdout=log,
      stderr=subprocess.STDOUT,
      check=False,
    )
    log.write(f"[RETURN] {proc.returncode}\n")
  if check and proc.returncode != 0:
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-100:])
    print(tail, flush=True)
    raise subprocess.CalledProcessError(proc.returncode, cmd)
  return proc.returncode


def _load_json(path: Path):
  return json.loads(path.read_text())


def _top_candidates(source_root: Path, top_k: int) -> list[dict]:
  summary = source_root / "summary.json"
  if not summary.exists():
    alt = source_root / "eval_summary.json"
    if alt.exists():
      summary = alt
  rows = _load_json(summary)
  good = [
    row for row in rows
    if row.get("score", -1.0) >= 0.0 and Path(row["checkpoint"]).exists()
  ]
  if not good:
    raise RuntimeError(f"no evaluated checkpoints found in {summary}")
  good.sort(key=lambda row: float(row.get("score", -1.0)), reverse=True)
  return good[:top_k]


def _write_final(cfg: Cfg, out_root: Path, top: list[dict]) -> Path:
  pairwise_path = out_root / "pairwise_top.json"
  mask_eval_path = out_root / "mask_eval" / "eval_summary.json"
  pairwise = _load_json(pairwise_path) if pairwise_path.exists() else {}
  mask_rows = _load_json(mask_eval_path) if mask_eval_path.exists() else []
  mask_rows.sort(key=lambda row: float(row.get("score", -1.0)), reverse=True)

  base_score = None
  for row in mask_rows:
    if row.get("name") == "base":
      base_score = row.get("score")
      break

  best_mask = mask_rows[0] if mask_rows else None
  recommendation = {
    "source_root": cfg.source_root,
    "base": cfg.base,
    "base_score_in_mask_eval": base_score,
    "top_input_candidates": top,
    "pairwise": pairwise,
    "mask_eval_summary": str(mask_eval_path),
    "best_mask_or_base": best_mask,
  }
  if best_mask and base_score is not None:
    recommendation["best_minus_base"] = float(best_mask.get("score", -1.0)) - float(base_score)

  out = out_root / "final_recommendation.json"
  out.write_text(json.dumps(recommendation, indent=2) + "\n")
  return out


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  log_path = out_root / "auto_pipeline.log"
  source_root = Path(cfg.source_root)

  top = _top_candidates(source_root, cfg.top_k)
  candidates = [row["checkpoint"] for row in top]
  names = [row["name"] for row in top]
  print("[INFO] selected candidates:", flush=True)
  for row in top:
    print(f"  {row['name']}: {100.0 * float(row['score']):.2f}% {row['checkpoint']}", flush=True)

  pairwise_cmd = [
    cfg.python_bin,
    "scripts/eval_keeper_pairwise.py",
    "--base",
    cfg.base,
    "--candidates",
    *candidates,
    "--names",
    *names,
    "--seeds",
    *[str(seed) for seed in cfg.seeds],
    "--trials-per-seed",
    str(cfg.pairwise_trials_per_seed),
    "--device",
    "cuda:0",
    "--out-json",
    str(out_root / "pairwise_top.json"),
    "--out-csv",
    str(out_root / "pairwise_top.csv"),
  ]
  _run("PAIRWISE", pairwise_cmd, log_path)

  mask_dir = out_root / "region_masks"
  mask_cmd = [
    cfg.python_bin,
    "scripts/mask_moe6_residual_regions.py",
    "--checkpoint",
    candidates[0],
    "--out-dir",
    str(mask_dir),
    "--regions",
    *[str(region) for region in cfg.regions],
  ]
  _run("MASK", mask_cmd, log_path)

  manifest = mask_dir / "manifest.txt"
  mask_checkpoints = [line.strip() for line in manifest.read_text().splitlines() if line.strip()]
  if not mask_checkpoints:
    raise RuntimeError(f"no mask checkpoints written to {manifest}")

  eval_cmd = [
    cfg.python_bin,
    "scripts/eval_keeper_big_repair.py",
    "--out-root",
    str(out_root / "mask_eval"),
    "--checkpoints",
    *mask_checkpoints,
    "--include-base",
    cfg.base,
    "--devices",
    *[str(device) for device in cfg.devices],
    "--seeds",
    *[str(seed) for seed in cfg.seeds],
    "--trials-per-seed",
    str(cfg.mask_trials_per_seed),
  ]
  if cfg.force:
    eval_cmd.append("--force")
  _run("MASK_EVAL", eval_cmd, log_path)

  final = _write_final(cfg, out_root, top)
  print(f"\n[INFO] wrote {final}", flush=True)
  data = _load_json(final)
  best = data.get("best_mask_or_base") or {}
  if best:
    print(
      f"[BEST] {best.get('name')} score={100.0 * float(best.get('score', -1.0)):.2f}% "
      f"checkpoint={best.get('checkpoint')}",
      flush=True,
    )
  if "best_minus_base" in data:
    print(f"[DELTA] best-minus-base={100.0 * float(data['best_minus_base']):+.2f}%", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_post_repair_analysis"))
