"""Repair the scenarios missed by every current goalkeeper policy.

This is a second-stage script for the situation where normal residual repairs
plateau around 95% and policy-union analysis shows a few remaining
``oracle_any_blocked == 0`` trials.  It reads the union CSV, collects CEM
repairs only for those all-policy misses, builds strict patch-library
checkpoints, evaluates them, and runs a final union check.

This is intentionally narrow.  It is useful for measuring whether the remaining
misses are physically repairable, not for proving broad policy generalization.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]
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
  union_csv: str = "logs/keeper_h20_autorepair_leftmid_big/union_final.csv"
  out_root: str = "logs/keeper_union_miss_patch"
  python_bin: str = "/data/mjlab-cu126/bin/python"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  hours: float = 3.0

  # Existing policies used only in the final oracle-union report.
  candidates: tuple[str, ...] = (
    "logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt",
    "logs/keeper_patch_library_quick/checkpoints/patch_thr0p35_scale0p75.pt",
    "logs/keeper_h20_autorepair_leftmid_big/distilled/moe6_residual_regions3_scale0p008_bc1.pt",
    "logs/keeper_h20_autorepair_leftmid_big/distilled/moe6_residual_regions3_scale0p012_bc1.pt",
    "logs/keeper_h20_autorepair_leftmid_big/distilled/moe6_residual_regions3_scale0p004_bc1.pt",
  )
  names: tuple[str, ...] = (
    "targeted",
    "patch",
    "leftup_scale0p008",
    "leftup_scale0p012",
    "leftup_scale0p004",
  )

  # CEM repair search for all-policy misses.
  G: int = 12
  P: int = 192
  iters: int = 13
  elites: int = 16
  collect_batches_per_shard: int = 4
  scenario_pos_jitter: float = 0.008
  scenario_vel_jitter: float = 0.02

  # Strict nearest-neighbor patch library.  Lower thresholds reduce regression.
  thresholds: tuple[float, ...] = (0.08, 0.12, 0.18, 0.25)
  residual_scales: tuple[float, ...] = (0.25, 0.4, 0.6)
  pos_scale: float = 0.045
  vel_scale: float = 0.14

  eval_trials_per_seed: int = 300
  final_trials_per_seed: int = 400
  batch_size: int = 768
  seeds: tuple[int, ...] = (42, 2810, 202686)


def _env(gpu: int | None = None) -> dict[str, str]:
  env = os.environ.copy()
  if gpu is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  return env


def _quote(cmd: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in cmd)


def _run(label: str, cmd: list[str], log_path: Path, gpu: int | None = None, check: bool = True) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] {_quote(cmd)}", flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  with log_path.open("w") as log:
    log.write("[CMD] " + _quote(cmd) + "\n")
    log.flush()
    proc = subprocess.run(
      cmd,
      cwd=_REPO_ROOT,
      env=_env(gpu),
      stdout=log,
      stderr=subprocess.STDOUT,
      check=False,
    )
  if check and proc.returncode != 0:
    print("\n".join(log_path.read_text(errors="replace").splitlines()[-120:]), flush=True)
    raise subprocess.CalledProcessError(proc.returncode, cmd)
  return proc.returncode


def _popen(label: str, cmd: list[str], log_path: Path, gpu: int):
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] {_quote(cmd)}", flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  log = log_path.open("w")
  log.write("[CMD] " + _quote(cmd) + "\n")
  log.flush()
  proc = subprocess.Popen(
    cmd,
    cwd=_REPO_ROOT,
    env=_env(gpu),
    stdout=log,
    stderr=subprocess.STDOUT,
  )
  return proc, log


def _load_json(path: Path):
  return json.loads(path.read_text())


def _truthy(value: str) -> bool:
  return str(value).strip().lower() in ("1", "true", "yes", "y")


def _write_miss_csv(cfg: Cfg, out_root: Path) -> tuple[Path, int]:
  src = Path(cfg.union_csv)
  if not src.exists():
    raise FileNotFoundError(src)
  out = out_root / "union_missed_by_all.csv"
  if out.exists():
    rows = list(csv.DictReader(out.open()))
    return out, len(rows)

  with src.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if not _truthy(row.get("oracle_any_blocked", ""))]
  if not rows:
    raise RuntimeError(f"{src} contains no oracle_any_blocked=0 rows")

  counts = Counter(int(float(row["region"])) for row in rows)
  print(f"[INFO] all-policy misses={len(rows)}", flush=True)
  for region, count in sorted(counts.items()):
    name = _REGION_NAMES[region] if 0 <= region < len(_REGION_NAMES) else "unknown"
    print(f"  region {region} {name}: {count}", flush=True)

  out.parent.mkdir(parents=True, exist_ok=True)
  with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(rows)
  print(f"[INFO] wrote {out}", flush=True)
  return out, len(rows)


def _repair_cmd(cfg: Cfg, miss_csv: Path, out: Path, lib_out: Path, seed: int) -> list[str]:
  return [
    cfg.python_bin,
    "scripts/repair_oracle.py",
    "--checkpoint",
    cfg.base,
    "--mode",
    "collect",
    "--scenario-csv",
    str(miss_csv),
    "--scenario-blocked-column",
    "oracle_any_blocked",
    "--scenario-pos-jitter",
    str(cfg.scenario_pos_jitter),
    "--scenario-vel-jitter",
    str(cfg.scenario_vel_jitter),
    "--G",
    str(cfg.G),
    "--P",
    str(cfg.P),
    "--iters",
    str(cfg.iters),
    "--elites",
    str(cfg.elites),
    "--batches",
    str(cfg.collect_batches_per_shard),
    "--seed",
    str(seed),
    "--w-stable",
    "26.0",
    "--w-final-upright",
    "40.0",
    "--collect-pre-steps",
    "44",
    "--collect-post-steps",
    "10",
    "--device",
    "cuda:0",
    "--out",
    str(out),
    "--library-out",
    str(lib_out),
  ]


def _existing_libraries(out_root: Path) -> list[str]:
  return sorted(str(path) for path in (out_root / "repairs").glob("library_shard*.pt"))


def _collect(cfg: Cfg, out_root: Path, miss_csv: Path, deadline: float) -> list[str]:
  repair_dir = out_root / "repairs"
  repair_dir.mkdir(parents=True, exist_ok=True)
  libs = _existing_libraries(out_root)
  shard_idx = len(libs)
  running = []
  while time.monotonic() < deadline:
    busy = {gpu for _, gpu, _, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while free and time.monotonic() < deadline:
      gpu = free.pop(0)
      out = repair_dir / f"repair_shard{shard_idx:03d}.pt"
      lib_out = repair_dir / f"library_shard{shard_idx:03d}.pt"
      log = out_root / "logs" / f"collect_shard{shard_idx:03d}_gpu{gpu}.log"
      cmd = _repair_cmd(cfg, miss_csv, out, lib_out, 12000 + shard_idx)
      proc, handle = _popen(f"COLLECT {shard_idx}", cmd, log, gpu)
      running.append((shard_idx, gpu, lib_out, proc, handle))
      shard_idx += 1

    time.sleep(8.0)
    still = []
    for idx, gpu, lib_out, proc, handle in running:
      code = proc.poll()
      if code is None:
        still.append((idx, gpu, lib_out, proc, handle))
        continue
      handle.close()
      if code == 0 and lib_out.exists():
        print(f"[COLLECT] library {idx} done: {lib_out}", flush=True)
        libs.append(str(lib_out))
      else:
        print(f"[COLLECT] shard {idx} failed code={code}", flush=True)
    running = still

  if running:
    print("[COLLECT] deadline reached; terminating active shards", flush=True)
    for idx, _, _, proc, handle in running:
      proc.terminate()
      try:
        proc.wait(timeout=30)
      except subprocess.TimeoutExpired:
        proc.kill()
      handle.close()
      print(f"[COLLECT] stopped shard {idx}", flush=True)
  return sorted(set(libs))


def _build_patch_checkpoints(cfg: Cfg, out_root: Path, libs: list[str]) -> list[str]:
  checkpoints_dir = out_root / "checkpoints"
  manifest = checkpoints_dir / "manifest.json"
  if not manifest.exists():
    cmd = [
      cfg.python_bin,
      "scripts/build_keeper_patch_library.py",
      "--library-data",
      *libs,
      "--base-checkpoint",
      cfg.base,
      "--out-dir",
      str(checkpoints_dir),
      "--thresholds",
      *[str(v) for v in cfg.thresholds],
      "--pos-scale",
      str(cfg.pos_scale),
      "--vel-scale",
      str(cfg.vel_scale),
      "--residual-scales",
      *[str(v) for v in cfg.residual_scales],
    ]
    _run("BUILD_PATCH", cmd, out_root / "logs" / "build_patch.log")
  rows = _load_json(manifest)
  return [row["checkpoint"] for row in rows]


def _eval_checkpoint(cfg: Cfg, out_root: Path, checkpoint: str) -> dict:
  name = Path(checkpoint).stem.replace(".", "p")
  eval_json = out_root / "eval" / f"eval_{name}.json"
  code = 0
  if not eval_json.exists():
    cmd = [
      cfg.python_bin,
      "scripts/eval_goalkeeper_official_batched.py",
      "--checkpoint",
      checkpoint,
      "--seeds",
      *[str(seed) for seed in cfg.seeds],
      "--trials-per-seed",
      str(cfg.eval_trials_per_seed),
      "--batch-size",
      str(cfg.batch_size),
      "--parallel-seeds",
      "--seed-gpus",
      *[str(gpu) for gpu in cfg.devices[: min(3, len(cfg.devices))]],
      "--out",
      str(eval_json),
    ]
    code = _run("EVAL", cmd, out_root / "logs" / f"eval_{name}.log", check=False)
  score = -1.0
  if eval_json.exists():
    score = float(_load_json(eval_json).get("mean_block_rate", -1.0))
  return {
    "name": name,
    "checkpoint": checkpoint,
    "score": score,
    "eval": str(eval_json),
    "eval_returncode": code,
  }


def _evaluate_all(cfg: Cfg, out_root: Path, checkpoints: list[str]) -> list[dict]:
  rows = [_eval_checkpoint(cfg, out_root, checkpoint) for checkpoint in checkpoints]
  rows.sort(key=lambda row: row["score"], reverse=True)
  summary = out_root / "summary.json"
  summary.write_text(json.dumps(rows, indent=2) + "\n")
  print(f"[INFO] wrote {summary}", flush=True)
  return rows


def _final_union(cfg: Cfg, out_root: Path, rows: list[dict]) -> None:
  top = [row for row in rows if row["score"] >= 0.0][:3]
  if not top:
    return
  candidates = list(cfg.candidates) + [row["checkpoint"] for row in top]
  names = list(cfg.names) + [f"miss_patch{i}_{row['name']}" for i, row in enumerate(top)]
  cmd = [
    cfg.python_bin,
    "scripts/eval_goalkeeper_policy_union_batched.py",
    "--base",
    cfg.base,
    "--candidates",
    *candidates,
    "--names",
    *names,
    "--seeds",
    *[str(seed) for seed in cfg.seeds],
    "--trials-per-seed",
    str(cfg.final_trials_per_seed),
    "--batch-size",
    str(cfg.batch_size),
    "--parallel-seeds",
    "--seed-gpus",
    *[str(gpu) for gpu in cfg.devices[: min(3, len(cfg.devices))]],
    "--out-json",
    str(out_root / "union_final.json"),
    "--out-csv",
    str(out_root / "union_final.csv"),
  ]
  _run("UNION_FINAL", cmd, out_root / "logs" / "union_final.log", check=False)


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "logs").mkdir(exist_ok=True)
  deadline = time.monotonic() + max(0.5, cfg.hours) * 3600.0
  miss_csv, miss_count = _write_miss_csv(cfg, out_root)
  print(f"[INFO] miss_count={miss_count}", flush=True)

  collect_deadline = deadline - 0.8 * 3600.0
  collect_deadline = max(time.monotonic() + 60.0, collect_deadline)
  libs = _collect(cfg, out_root, miss_csv, collect_deadline)
  print(f"[INFO] library shards={len(libs)}", flush=True)
  for path in libs:
    print(f"  {path}", flush=True)
  if not libs:
    raise RuntimeError("no repair libraries collected")

  checkpoints = _build_patch_checkpoints(cfg, out_root, libs)
  rows = _evaluate_all(cfg, out_root, checkpoints)
  if rows:
    best = rows[0]
    print(
      f"[BEST] {best['name']} score={100.0 * float(best['score']):.2f}% "
      f"checkpoint={best['checkpoint']}",
      flush=True,
    )
  _final_union(cfg, out_root, rows)
  rec = {
    "base": cfg.base,
    "union_csv": cfg.union_csv,
    "miss_csv": str(miss_csv),
    "miss_count": miss_count,
    "libraries": libs,
    "summary": str(out_root / "summary.json"),
    "final_union": str(out_root / "union_final.json"),
  }
  (out_root / "recommendation.json").write_text(json.dumps(rec, indent=2) + "\n")
  print(f"[INFO] wrote {out_root / 'recommendation.json'}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_union_miss_patch"))
