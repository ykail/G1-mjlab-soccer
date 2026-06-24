"""Fully automated 4-H20 keeper repair run.

This pipeline avoids another blind PPO sweep.  It spends the GPU budget on the
current bottleneck:

1. batched paired union eval for the supplied policies;
2. identify weak regions from the base policy;
3. run CEM repair shards on those regions across all GPUs;
4. distill conservative region-scoped residual checkpoints;
5. batched-evaluate and rank them;
6. run a final union eval over the best new checkpoints.

The script is resumable: existing repair shards and evaluated JSON files are
reused unless outputs are missing.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
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
  candidates: tuple[str, ...] = (
    "logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt",
    "logs/keeper_patch_library_quick/checkpoints/patch_thr0p35_scale0p75.pt",
  )
  names: tuple[str, ...] = ("targeted", "patch")
  out_root: str = "logs/keeper_h20_autorepair"
  python_bin: str = "/data/mjlab-cu126/bin/python"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  hours: float = 4.0

  # Fast eval parameters.
  eval_trials_per_seed: int = 200
  final_trials_per_seed: int = 300
  batch_size: int = 512
  seeds: tuple[int, ...] = (42, 2810, 202686)

  # Region selection.  Empty means auto-select from base by-region rates.
  repair_regions: tuple[int, ...] = ()
  max_auto_regions: int = 2
  weak_region_threshold: float = 0.94

  # CEM repair search.  Increase G/P first on H20 if memory is still low.
  G: int = 18
  P: int = 128
  iters: int = 11
  elites: int = 12
  collect_batches_per_shard: int = 5
  max_shards: int = 999
  collect_fraction: float = 0.68

  # Distillation sweep kept deliberately small.
  distill_epochs: int = 180
  distill_batch_size: int = 65536
  residual_scales: tuple[float, ...] = (0.006, 0.01, 0.016)
  base_bc_coefs: tuple[float, ...] = (0.8, 1.4)
  max_frames_per_file: int = 0


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


def _run(
  label: str,
  cmd: list[str],
  log_path: Path,
  gpu: int | None = None,
  check: bool = True,
) -> int:
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
    tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-120:])
    print(tail, flush=True)
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


def _union_eval_cmd(
  cfg: Cfg,
  out_json: Path,
  out_csv: Path,
  candidates: list[str] | None = None,
  names: list[str] | None = None,
  trials_per_seed: int | None = None,
) -> list[str]:
  candidates = list(cfg.candidates if candidates is None else candidates)
  names = list(cfg.names if names is None else names)
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
    str(trials_per_seed or cfg.eval_trials_per_seed),
    "--batch-size",
    str(cfg.batch_size),
    "--parallel-seeds",
    "--seed-gpus",
    *[str(gpu) for gpu in cfg.devices[: min(3, len(cfg.devices))]],
    "--out-json",
    str(out_json),
    "--out-csv",
    str(out_csv),
  ]
  return cmd


def _initial_union(cfg: Cfg, out_root: Path) -> dict:
  out_json = out_root / "union_initial.json"
  out_csv = out_root / "union_initial.csv"
  if out_json.exists():
    print(f"[INFO] using existing {out_json}", flush=True)
    return _load_json(out_json)
  cmd = _union_eval_cmd(cfg, out_json, out_csv)
  _run("UNION_INITIAL", cmd, out_root / "logs" / "union_initial.log", check=True)
  return _load_json(out_json)


def _choose_regions(cfg: Cfg, union: dict) -> tuple[int, ...]:
  if cfg.repair_regions:
    return tuple(cfg.repair_regions)
  base_regions = union.get("policy_stats", {}).get("base", {}).get("by_region", {})
  scored = []
  for idx, name in enumerate(_REGION_NAMES):
    stats = base_regions.get(name, {})
    trials = int(stats.get("trials", 0))
    rate = float(stats.get("rate", 1.0))
    misses = trials - int(stats.get("blocked", 0))
    if trials > 0:
      scored.append((rate, -misses, idx, name, trials))
  scored.sort()
  selected = [
    idx for rate, _, idx, _, _ in scored
    if rate <= cfg.weak_region_threshold
  ][: cfg.max_auto_regions]
  if not selected and scored:
    selected = [scored[0][2]]
  print("[INFO] region ranking:", flush=True)
  for rate, neg_misses, idx, name, trials in scored:
    print(f"  {idx} {name}: rate={100.0 * rate:.2f}% misses={-neg_misses}/{trials}", flush=True)
  print(f"[INFO] selected repair regions={selected}", flush=True)
  return tuple(selected)


def _region_weights(regions: tuple[int, ...], union: dict) -> list[float]:
  base_regions = union.get("policy_stats", {}).get("base", {}).get("by_region", {})
  weights = []
  for idx in regions:
    name = _REGION_NAMES[idx]
    stats = base_regions.get(name, {})
    rate = float(stats.get("rate", 0.9))
    # More misses -> more repair samples, bounded to keep the scenario mix sane.
    weights.append(max(1.0, min(4.0, (1.0 - rate) * 35.0)))
  return weights


def _repair_cmd(
  cfg: Cfg,
  out: Path,
  lib_out: Path,
  seed: int,
  regions: tuple[int, ...],
  weights: list[float],
) -> list[str]:
  cmd = [
    cfg.python_bin,
    "scripts/repair_oracle.py",
    "--checkpoint",
    cfg.base,
    "--mode",
    "collect",
    "--regions",
    *[str(region) for region in regions],
  ]
  if weights:
    cmd.extend(["--region-weights", *[str(weight) for weight in weights]])
  cmd.extend(
    [
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
      "24.0",
      "--w-final-upright",
      "36.0",
      "--collect-pre-steps",
      "42",
      "--collect-post-steps",
      "10",
      "--device",
      "cuda:0",
      "--out",
      str(out),
      "--library-out",
      str(lib_out),
    ]
  )
  return cmd


def _existing_shards(out_root: Path) -> list[str]:
  return sorted(str(path) for path in (out_root / "repairs").glob("repair_shard*.pt"))


def _collect(cfg: Cfg, out_root: Path, deadline: float, regions: tuple[int, ...], weights: list[float]) -> list[str]:
  repair_dir = out_root / "repairs"
  repair_dir.mkdir(parents=True, exist_ok=True)
  shards = _existing_shards(out_root)
  shard_idx = len(shards)
  running = []
  while time.monotonic() < deadline and shard_idx < cfg.max_shards:
    busy = {gpu for _, gpu, _, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while free and time.monotonic() < deadline and shard_idx < cfg.max_shards:
      gpu = free.pop(0)
      out = repair_dir / f"repair_shard{shard_idx:03d}.pt"
      lib_out = repair_dir / f"library_shard{shard_idx:03d}.pt"
      log = out_root / "logs" / f"collect_shard{shard_idx:03d}_gpu{gpu}.log"
      cmd = _repair_cmd(cfg, out, lib_out, 9000 + shard_idx, regions, weights)
      proc, handle = _popen(f"COLLECT {shard_idx}", cmd, log, gpu)
      running.append((shard_idx, gpu, out, proc, handle))
      shard_idx += 1

    time.sleep(8.0)
    still = []
    for idx, gpu, out, proc, handle in running:
      code = proc.poll()
      if code is None:
        still.append((idx, gpu, out, proc, handle))
        continue
      handle.close()
      if code == 0 and out.exists():
        print(f"[COLLECT] shard {idx} done: {out}", flush=True)
        shards.append(str(out))
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
  return sorted(set(shards))


def _eval_ckpt_cmd(cfg: Cfg, ckpt: Path, out_json: Path) -> list[str]:
  return [
    cfg.python_bin,
    "scripts/eval_goalkeeper_official_batched.py",
    "--checkpoint",
    str(ckpt),
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
    str(out_json),
  ]


def _distill_one(
  cfg: Cfg,
  out_root: Path,
  shards: list[str],
  regions: tuple[int, ...],
  scale: float,
  bc_coef: float,
  gpu: int,
) -> dict:
  name = f"regions{'-'.join(str(r) for r in regions)}_scale{scale:g}_bc{bc_coef:g}".replace(".", "p")
  ckpt = out_root / "distilled" / f"moe6_residual_{name}.pt"
  eval_json = out_root / "eval" / f"eval_{name}.json"
  if not ckpt.exists():
    cmd = [
      cfg.python_bin,
      "scripts/distill_moe6_residual_repairs.py",
      "--data",
      *shards,
      "--base",
      cfg.base,
      "--out",
      str(ckpt),
      "--epochs",
      str(cfg.distill_epochs),
      "--batch-size",
      str(cfg.distill_batch_size),
      "--residual-scale",
      str(scale),
      "--residual-regions",
      *[str(region) for region in regions],
      "--base-bc-coef",
      str(bc_coef),
      "--max-frames-per-file",
      str(cfg.max_frames_per_file),
      "--device",
      "cuda:0",
    ]
    _run("DISTILL", cmd, out_root / "logs" / f"distill_{name}.log", gpu=gpu)
  else:
    print(f"[DISTILL] using existing {ckpt}", flush=True)

  code = 0
  if not eval_json.exists():
    code = _run(
      "EVAL",
      _eval_ckpt_cmd(cfg, ckpt, eval_json),
      out_root / "logs" / f"eval_{name}.log",
      gpu=None,
      check=False,
    )
  score = -1.0
  if eval_json.exists():
    score = float(_load_json(eval_json).get("mean_block_rate", -1.0))
  return {
    "name": name,
    "checkpoint": str(ckpt),
    "score": score,
    "eval": str(eval_json),
    "eval_returncode": code,
  }


def _distill_eval(cfg: Cfg, out_root: Path, shards: list[str], regions: tuple[int, ...]) -> list[dict]:
  if not shards:
    raise RuntimeError("no repair shards available")
  configs = [(scale, bc) for scale in cfg.residual_scales for bc in cfg.base_bc_coefs]
  results = []
  # Distillation itself is fast; run sequentially to avoid several evals fighting
  # over the same 3 seed GPUs.
  for idx, (scale, bc) in enumerate(configs):
    gpu = cfg.devices[idx % len(cfg.devices)]
    results.append(_distill_one(cfg, out_root, shards, regions, scale, bc, gpu))
    results.sort(key=lambda row: row["score"], reverse=True)
    (out_root / "summary_partial.json").write_text(json.dumps(results, indent=2) + "\n")
  results.sort(key=lambda row: row["score"], reverse=True)
  return results


def _final_union(cfg: Cfg, out_root: Path, results: list[dict]) -> dict | None:
  good = [row for row in results if row.get("score", -1.0) >= 0.0]
  if not good:
    return None
  top = good[: min(3, len(good))]
  candidates = list(cfg.candidates) + [row["checkpoint"] for row in top]
  names = list(cfg.names) + [f"new{i}_{row['name']}" for i, row in enumerate(top)]
  out_json = out_root / "union_final.json"
  out_csv = out_root / "union_final.csv"
  cmd = _union_eval_cmd(
    cfg,
    out_json,
    out_csv,
    candidates=candidates,
    names=names,
    trials_per_seed=cfg.final_trials_per_seed,
  )
  _run("UNION_FINAL", cmd, out_root / "logs" / "union_final.log", check=False)
  return _load_json(out_json) if out_json.exists() else None


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "logs").mkdir(exist_ok=True)
  deadline = time.monotonic() + max(0.5, cfg.hours) * 3600.0
  print(f"[INFO] out_root={out_root}", flush=True)
  print(f"[INFO] devices={cfg.devices} hours={cfg.hours}", flush=True)
  print(f"[INFO] base={cfg.base}", flush=True)

  initial = _initial_union(cfg, out_root)
  regions = _choose_regions(cfg, initial)
  weights = _region_weights(regions, initial)
  if not regions:
    raise RuntimeError("no repair regions selected")
  print(f"[INFO] repair regions={regions} weights={weights}", flush=True)

  collect_deadline = deadline - (1.25 * 3600.0)
  collect_deadline = max(time.monotonic() + 60.0, collect_deadline)
  shards = _collect(cfg, out_root, collect_deadline, regions, weights)
  print(f"[INFO] repair shards={len(shards)}", flush=True)
  for shard in shards:
    print(f"  {shard}", flush=True)
  if not shards:
    raise RuntimeError("collection produced no shards")

  results = _distill_eval(cfg, out_root, shards, regions)
  summary = out_root / "summary.json"
  summary.write_text(json.dumps(results, indent=2) + "\n")
  print(f"[INFO] wrote {summary}", flush=True)
  if results:
    best = results[0]
    print(
      f"[BEST] {best['name']} score={100.0 * float(best['score']):.2f}% "
      f"checkpoint={best['checkpoint']}",
      flush=True,
    )

  final_union = _final_union(cfg, out_root, results)
  recommendation = {
    "base": cfg.base,
    "initial_union": str(out_root / "union_initial.json"),
    "selected_regions": list(regions),
    "region_weights": weights,
    "shards": shards,
    "summary": str(summary),
    "best": results[0] if results else None,
    "final_union": str(out_root / "union_final.json") if final_union else "",
  }
  rec = out_root / "recommendation.json"
  rec.write_text(json.dumps(recommendation, indent=2) + "\n")
  print(f"[INFO] wrote {rec}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_h20_autorepair"))
