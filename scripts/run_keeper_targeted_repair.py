"""Targeted repair loop from paired-eval failures.

This is the next step after a residual reaches ~94-95% but region masks do not
improve it.  It reuses ``pairwise_top.csv`` to collect CEM repairs only for
scenarios that a selected candidate still failed, then distills and evaluates
new conservative residuals.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  base: str = "checkpoints/keeper_93_moe6.pt"
  pairwise_csv: str = "logs/keeper_post_repair_auto/pairwise_top.csv"
  candidate: str = "scale0p01_bc0p6"
  extra_data: tuple[str, ...] = ()
  out_root: str = "logs/keeper_targeted_repair"
  python_bin: str = "/data/mjlab-cu126/bin/python"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  hours: float = 4.0
  G: int = 12
  P: int = 96
  iters: int = 10
  elites: int = 10
  collect_batches_per_shard: int = 6
  max_shards: int = 999
  scenario_pos_jitter: float = 0.015
  scenario_vel_jitter: float = 0.035
  distill_epochs: int = 140
  batch_size: int = 32768
  residual_scales: tuple[float, ...] = (0.005, 0.01, 0.02, 0.04)
  base_bc_coefs: tuple[float, ...] = (0.3, 0.6, 1.0)
  official_trials_per_seed: int = 50
  seed: int = 8110


def _env(gpu: int) -> dict[str, str]:
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  return env


def _quote(cmd: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in cmd)


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


def _run(label: str, cmd: list[str], log_path: Path, gpu: int | None = None, check: bool = True) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] {_quote(cmd)}", flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  env = _env(gpu) if gpu is not None else os.environ.copy()
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  with log_path.open("w") as log:
    log.write("[CMD] " + _quote(cmd) + "\n")
    proc = subprocess.run(
      cmd,
      cwd=_REPO_ROOT,
      env=env,
      stdout=log,
      stderr=subprocess.STDOUT,
      check=False,
    )
  if check and proc.returncode != 0:
    print("\n".join(log_path.read_text(errors="replace").splitlines()[-100:]), flush=True)
    raise subprocess.CalledProcessError(proc.returncode, cmd)
  return proc.returncode


def _repair_cmd(cfg: Cfg, out: str, seed: int, batches: int) -> list[str]:
  return [
    cfg.python_bin,
    "scripts/repair_oracle.py",
    "--checkpoint",
    cfg.base,
    "--mode",
    "collect",
    "--scenario-csv",
    cfg.pairwise_csv,
    "--scenario-candidate",
    cfg.candidate,
    "--scenario-blocked-column",
    "candidate_blocked",
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
    str(batches),
    "--seed",
    str(seed),
    "--w-stable",
    "20.0",
    "--w-final-upright",
    "30.0",
    "--collect-pre-steps",
    "40",
    "--collect-post-steps",
    "10",
    "--device",
    "cuda:0",
    "--out",
    out,
  ]


def _existing_shards(out_root: Path) -> list[str]:
  return sorted(str(path) for path in (out_root / "repairs").glob("targeted_shard*.pt"))


def _collect(cfg: Cfg, out_root: Path, deadline: float) -> list[str]:
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
      out = repair_dir / f"targeted_shard{shard_idx:03d}.pt"
      log = out_root / "logs" / f"collect_targeted_shard{shard_idx:03d}_gpu{gpu}.log"
      cmd = _repair_cmd(cfg, str(out), cfg.seed + shard_idx, cfg.collect_batches_per_shard)
      proc, handle = _popen(f"COLLECT {shard_idx}", cmd, log, gpu)
      running.append((shard_idx, gpu, out, proc, handle))
      shard_idx += 1

    time.sleep(10.0)
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


def _distill_eval(cfg: Cfg, out_root: Path, shards: list[str]) -> list[dict]:
  results = []
  data = [*cfg.extra_data, *shards]
  for scale in cfg.residual_scales:
    for bc_coef in cfg.base_bc_coefs:
      name = f"scale{scale:g}_bc{bc_coef:g}".replace(".", "p")
      ckpt = out_root / "distilled" / f"targeted_moe6_residual_{name}.pt"
      distill_cmd = [
        cfg.python_bin,
        "scripts/distill_moe6_residual_repairs.py",
        "--data",
        *data,
        "--base",
        cfg.base,
        "--out",
        str(ckpt),
        "--epochs",
        str(cfg.distill_epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--residual-scale",
        str(scale),
        "--base-bc-coef",
        str(bc_coef),
        "--device",
        "cuda:0",
      ]
      _run("DISTILL", distill_cmd, out_root / "logs" / f"distill_{name}.log", gpu=cfg.devices[0])
      eval_json = out_root / "eval" / f"eval_{name}.json"
      eval_cmd = [
        cfg.python_bin,
        "scripts/eval_goalkeeper_official_seeds.py",
        "--checkpoint",
        str(ckpt),
        "--trials-per-seed",
        str(cfg.official_trials_per_seed),
        "--parallel-seeds",
        "--seed-gpus",
        *[str(gpu) for gpu in cfg.devices[:3]],
        "--out",
        str(eval_json),
      ]
      code = _run("EVAL", eval_cmd, out_root / "logs" / f"eval_{name}.log", check=False)
      score = -1.0
      if code == 0 and eval_json.exists():
        score = float(json.loads(eval_json.read_text()).get("mean_block_rate", -1.0))
      results.append(
        {
          "name": name,
          "checkpoint": str(ckpt),
          "score": score,
          "eval": str(eval_json),
          "eval_returncode": code,
        }
      )
  return sorted(results, key=lambda row: row["score"], reverse=True)


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "logs").mkdir(exist_ok=True)
  deadline = time.monotonic() + max(0.25, cfg.hours) * 3600.0
  print(f"[INFO] base={cfg.base}", flush=True)
  print(f"[INFO] pairwise_csv={cfg.pairwise_csv} candidate={cfg.candidate}", flush=True)
  print(f"[INFO] out_root={out_root} devices={cfg.devices}", flush=True)

  collect_deadline = deadline - 1.0 * 3600.0
  collect_deadline = max(time.monotonic() + 60.0, collect_deadline)
  shards = _collect(cfg, out_root, collect_deadline)
  print(f"[INFO] targeted shards={len(shards)}", flush=True)
  for shard in shards:
    print(f"  {shard}", flush=True)
  if not shards:
    raise RuntimeError("no targeted repair shards collected")

  if cfg.extra_data:
    print(f"[INFO] extra distill data={len(cfg.extra_data)}", flush=True)
    for path in cfg.extra_data:
      print(f"  {path}", flush=True)
  results = _distill_eval(cfg, out_root, shards)
  summary = out_root / "summary.json"
  summary.write_text(json.dumps(results, indent=2) + "\n")
  print(f"[INFO] wrote {summary}", flush=True)
  if results:
    best = results[0]
    print(
      f"[BEST] {best['name']} score={100.0 * best['score']:.2f}% "
      f"checkpoint={best['checkpoint']}",
      flush=True,
    )


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_targeted_repair"))
