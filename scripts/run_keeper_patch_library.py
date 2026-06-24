"""Collect and evaluate nearest-neighbor patch-library goalkeeper repairs."""

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


@dataclass
class Cfg:
  base: str = "logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt"
  oracle_base: str = "checkpoints/keeper_93_moe6.pt"
  pairwise_csv: str = "logs/keeper_post_repair_auto/pairwise_top.csv"
  candidate: str = "scale0p01_bc0p6"
  out_root: str = "logs/keeper_patch_library"
  python_bin: str = "/data/mjlab-cu126/bin/python"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  hours: float = 2.5
  G: int = 8
  P: int = 128
  iters: int = 12
  elites: int = 12
  batches_per_shard: int = 4
  scenario_pos_jitter: float = 0.0
  scenario_vel_jitter: float = 0.0
  thresholds: tuple[float, ...] = (0.25, 0.35, 0.5, 0.75)
  residual_scales: tuple[float, ...] = (0.5, 0.75, 1.0)
  pos_scale: float = 0.05
  vel_scale: float = 0.15
  trials_per_seed: int = 50
  seed: int = 9910


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
  env = _env(gpu) if gpu is not None else os.environ.copy()
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  print(f"\n[{label}] {_quote(cmd)}", flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
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


def _collect_cmd(cfg: Cfg, out: Path, lib_out: Path, seed: int) -> list[str]:
  return [
    cfg.python_bin,
    "scripts/repair_oracle.py",
    "--checkpoint",
    cfg.oracle_base,
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
    str(cfg.batches_per_shard),
    "--seed",
    str(seed),
    "--w-stable",
    "25.0",
    "--w-final-upright",
    "40.0",
    "--collect-pre-steps",
    "40",
    "--collect-post-steps",
    "10",
    "--device",
    "cuda:0",
    "--out",
    str(out),
    "--library-out",
    str(lib_out),
  ]


def _collect(cfg: Cfg, out_root: Path) -> list[str]:
  deadline = time.monotonic() + max(0.25, cfg.hours) * 3600.0
  repair_dir = out_root / "repairs"
  repair_dir.mkdir(parents=True, exist_ok=True)
  running = []
  shard = 0
  libs: list[str] = []
  while time.monotonic() < deadline:
    busy = {gpu for _, gpu, _, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while free and time.monotonic() < deadline:
      gpu = free.pop(0)
      out = repair_dir / f"patch_data{shard:03d}.pt"
      lib_out = repair_dir / f"patch_library{shard:03d}.pt"
      log = out_root / "logs" / f"collect_patch{shard:03d}_gpu{gpu}.log"
      cmd = _collect_cmd(cfg, out, lib_out, cfg.seed + shard)
      proc, handle = _popen(f"COLLECT {shard}", cmd, log, gpu)
      running.append((shard, gpu, lib_out, proc, handle))
      shard += 1

    time.sleep(10.0)
    still = []
    for idx, gpu, lib_out, proc, handle in running:
      code = proc.poll()
      if code is None:
        still.append((idx, gpu, lib_out, proc, handle))
        continue
      handle.close()
      if code == 0 and lib_out.exists():
        libs.append(str(lib_out))
        print(f"[COLLECT] library {idx} done: {lib_out}", flush=True)
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


def main(cfg: Cfg) -> None:
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "logs").mkdir(exist_ok=True)
  libs = _collect(cfg, out_root)
  if not libs:
    raise RuntimeError("no library shards collected")
  print(f"[INFO] library shards={len(libs)}", flush=True)

  build_cmd = [
    cfg.python_bin,
    "scripts/build_keeper_patch_library.py",
    "--library-data",
    *libs,
    "--base-checkpoint",
    cfg.base,
    "--out-dir",
    str(out_root / "checkpoints"),
    "--thresholds",
    *[str(v) for v in cfg.thresholds],
    "--pos-scale",
    str(cfg.pos_scale),
    "--vel-scale",
    str(cfg.vel_scale),
    "--residual-scales",
    *[str(v) for v in cfg.residual_scales],
  ]
  _run("BUILD", build_cmd, out_root / "logs" / "build_patch_library.log")
  manifest = json.loads((out_root / "checkpoints" / "manifest.json").read_text())
  checkpoints = [row["checkpoint"] for row in manifest]

  eval_cmd = [
    cfg.python_bin,
    "scripts/eval_keeper_big_repair.py",
    "--out-root",
    str(out_root / "eval_sweep"),
    "--checkpoints",
    *checkpoints,
    "--include-base",
    cfg.base,
    "--devices",
    *[str(gpu) for gpu in cfg.devices],
    "--trials-per-seed",
    str(cfg.trials_per_seed),
    "--force",
  ]
  _run("EVAL", eval_cmd, out_root / "logs" / "eval_patch_library.log", check=False)
  summary_path = out_root / "eval_sweep" / "eval_summary.json"
  if summary_path.exists():
    rows = json.loads(summary_path.read_text())
    rows.sort(key=lambda row: float(row.get("score", -1.0)), reverse=True)
    final = out_root / "summary.json"
    final.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"[INFO] wrote {final}", flush=True)
    if rows:
      best = rows[0]
      print(
        f"[BEST] {best['name']} score={100.0 * float(best['score']):.2f}% "
        f"checkpoint={best['checkpoint']}",
        flush=True,
      )


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_patch_library"))
