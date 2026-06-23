"""Failure-replay fine-tuning for MoE6 goalkeeper specialists.

This keeps the PR #10 architecture intact: six frozen-base ballistic-residual
experts plus a ballistic gate.  Instead of wrapping the whole MoE in a new
global residual, it continues the region experts on the failure bank collected
from the bundled MoE checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  expert_dir: str = "logs/keeper_moe6_failure_replay/base_experts"
  failure_csv: str = "logs/keeper_moe6_failure_replay/failures_keeper_93.csv"
  out_dir: str = "logs/keeper_moe6_failure_replay/experts"
  log_dir: str = "logs/keeper_moe6_failure_replay/train_logs"
  bundle_out: str = "logs/keeper_moe6_failure_replay/keeper_93_failure_replay_moe6.pt"
  prefix: str = "stable_sr"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  regions: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
  num_envs: int = 8192
  warmup: int = 0
  block_iters: int = 12
  blocks: int = 60
  eval_resets: int = 6
  lr: float = 2.0e-5
  std: float = 0.02
  residual_scale: float = 0.18
  failure_replay_ratio: float = 0.65
  failure_pos_jitter: float = 0.012
  failure_vel_jitter: float = 0.025
  stable_save_weight: float = 0.75
  rollback_drop: float = 0.003
  extra_args: tuple[str, ...] = ()


def _meta(expert_dir: str) -> dict:
  path = Path(expert_dir) / "moe6_meta.json"
  if not path.exists():
    return {}
  return json.loads(path.read_text())


def _cmd(cfg: Cfg, region: int) -> list[str]:
  init = Path(cfg.expert_dir) / f"{cfg.prefix}{region}.pt"
  out = Path(cfg.out_dir) / f"{cfg.prefix}{region}.pt"
  return [
    sys.executable,
    "scripts/train_ballistic_residual.py",
    "--init",
    str(init),
    "--out",
    str(out),
    "--train-regions",
    str(region),
    "--failure-csv",
    cfg.failure_csv,
    "--failure-regions",
    str(region),
    "--failure-replay-ratio",
    str(cfg.failure_replay_ratio),
    "--failure-pos-jitter",
    str(cfg.failure_pos_jitter),
    "--failure-vel-jitter",
    str(cfg.failure_vel_jitter),
    "--num-envs",
    str(cfg.num_envs),
    "--warmup",
    str(cfg.warmup),
    "--block-iters",
    str(cfg.block_iters),
    "--blocks",
    str(cfg.blocks),
    "--eval-resets",
    str(cfg.eval_resets),
    "--lr",
    str(cfg.lr),
    "--std",
    str(cfg.std),
    "--residual-scale",
    str(cfg.residual_scale),
    "--stable-save-weight",
    str(cfg.stable_save_weight),
    "--rollback-drop",
    str(cfg.rollback_drop),
    "--w-conceded",
    "18.0",
    "--w-intercept",
    "3.0",
    "--w-body",
    "1.5",
    "--w-stop",
    "1.0",
    "--w-posture",
    "1.6",
    "--w-recovery",
    "6.0",
    "--w-line",
    "0.2",
    "--w-no-retreat",
    "0.3",
    "--w-feet-slip",
    "0.06",
    "--w-ang-vel",
    "0.03",
    "--w-post-save-ang-vel",
    "0.08",
    "--post-save-action-rate",
    "0.06",
    "--action-rate",
    "0.07",
    "--clip-param",
    "0.05",
    "--desired-kl",
    "0.0025",
    "--device",
    "cuda:0",
    *cfg.extra_args,
  ]


def _bundle(cfg: Cfg) -> None:
  meta = _meta(cfg.expert_dir)
  cmd = [
    sys.executable,
    "scripts/bundle_moe6.py",
    "--expert-dir",
    cfg.out_dir,
    "--prefix",
    cfg.prefix,
    "--out",
    cfg.bundle_out,
    "--z-low",
    str(meta.get("z_low", 0.85)),
    "--z-up",
    str(meta.get("z_up", 1.35)),
    "--vz-low",
    str(meta.get("vz_low", -99.0)),
    "--latch-hi",
    str(meta.get("latch_hi", 5.0)),
    "--land-x",
    str(meta.get("land_x", 0.0)),
    "--mirror-map",
    str(meta.get("mirror_map", "")),
  ]
  print("[BUNDLE] " + " ".join(cmd), flush=True)
  subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def main(cfg: Cfg) -> None:
  if not cfg.devices:
    raise ValueError("at least one device is required")
  if not Path(cfg.failure_csv).exists():
    raise FileNotFoundError(cfg.failure_csv)
  for region in cfg.regions:
    path = Path(cfg.expert_dir) / f"{cfg.prefix}{region}.pt"
    if not path.exists():
      raise FileNotFoundError(path)

  Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
  Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
  for region in range(6):
    src = Path(cfg.expert_dir) / f"{cfg.prefix}{region}.pt"
    dst = Path(cfg.out_dir) / f"{cfg.prefix}{region}.pt"
    if not dst.exists():
      shutil.copy2(src, dst)

  pending = list(cfg.regions)
  running: list[tuple[int, int, subprocess.Popen, object]] = []
  failed: list[tuple[int, int]] = []

  def launch(region: int, gpu: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("WANDB_MODE", "disabled")
    log_path = Path(cfg.log_dir) / f"{cfg.prefix}{region}_gpu{gpu}.log"
    cmd = _cmd(cfg, region)
    with (Path(cfg.log_dir) / f"{cfg.prefix}{region}.cmd").open("w") as f:
      f.write(" ".join(cmd) + "\n")
    print(f"[LAUNCH] region {region} on gpu {gpu}", flush=True)
    print(f"[LAUNCH] log -> {log_path}", flush=True)
    log = open(log_path, "w")
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    running.append((region, gpu, proc, log))

  while pending or running:
    busy = {gpu for _, gpu, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while pending and free:
      launch(pending.pop(0), free.pop(0))

    time.sleep(10.0)
    still_running = []
    for region, gpu, proc, log in running:
      code = proc.poll()
      if code is None:
        still_running.append((region, gpu, proc, log))
        continue
      log.close()
      if code != 0:
        failed.append((region, code))
        print(f"[FAIL] region {region} exited with code {code}", flush=True)
      else:
        print(f"[DONE] region {region} on gpu {gpu}", flush=True)
    running = still_running

  if failed:
    raise SystemExit(f"failed regions: {failed}")
  _bundle(cfg)
  print(f"[DONE] bundled checkpoint: {cfg.bundle_out}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="launch_moe6_failure_replay"))
