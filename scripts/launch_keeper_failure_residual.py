"""Launch failure-replay ballistic-residual keeper runs across GPUs.

This wrapper assumes a failure CSV has already been collected with
``collect_goalkeeper_failures.py``.  It starts one independent training process
per seed/config, which is usually a better use of 4 H20 cards than trying to
make one small PPO model data-parallel.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  init: str = "src/assets/soccer/weight/model_repaired_lyk.pt"
  failure_csv: str = "logs/keeper_failure_residual/failures.csv"
  out_dir: str = "logs/keeper_failure_residual/checkpoints"
  log_dir: str = "logs/keeper_failure_residual/train_logs"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  seeds: tuple[int, ...] = (11, 22, 33, 44)
  num_envs: int = 8192
  warmup: int = 20
  block_iters: int = 15
  blocks: int = 80
  eval_resets: int = 6
  lr_values: tuple[float, ...] = (3.0e-5, 5.0e-5)
  std_values: tuple[float, ...] = (0.025, 0.035)
  residual_scale_values: tuple[float, ...] = (0.18, 0.25)
  failure_replay_ratios: tuple[float, ...] = (0.35, 0.55)
  failure_pos_jitter: float = 0.015
  failure_vel_jitter: float = 0.035
  stable_save_weight: float = 0.75
  rollback_drop: float = 0.005
  max_runs: int = 0
  extra_args: tuple[str, ...] = ()


def _fmt_float(value: float) -> str:
  return f"{value:g}".replace(".", "p").replace("-", "m")


def _run_specs(cfg: Cfg) -> list[dict[str, float | int]]:
  specs: list[dict[str, float | int]] = []
  for seed in cfg.seeds:
    for lr in cfg.lr_values:
      for std in cfg.std_values:
        for scale in cfg.residual_scale_values:
          for replay_ratio in cfg.failure_replay_ratios:
            specs.append(
              {
                "seed": seed,
                "lr": lr,
                "std": std,
                "scale": scale,
                "replay_ratio": replay_ratio,
              }
            )
  if cfg.max_runs > 0:
    specs = specs[: cfg.max_runs]
  return specs


def _run_name(spec: dict[str, float | int]) -> str:
  return (
    f"seed{spec['seed']}_lr{_fmt_float(float(spec['lr']))}"
    f"_std{_fmt_float(float(spec['std']))}"
    f"_rs{_fmt_float(float(spec['scale']))}"
    f"_fr{_fmt_float(float(spec['replay_ratio']))}"
  )


def _cmd(cfg: Cfg, spec: dict[str, float | int]) -> list[str]:
  name = _run_name(spec)
  return [
    sys.executable,
    "scripts/train_ballistic_residual.py",
    "--init",
    cfg.init,
    "--out",
    str(Path(cfg.out_dir) / f"{name}.pt"),
    "--failure-csv",
    cfg.failure_csv,
    "--failure-replay-ratio",
    str(spec["replay_ratio"]),
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
    str(spec["lr"]),
    "--std",
    str(spec["std"]),
    "--residual-scale",
    str(spec["scale"]),
    "--seed",
    str(spec["seed"]),
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
    "1.4",
    "--w-recovery",
    "5.0",
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
    "0.06",
    "--desired-kl",
    "0.003",
    "--device",
    "cuda:0",
    *cfg.extra_args,
  ]


def main(cfg: Cfg) -> None:
  if not cfg.devices:
    raise ValueError("at least one device is required")
  if not Path(cfg.failure_csv).exists():
    raise FileNotFoundError(
      f"failure csv not found: {cfg.failure_csv}; run collect_goalkeeper_failures.py first"
    )
  Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
  Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)

  pending = _run_specs(cfg)
  if not pending:
    raise ValueError("no runs selected")
  running: list[tuple[str, int, subprocess.Popen, object]] = []
  failed: list[tuple[str, int]] = []
  started = 0

  def launch(spec: dict[str, float | int], gpu: int) -> None:
    nonlocal started
    name = _run_name(spec)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["MUJOCO_EGL_DEVICE_ID"] = "0"
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("WANDB_MODE", "disabled")
    log_path = Path(cfg.log_dir) / f"{name}_gpu{gpu}.log"
    cmd = _cmd(cfg, spec)
    with (Path(cfg.log_dir) / f"{name}.cmd").open("w") as f:
      f.write(" ".join(cmd) + "\n")
    log = open(log_path, "w")
    print(f"[LAUNCH] {name} on gpu {gpu}", flush=True)
    print(f"[LAUNCH] log -> {log_path}", flush=True)
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    running.append((name, gpu, proc, log))
    started += 1

  while pending or running:
    busy = {gpu for _, gpu, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while pending and free:
      launch(pending.pop(0), free.pop(0))

    time.sleep(10.0)
    still_running = []
    for name, gpu, proc, log in running:
      code = proc.poll()
      if code is None:
        still_running.append((name, gpu, proc, log))
        continue
      log.close()
      if code != 0:
        failed.append((name, code))
        print(f"[FAIL] {name} exited with code {code}", flush=True)
      else:
        print(f"[DONE] {name} on gpu {gpu}", flush=True)
    running = still_running

  print(f"[INFO] started {started} runs", flush=True)
  if failed:
    raise SystemExit(f"failed runs: {failed}")
  print(f"[DONE] checkpoints saved under {cfg.out_dir}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="launch_keeper_failure_residual"))
