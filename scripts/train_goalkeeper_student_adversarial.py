"""Train a single-network goalkeeper student against a frozen shooter.

This entrypoint is intentionally separate from ``train_adversarial.py``.  The
alternating scheduler supports MoE and expert-level experiments, but this path
keeps the trainable goalkeeper as one end-to-end FiLM LSTM actor.  MoE, residual,
and patch-library keepers can still be used as teachers or Phase-1 deploy
policies, but they are rejected as PPO init checkpoints here because they are
not a single trainable student actor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import tyro

try:
  from scripts.train import TrainConfig, launch_training
except ModuleNotFoundError:  # Direct execution: python scripts/train_goalkeeper_student_adversarial.py
  from train import TrainConfig, launch_training

from scripts.train_goalkeeper_student_ppo import FineTuneConfig, build_train_config


_TASK_ID = "Unitree-G1-Goalkeeper-Student-Compete-Adversarial"


@dataclass
class Cfg:
  init: str
  """Single-network goalkeeper student checkpoint used as PPO init."""

  shooter: str = ""
  """Frozen shooter checkpoint. If omitted, search common checkpoint paths."""

  shooter_task_id: str = ""
  """Eval task used to reconstruct the frozen shooter; inferred when empty."""

  teacher: str = ""
  """Optional MoE/MoE7 teacher for online distillation. Leave empty to disable."""

  out_dir: str = "logs/adversarial_student_keeper"
  run_name: str = "student_keeper_adv"
  python_bin: str = sys.executable
  gpu_ids: str = "0"
  num_envs: int = 1024
  max_iterations: int = 3000
  seed: int = 2810
  save_interval: int = 100

  distill_coef: float = 0.2
  distill_final_coef: float = 0.0
  distill_anneal_updates: int = 1500
  entropy_coef: float = 5.0e-4
  learning_rate: float = 1.0e-4
  min_action_std: float = 0.05
  max_action_std: float = 0.35
  actor_mean_clip: float = 2.0
  mask_idle_actor_loss: bool = True
  idle_deterministic_actions: bool = True
  critic_warmup_iterations: int = 50

  dry_run: bool = False
  eval_trials: int = 0
  eval_device: str = "cuda:0"


def _load_checkpoint(path: str) -> dict[str, Any]:
  loaded = torch.load(path, map_location="cpu", weights_only=False)
  if not isinstance(loaded, dict):
    raise ValueError(f"{path} is not a dict checkpoint")
  return loaded


def _checkpoint_kind(path: str) -> str:
  loaded = _load_checkpoint(path)
  if loaded.get("keeper_patch_library"):
    return "patch-library"
  if loaded.get("moe6") or "sr" in loaded:
    return "moe"
  if loaded.get("moe6_residual"):
    return "moe-residual"
  actor = loaded.get("actor_state_dict")
  if isinstance(actor, dict):
    if any(key.startswith("condition_encoder.") for key in actor):
      return "student"
    if any(key.startswith("history_encoder.") for key in actor):
      return "goalkeeper-actor-critic"
    if any(".rnn." in key or key.startswith("rnn.") for key in actor):
      return "recurrent"
  if "model_state_dict" in loaded:
    return "reference-himppo"
  return "unknown"


def _require_student_init(path: str) -> None:
  kind = _checkpoint_kind(path)
  if kind != "student":
    raise ValueError(
      "--init must be a single-network GoalkeeperStudentFiLMActor checkpoint. "
      f"Got {kind!r}: {path}. Use MoE/patch checkpoints as --teacher or final "
      "Phase-1 deploy policies, not as adversarial PPO init."
    )


def _require_moe_teacher(path: str) -> None:
  if not path:
    return
  loaded = _load_checkpoint(path)
  bundle = loaded["actor_state_dict"] if isinstance(loaded.get("actor_state_dict"), dict) else loaded
  if not isinstance(bundle, dict) or "sr" not in bundle:
    raise ValueError(
      "--teacher must be a MoE/MoE7 bundle containing sr experts because "
      f"GoalkeeperStudentPPO distills from MoE7PrepareGoalkeeperActor: {path}."
    )
  if not any(key in bundle for key in ("idle", "prepare", "idle_expert")):
    raise ValueError(
      "--teacher must include an idle/prepare expert. A plain MoE6 bundle can "
      f"be a Phase-1 deploy policy, but not the student PPO teacher: {path}."
    )


def _find_shooter_checkpoint(explicit: str) -> str:
  if explicit:
    if not Path(explicit).exists():
      raise FileNotFoundError(f"shooter checkpoint not found: {explicit}")
    return explicit

  candidates = [
    "checkpoints/stage6/model_*.pt",
    "checkpoints/stage5/model_*.pt",
    "checkpoints/stage4/model_*.pt",
    "logs/rsl_rl/g1_soccer/*stage6*/model_*.pt",
    "logs/rsl_rl/g1_soccer/*stage5*/model_*.pt",
    "logs/rsl_rl/g1_soccer/*stage4*/model_*.pt",
    "logs/rsl_rl/g1_soccer/*stage3*/model_*.pt",
    "checkpoints/stage2/model_100000.pt",
  ]
  found: list[Path] = []
  for pattern in candidates:
    found.extend(Path(".").glob(pattern))
  found = [path for path in found if path.is_file()]
  if not found:
    raise FileNotFoundError(
      "No shooter checkpoint found. Pass --shooter explicitly; common good paths "
      "are logs/rsl_rl/g1_soccer/<stage4-or-stage6-run>/model_*.pt."
    )
  found.sort(key=lambda p: p.stat().st_mtime)
  return str(found[-1])


def _infer_shooter_task_id(checkpoint: str, explicit: str) -> str:
  if explicit:
    return explicit
  lowered = checkpoint.lower()
  if "stage6" in lowered:
    return "Eval-Shooter-Stage6"
  if "stage5" in lowered:
    return "Eval-Shooter-Stage5"
  if "stage4" in lowered:
    return "Eval-Shooter-Stage4"
  if "stage3" in lowered:
    return "Eval-Shooter-Stage3"
  return "Eval-Shooter"


def _parse_gpu_ids(raw: str) -> list[int] | str | None:
  raw = raw.strip().lower()
  if raw in ("", "none", "cpu"):
    return None
  if raw == "all":
    return "all"
  return [int(part) for part in raw.split(",") if part.strip()]


def _build_config(cfg: Cfg, shooter_ckpt: str) -> TrainConfig:
  parsed_gpus = _parse_gpu_ids(cfg.gpu_ids)
  base = FineTuneConfig(
    init=cfg.init,
    task_id=_TASK_ID,
    num_envs=cfg.num_envs,
    max_iterations=cfg.max_iterations,
    run_name=cfg.run_name,
    device_ids=parsed_gpus if isinstance(parsed_gpus, list) else None,
    teacher=cfg.teacher,
    distill_coef=cfg.distill_coef if cfg.teacher else 0.0,
    distill_final_coef=cfg.distill_final_coef,
    distill_anneal_updates=cfg.distill_anneal_updates,
    entropy_coef=cfg.entropy_coef,
    learning_rate=cfg.learning_rate,
    min_action_std=cfg.min_action_std,
    max_action_std=cfg.max_action_std,
    actor_mean_clip=cfg.actor_mean_clip,
    mask_idle_actor_loss=cfg.mask_idle_actor_loss,
    idle_deterministic_actions=cfg.idle_deterministic_actions,
    critic_warmup_iterations=cfg.critic_warmup_iterations,
    seed=cfg.seed,
    save_interval=cfg.save_interval,
    delayed_launch=False,
  )
  train_cfg = build_train_config(base)
  train_cfg.frozen_opponent_checkpoint_path = shooter_ckpt
  train_cfg.frozen_opponent_role = "shooter"
  train_cfg.frozen_opponent_task_id = _infer_shooter_task_id(shooter_ckpt, cfg.shooter_task_id)
  train_cfg.gpu_ids = parsed_gpus
  train_cfg.agent.experiment_name = Path(cfg.out_dir).name
  return train_cfg


def _write_manifest(cfg: Cfg, shooter_ckpt: str, train_cfg: TrainConfig) -> Path:
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  manifest = {
    "created_at": datetime.now().isoformat(timespec="seconds"),
    "task_id": _TASK_ID,
    "init": cfg.init,
    "init_kind": _checkpoint_kind(cfg.init),
    "shooter": shooter_ckpt,
    "shooter_task_id": train_cfg.frozen_opponent_task_id,
    "teacher": cfg.teacher or None,
    "gpu_ids": cfg.gpu_ids,
    "num_envs": cfg.num_envs,
    "max_iterations": cfg.max_iterations,
    "run_name": cfg.run_name,
    "train_config": asdict(train_cfg),
  }
  path = out / "manifest.json"
  path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
  return path


def _run_compete_eval(cfg: Cfg, checkpoint: str, shooter_ckpt: str) -> None:
  if cfg.eval_trials <= 0:
    return
  out = Path(cfg.out_dir)
  out.mkdir(parents=True, exist_ok=True)
  shooter_log = out / "shooter_api.log"
  keeper_log = out / "keeper_api.log"
  compete_log = out / "compete_eval.log"
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("PYOPENGL_PLATFORM", "egl")

  with shooter_log.open("w") as s_log, keeper_log.open("w") as k_log:
    shooter_proc = subprocess.Popen(
      [
        cfg.python_bin,
        "scripts/api_server.py",
        "--checkpoint", shooter_ckpt,
        "--port", "8100",
        "--task", "shooter",
        "--strategy", "gk-aware",
        "--device", cfg.eval_device,
      ],
      env=env,
      stdout=s_log,
      stderr=subprocess.STDOUT,
    )
    keeper_proc = subprocess.Popen(
      [
        cfg.python_bin,
        "scripts/api_server.py",
        "--checkpoint", checkpoint,
        "--port", "8101",
        "--task", "goalkeeper",
        "--device", cfg.eval_device,
      ],
      env=env,
      stdout=k_log,
      stderr=subprocess.STDOUT,
    )
    try:
      import time

      time.sleep(15.0)
      with compete_log.open("w") as c_log:
        subprocess.run(
          [
            cfg.python_bin,
            "scripts/compete.py",
            "--shooter-api", "http://127.0.0.1:8100",
            "--goalkeeper-api", "http://127.0.0.1:8101",
            "--headless",
            "--num-trials", str(cfg.eval_trials),
            "--device", cfg.eval_device,
          ],
          env=env,
          stdout=c_log,
          stderr=subprocess.STDOUT,
          check=False,
        )
    finally:
      shooter_proc.terminate()
      keeper_proc.terminate()
      try:
        shooter_proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        shooter_proc.kill()
      try:
        keeper_proc.wait(timeout=10)
      except subprocess.TimeoutExpired:
        keeper_proc.kill()


def _latest_run_checkpoint(out_dir: str, run_name: str) -> str | None:
  log_root = Path("logs") / "rsl_rl" / Path(out_dir).name
  run_dirs = sorted(log_root.glob(f"*_{run_name}"), key=lambda path: path.stat().st_mtime)
  if not run_dirs:
    return None
  models = sorted(
    run_dirs[-1].glob("model_*.pt"),
    key=lambda path: int(path.stem.split("_", maxsplit=1)[1]),
  )
  return str(models[-1]) if models else None


def main(cfg: Cfg) -> None:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  _require_student_init(cfg.init)
  _require_moe_teacher(cfg.teacher)
  shooter_ckpt = _find_shooter_checkpoint(cfg.shooter)
  train_cfg = _build_config(cfg, shooter_ckpt)
  manifest = _write_manifest(cfg, shooter_ckpt, train_cfg)
  print(f"[INFO] single-network keeper adversarial manifest: {manifest}", flush=True)
  print(f"[INFO] init={cfg.init}", flush=True)
  print(f"[INFO] shooter={shooter_ckpt}", flush=True)
  print(f"[INFO] shooter_task_id={train_cfg.frozen_opponent_task_id}", flush=True)
  if cfg.dry_run:
    print("[INFO] dry-run only; no training launched.", flush=True)
    return
  launch_training(_TASK_ID, train_cfg)
  latest = _latest_run_checkpoint(cfg.out_dir, cfg.run_name)
  if latest is not None:
    print(f"[INFO] latest checkpoint: {latest}", flush=True)
    _run_compete_eval(cfg, latest, shooter_ckpt)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="train_goalkeeper_student_adversarial"))
