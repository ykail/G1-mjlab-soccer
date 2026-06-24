"""Create a zero-residual MoE6 checkpoint for passthrough validation."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from src.tasks.soccer.modules.gk_moe6_residual import GoalkeeperMoE6ResidualPolicy

import mjlab.tasks  # noqa: F401
import src.tasks.soccer.config.eval  # noqa: F401


@dataclass
class Cfg:
  base: str = "checkpoints/keeper_93_moe6.pt"
  out: str = "logs/keeper_big_repair_v2/distilled/moe6_zero_residual.pt"
  residual_scale: float = 0.0
  device: str = "cuda:0"


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 1
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device), clip_actions=100.0)
  bundle = torch.load(cfg.base, map_location="cpu", weights_only=False)
  if not isinstance(bundle, dict) or not bundle.get("moe6"):
    raise ValueError(f"base is not a MoE6 bundle: {cfg.base}")
  policy = GoalkeeperMoE6ResidualPolicy(
    env,
    bundle,
    cfg.device,
    residual_scale=cfg.residual_scale,
  ).to(cfg.device)
  saved = {
    "moe6_residual": True,
    "base_moe6": bundle,
    "policy_state_dict": policy.state_dict(),
    "hidden_dims": (512, 256, 128),
    "activation": "elu",
    "residual_scale": cfg.residual_scale,
    "residual_regions": (1, 2, 3, 5),
    "source_data": (),
    "passthrough_validation": True,
  }
  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  torch.save(saved, cfg.out)
  print(f"[INFO] wrote zero-residual MoE6 checkpoint to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="make_moe6_zero_residual"))
