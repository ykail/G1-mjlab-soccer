"""Distill repair-oracle data into a frozen-base MoE6 residual policy."""

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
  data: tuple[str, ...] = ("logs/keeper_big_repair/repairs.pt",)
  base: str = "checkpoints/keeper_93_moe6.pt"
  out: str = "logs/keeper_big_repair/keeper_93_moe6_residual.pt"
  epochs: int = 80
  batch_size: int = 32768
  lr: float = 3.0e-4
  lr_final: float = 3.0e-5
  residual_scale: float = 0.18
  residual_l2: float = 5.0e-4
  base_bc_coef: float = 0.05
  val_frac: float = 0.02
  blocked_only: bool = True
  max_frames_per_file: int = 0
  max_frames: int = 0
  seed: int = 2810
  device: str = "cuda:0"


def _subsample(
  obs: torch.Tensor,
  act: torch.Tensor,
  base_act: torch.Tensor | None,
  max_frames: int,
  gen: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  if max_frames <= 0 or obs.shape[0] <= max_frames:
    return obs, act, base_act
  idx = torch.randperm(obs.shape[0], generator=gen)[:max_frames]
  return obs[idx], act[idx], base_act[idx] if base_act is not None else None


def _load_data(cfg: Cfg) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  gen = torch.Generator(device="cpu")
  gen.manual_seed(cfg.seed)
  obs_parts, act_parts, base_parts = [], [], []
  has_base_act = True
  for path in cfg.data:
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = data["obs"]
    act = data["act"]
    base_act = data.get("base_act")
    if base_act is None:
      has_base_act = False
    if cfg.blocked_only:
      mask = data["blocked"].bool()
      obs = obs[mask]
      act = act[mask]
      if base_act is not None:
        base_act = base_act[mask]
    obs, act, base_act = _subsample(obs, act, base_act, cfg.max_frames_per_file, gen)
    obs_parts.append(obs)
    act_parts.append(act)
    if base_act is not None:
      base_parts.append(base_act)
    print(f"[INFO] loaded {path}: {obs.shape[0]} frames", flush=True)
  obs = torch.cat(obs_parts)
  act = torch.cat(act_parts)
  base_act = torch.cat(base_parts) if has_base_act and base_parts else None
  return _subsample(obs, act, base_act, cfg.max_frames, gen)


def main(cfg: Cfg) -> None:
  configure_torch_backends()
  torch.manual_seed(cfg.seed)
  dev = cfg.device

  X, Y, B = _load_data(cfg)
  n = X.shape[0]
  if n == 0:
    raise ValueError("no training frames after filtering")
  print(f"[INFO] total {n} pairs obs={X.shape[1]} act={Y.shape[1]}", flush=True)

  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = 2
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)
  bundle = torch.load(cfg.base, map_location="cpu", weights_only=False)
  if not isinstance(bundle, dict) or not bundle.get("moe6"):
    raise ValueError(f"base is not a MoE6 bundle: {cfg.base}")
  policy = GoalkeeperMoE6ResidualPolicy(
    env,
    bundle,
    dev,
    residual_scale=cfg.residual_scale,
  ).to(dev)
  optim = torch.optim.Adam(policy.residual.parameters(), lr=cfg.lr)

  perm = torch.randperm(n)
  n_val = int(max(0, min(n - 1, round(n * cfg.val_frac)))) if n > 1 else 0
  val_idx = perm[:n_val]
  train_idx = perm[n_val:]
  print(f"[INFO] train={train_idx.numel()} val={val_idx.numel()}", flush=True)

  for epoch in range(cfg.epochs):
    frac = epoch / max(1, cfg.epochs - 1)
    lr = cfg.lr + (cfg.lr_final - cfg.lr) * frac
    for group in optim.param_groups:
      group["lr"] = lr

    policy.train()
    train_perm = train_idx[torch.randperm(train_idx.numel())]
    last = 0.0
    for start in range(0, train_perm.numel(), cfg.batch_size):
      idx = train_perm[start : start + cfg.batch_size]
      xb = X[idx].to(dev, non_blocking=True)
      yb = Y[idx].to(dev, non_blocking=True)
      bb = B[idx].to(dev, non_blocking=True) if B is not None else None
      obs = {"actor": xb}
      pred = policy.forward(obs, use_latch=False)
      with torch.no_grad():
        base = bb if bb is not None else policy.base_action(obs, use_latch=False)
      loss_bc = torch.nn.functional.smooth_l1_loss(pred, yb)
      loss_res = (pred - base).pow(2).mean()
      loss_base = torch.nn.functional.smooth_l1_loss(pred, base)
      loss = loss_bc + cfg.residual_l2 * loss_res + cfg.base_bc_coef * loss_base
      optim.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(policy.residual.parameters(), 1.0)
      optim.step()
      last = float(loss_bc.detach().cpu())

    if (epoch + 1) % 5 == 0 or epoch == 0 or epoch + 1 == cfg.epochs:
      msg = f"[INFO] epoch {epoch + 1}/{cfg.epochs} lr={lr:.1e} train_huber={last:.5f}"
      if n_val > 0:
        policy.eval()
        losses = []
        with torch.inference_mode():
          for start in range(0, val_idx.numel(), cfg.batch_size):
            idx = val_idx[start : start + cfg.batch_size]
            xb = X[idx].to(dev, non_blocking=True)
            yb = Y[idx].to(dev, non_blocking=True)
            pred = policy.forward({"actor": xb}, use_latch=False)
            losses.append(torch.nn.functional.smooth_l1_loss(pred, yb).detach().cpu())
        msg += f" val_huber={float(torch.stack(losses).mean()):.5f}"
      print(msg, flush=True)

  os.makedirs(os.path.dirname(cfg.out), exist_ok=True)
  saved = {
    "moe6_residual": True,
    "base_moe6": bundle,
    "policy_state_dict": policy.state_dict(),
    "hidden_dims": (512, 256, 128),
    "activation": "elu",
    "residual_scale": cfg.residual_scale,
    "source_data": tuple(cfg.data),
  }
  torch.save(saved, cfg.out)
  print(f"[INFO] saved MoE6 residual repaired keeper to {cfg.out}", flush=True)
  env.close()


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="distill_moe6_residual_repairs"))
