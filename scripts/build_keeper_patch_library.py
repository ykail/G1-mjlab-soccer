"""Bundle repair-oracle library shards into patch-library checkpoints."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Cfg:
  library_data: tuple[str, ...]
  base_checkpoint: str = "logs/keeper_targeted_repair/distilled/targeted_moe6_residual_scale0p01_bc0p6.pt"
  out_dir: str = "logs/keeper_patch_library/checkpoints"
  thresholds: tuple[float, ...] = (0.35, 0.5, 0.75, 1.0)
  pos_scale: float = 0.05
  vel_scale: float = 0.15
  residual_scales: tuple[float, ...] = (0.5, 0.75, 1.0)


def _load_library(paths: tuple[str, ...]) -> dict:
  parts = []
  for path in paths:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if data.get("residual_seq", torch.empty(0)).numel() == 0:
      continue
    parts.append(data)
  if not parts:
    raise RuntimeError("no non-empty library shards")
  return {
    "start": torch.cat([p["start"] for p in parts]),
    "vel": torch.cat([p["vel"] for p in parts]),
    "region": torch.cat([p["region"] for p in parts]).long(),
    "residual_seq": torch.cat([p["residual_seq"] for p in parts]),
  }


def main(cfg: Cfg) -> None:
  library = _load_library(cfg.library_data)
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  rows = []
  for threshold in cfg.thresholds:
    for residual_scale in cfg.residual_scales:
      name = f"patch_thr{threshold:g}_scale{residual_scale:g}".replace(".", "p")
      ckpt = {
        "keeper_patch_library": True,
        "base_checkpoint": cfg.base_checkpoint,
        "library": library,
        "threshold": threshold,
        "pos_scale": cfg.pos_scale,
        "vel_scale": cfg.vel_scale,
        "residual_scale": residual_scale,
      }
      path = out_dir / f"{name}.pt"
      torch.save(ckpt, path)
      rows.append({"name": name, "checkpoint": str(path)})
      print(f"[INFO] wrote {path}", flush=True)
  manifest = out_dir / "manifest.json"
  manifest.write_text(json.dumps(rows, indent=2) + "\n")
  print(f"[INFO] library entries={library['start'].shape[0]}", flush=True)
  print(f"[INFO] wrote {manifest}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="build_keeper_patch_library"))
