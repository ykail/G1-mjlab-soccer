"""Extract six experts from a bundled goalkeeper MoE6 checkpoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Cfg:
  checkpoint: str = "checkpoints/keeper_93.pt"
  out_dir: str = "logs/keeper_moe6_failure_replay/base_experts"
  prefix: str = "stable_sr"


def main(cfg: Cfg) -> None:
  bundle = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
  if not isinstance(bundle, dict) or not bundle.get("moe6"):
    raise ValueError(f"checkpoint is not a MoE6 bundle: {cfg.checkpoint}")
  experts = bundle.get("sr")
  if not isinstance(experts, (list, tuple)) or len(experts) != 6:
    raise ValueError("MoE6 bundle must contain key 'sr' with six experts")

  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  for idx, expert in enumerate(experts):
    path = out_dir / f"{cfg.prefix}{idx}.pt"
    torch.save(expert, path)
    print(f"[INFO] wrote expert {idx}: {path}", flush=True)

  meta = {
    "source": cfg.checkpoint,
    "z_low": float(bundle.get("z_low", 0.85)),
    "z_up": float(bundle.get("z_up", 1.35)),
    "vz_low": float(bundle.get("vz_low", -99.0)),
    "latch_hi": float(bundle.get("latch_hi", 5.0)),
    "land_x": float(bundle.get("land_x", 0.0)),
    "mirror_map": str(bundle.get("mirror_map", "")),
  }
  meta_path = out_dir / "moe6_meta.json"
  meta_path.write_text(json.dumps(meta, indent=2) + "\n")
  print(f"[INFO] wrote metadata: {meta_path}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="extract_moe6_experts"))
