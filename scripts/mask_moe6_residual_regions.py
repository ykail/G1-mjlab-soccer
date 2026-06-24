"""Create region-masked variants of a MoE6 residual checkpoint."""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro


@dataclass
class Cfg:
  checkpoint: str
  out_dir: str = "logs/keeper_region_masks"
  regions: tuple[int, ...] = (1, 2, 3, 5)
  include_empty: bool = False


def _name(regions: tuple[int, ...]) -> str:
  return "base_only" if not regions else "r" + "_".join(str(r) for r in regions)


def main(cfg: Cfg) -> None:
  ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
  if not isinstance(ckpt, dict) or not ckpt.get("moe6_residual"):
    raise ValueError(f"not a MoE6 residual checkpoint: {cfg.checkpoint}")
  out_dir = Path(cfg.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  unique = tuple(dict.fromkeys(cfg.regions))
  start = 0 if cfg.include_empty else 1
  written = []
  for size in range(start, len(unique) + 1):
    for subset in itertools.combinations(unique, size):
      item = dict(ckpt)
      item["residual_regions"] = tuple(int(r) for r in subset)
      state = dict(item["policy_state_dict"])
      state["_residual_regions"] = torch.tensor(item["residual_regions"], dtype=torch.long)
      item["policy_state_dict"] = state
      path = out_dir / f"{Path(cfg.checkpoint).stem}_{_name(item['residual_regions'])}.pt"
      torch.save(item, path)
      written.append(str(path))
      print(f"[INFO] wrote {path} residual_regions={item['residual_regions']}", flush=True)
  manifest = out_dir / "manifest.txt"
  manifest.write_text("\n".join(written) + "\n")
  print(f"[INFO] wrote {manifest}", flush=True)


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="mask_moe6_residual_regions"))
