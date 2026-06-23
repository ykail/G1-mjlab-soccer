"""Select a hybrid MoE6 bundle from base and tuned experts.

Failure-replay training can improve one region while hurting another.  This
script builds all requested base/tuned expert subsets, evaluates each candidate
with ``eval_stable_moe6.py``, and bundles the best one.  It is intended as a
coarse filter before running the slower official fixed-seed evaluation.
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]
_BLOCK_RE = re.compile(r"MoE6 block:\s+\d+/\d+\s+=\s+([0-9.]+)%")


@dataclass
class Cfg:
  base_dir: str = "logs/keeper_moe6_failure_replay/base_experts"
  tuned_dir: str = "logs/keeper_moe6_failure_replay/experts"
  out_root: str = "logs/keeper_moe6_failure_replay/hybrid_select"
  final: str = "logs/keeper_moe6_failure_replay/keeper_93_hybrid_selected.pt"
  prefix: str = "stable_sr"
  regions: tuple[int, ...] = (1, 2, 3, 5)
  mirror_maps: tuple[str, ...] = ("meta", "none")
  num_envs: int = 512
  batches: int = 24
  steps: int = 149
  seed: int = 2810
  max_candidates: int = 0
  device: str = "cuda:0"


def _load_meta(base_dir: str) -> dict:
  path = Path(base_dir) / "moe6_meta.json"
  if not path.exists():
    return {
      "z_low": 0.85,
      "z_up": 1.35,
      "vz_low": -99.0,
      "latch_hi": 5.0,
      "land_x": 0.0,
      "mirror_map": "1:0,3:2",
    }
  return json.loads(path.read_text())


def _mirror_value(raw: str, meta: dict) -> str:
  if raw == "meta":
    return str(meta.get("mirror_map", ""))
  if raw in ("none", "off", "empty"):
    return ""
  return raw


def _expert_path(directory: str | Path, prefix: str, region: int) -> Path:
  return Path(directory) / f"{prefix}{region}.pt"


def _check_inputs(cfg: Cfg) -> None:
  for region in range(6):
    path = _expert_path(cfg.base_dir, cfg.prefix, region)
    if not path.exists():
      raise FileNotFoundError(path)
  for region in cfg.regions:
    if region < 0 or region > 5:
      raise ValueError(f"regions must be in [0, 5], got {region}")
    path = _expert_path(cfg.tuned_dir, cfg.prefix, region)
    if not path.exists():
      raise FileNotFoundError(path)


def _candidate_subsets(regions: tuple[int, ...]) -> list[tuple[int, ...]]:
  out = []
  unique = tuple(dict.fromkeys(regions))
  for size in range(len(unique) + 1):
    out.extend(itertools.combinations(unique, size))
  return out


def _candidate_name(regions: tuple[int, ...], mirror_label: str) -> str:
  body = "base" if not regions else "tuned_" + "_".join(f"r{r}" for r in regions)
  safe_mirror = mirror_label.replace(":", "-").replace(",", "_") or "none"
  return f"{body}__mirror_{safe_mirror}"


def _build_candidate(cfg: Cfg, regions: tuple[int, ...], name: str) -> Path:
  out_dir = Path(cfg.out_root) / "candidates" / name
  out_dir.mkdir(parents=True, exist_ok=True)
  for region in range(6):
    shutil.copy2(
      _expert_path(cfg.base_dir, cfg.prefix, region),
      _expert_path(out_dir, cfg.prefix, region),
    )
  for region in regions:
    shutil.copy2(
      _expert_path(cfg.tuned_dir, cfg.prefix, region),
      _expert_path(out_dir, cfg.prefix, region),
    )
  return out_dir


def _run_eval(cfg: Cfg, meta: dict, expert_dir: Path, mirror_map: str, log_path: Path) -> float:
  env = os.environ.copy()
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
  env.setdefault("WANDB_MODE", "disabled")
  cmd = [
    sys.executable,
    "scripts/eval_stable_moe6.py",
    "--expert-dir",
    str(expert_dir),
    "--prefix",
    cfg.prefix,
    "--mirror-map",
    mirror_map,
    "--num-envs",
    str(cfg.num_envs),
    "--batches",
    str(cfg.batches),
    "--steps",
    str(cfg.steps),
    "--seed",
    str(cfg.seed),
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
    "--device",
    cfg.device,
  ]
  log_path.parent.mkdir(parents=True, exist_ok=True)
  with log_path.open("w") as log:
    log.write("[CMD] " + " ".join(shlex.quote(part) for part in cmd) + "\n")
    proc = subprocess.run(
      cmd,
      cwd=_REPO_ROOT,
      env=env,
      stdout=log,
      stderr=subprocess.STDOUT,
      check=False,
    )
  text = log_path.read_text(errors="replace")
  if proc.returncode != 0:
    print(f"[WARN] eval failed for {expert_dir.name}; see {log_path}", flush=True)
    return -1.0
  match = _BLOCK_RE.search(text)
  if not match:
    print(f"[WARN] could not parse block rate for {expert_dir.name}; see {log_path}", flush=True)
    return -1.0
  return float(match.group(1)) / 100.0


def _bundle(cfg: Cfg, meta: dict, expert_dir: Path, mirror_map: str) -> None:
  cmd = [
    sys.executable,
    "scripts/bundle_moe6.py",
    "--expert-dir",
    str(expert_dir),
    "--prefix",
    cfg.prefix,
    "--out",
    cfg.final,
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
    mirror_map,
  ]
  print("[BUNDLE] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
  subprocess.run(cmd, cwd=_REPO_ROOT, check=True)


def main(cfg: Cfg) -> None:
  _check_inputs(cfg)
  meta = _load_meta(cfg.base_dir)
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  rows = []
  best: dict | None = None

  candidate_count = 0
  for subset in _candidate_subsets(cfg.regions):
    for mirror_label in cfg.mirror_maps:
      if cfg.max_candidates > 0 and candidate_count >= cfg.max_candidates:
        break
      mirror_map = _mirror_value(mirror_label, meta)
      name = _candidate_name(subset, mirror_label)
      expert_dir = _build_candidate(cfg, subset, name)
      log_path = out_root / "logs" / f"{name}.log"
      rate = _run_eval(cfg, meta, expert_dir, mirror_map, log_path)
      row = {
        "name": name,
        "regions": " ".join(str(r) for r in subset),
        "mirror_label": mirror_label,
        "mirror_map": mirror_map,
        "block_rate": rate,
        "expert_dir": str(expert_dir),
        "log": str(log_path),
      }
      rows.append(row)
      candidate_count += 1
      print(
        f"[RESULT] {name}: block={100.0 * rate:.2f}% "
        f"regions={row['regions'] or '<base>'} mirror={mirror_label}",
        flush=True,
      )
      if rate >= 0.0 and (best is None or rate > best["block_rate"]):
        best = row
    else:
      continue
    break

  rows.sort(key=lambda row: row["block_rate"], reverse=True)
  csv_path = out_root / "hybrid_results.csv"
  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
  print(f"[INFO] wrote {csv_path}", flush=True)

  if best is None:
    raise RuntimeError("no candidate evaluated successfully")
  print(
    f"[BEST] {best['name']} block={100.0 * best['block_rate']:.2f}% "
    f"mirror={best['mirror_label']} regions={best['regions'] or '<base>'}",
    flush=True,
  )
  _bundle(cfg, meta, Path(best["expert_dir"]), str(best["mirror_map"]))


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="select_moe6_hybrid"))
