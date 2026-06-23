"""8-hour, 4-GPU keeper repair pipeline.

This is intentionally not another PPO sweep.  It uses the MoE6 keeper as a
frozen base, searches repair actions with CEM on hard scenarios, distills those
repairs into a small residual head, then evaluates multiple residual strengths.
Every stage is resumable and logs to disk.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import tyro


_REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Cfg:
  base: str = "checkpoints/keeper_93_moe6.pt"
  out_root: str = "logs/keeper_big_repair"
  devices: tuple[int, ...] = (0, 1, 2, 3)
  hours: float = 8.0
  prove: bool = True
  collect: bool = True
  distill: bool = True
  eval: bool = True
  regions: tuple[int, ...] = (1, 2, 3, 5)
  region_weights: tuple[float, ...] = (1.2, 1.4, 2.0, 1.2)
  G: int = 16
  P: int = 48
  iters: int = 7
  elites: int = 6
  knots: int = 12
  knot_span: int = 80
  release_steps: int = 20
  collect_batches_per_shard: int = 4
  max_shards: int = 999
  prove_min_union_gain: float = 0.02
  continue_on_bad_prove: bool = False
  distill_epochs: int = 70
  batch_size: int = 32768
  residual_scales: tuple[float, ...] = (0.08, 0.12, 0.18)
  base_bc_coefs: tuple[float, ...] = (0.02, 0.05)
  official_trials_per_seed: int = 50
  seed: int = 2810


def _env(gpu: int) -> dict[str, str]:
  env = os.environ.copy()
  env["CUDA_VISIBLE_DEVICES"] = str(gpu)
  env["MUJOCO_EGL_DEVICE_ID"] = "0"
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
  return env


def _run(label: str, cmd: list[str], log_path: Path, gpu: int | None = None, check: bool = True) -> int:
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  env = _env(gpu) if gpu is not None else os.environ.copy()
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")
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
  if check and proc.returncode != 0:
    print(log_path.read_text(errors="replace")[-6000:], flush=True)
    raise subprocess.CalledProcessError(proc.returncode, cmd)
  return proc.returncode


def _popen(label: str, cmd: list[str], log_path: Path, gpu: int):
  log_path.parent.mkdir(parents=True, exist_ok=True)
  print(f"\n[{label}] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
  print(f"[{label}] log -> {log_path}", flush=True)
  log = open(log_path, "w")
  log.write("[CMD] " + " ".join(shlex.quote(part) for part in cmd) + "\n")
  log.flush()
  proc = subprocess.Popen(
    cmd,
    cwd=_REPO_ROOT,
    env=_env(gpu),
    stdout=log,
    stderr=subprocess.STDOUT,
  )
  return proc, log


def _region_args(cfg: Cfg) -> list[str]:
  args = ["--regions", *[str(r) for r in cfg.regions]]
  if cfg.region_weights:
    args.extend(["--region-weights", *[str(w) for w in cfg.region_weights]])
  return args


def _repair_cmd(cfg: Cfg, mode: str, out: str, seed: int, batches: int) -> list[str]:
  cmd = [
    sys.executable,
    "scripts/repair_oracle.py",
    "--checkpoint",
    cfg.base,
    "--mode",
    mode,
    *_region_args(cfg),
    "--G",
    str(cfg.G),
    "--P",
    str(cfg.P),
    "--iters",
    str(cfg.iters),
    "--elites",
    str(cfg.elites),
    "--knots",
    str(cfg.knots),
    "--knot-span",
    str(cfg.knot_span),
    "--release-steps",
    str(cfg.release_steps),
    "--batches",
    str(batches),
    "--seed",
    str(seed),
    "--w-stable",
    "20.0",
    "--w-final-upright",
    "20.0",
    "--collect-pre-steps",
    "35",
    "--collect-post-steps",
    "12",
    "--device",
    "cuda:0",
  ]
  if out:
    cmd.extend(["--out", out])
  return cmd


def _existing_shards(out_root: Path) -> list[str]:
  return sorted(str(path) for path in (out_root / "repairs").glob("repairs_shard*.pt"))


def _read_prove_result(log_path: Path) -> tuple[float, float, float] | None:
  """Return final (base, repaired, base_or_repair) rates in [0,1]."""
  if not log_path.exists():
    return None
  text = log_path.read_text(errors="replace")
  matches = re.findall(
    r"base\s+([0-9.]+)%\s+->\s+repaired\s+([0-9.]+)%\s+;\s+base_or_repair\s+([0-9.]+)%",
    text,
  )
  if not matches:
    return None
  base, repaired, union = matches[-1]
  return float(base) / 100.0, float(repaired) / 100.0, float(union) / 100.0


def _collect(cfg: Cfg, out_root: Path, deadline: float) -> list[str]:
  repair_dir = out_root / "repairs"
  repair_dir.mkdir(parents=True, exist_ok=True)
  shards = _existing_shards(out_root)
  shard_idx = len(shards)
  running: list[tuple[int, int, Path, subprocess.Popen, object]] = []

  while time.monotonic() < deadline and shard_idx < cfg.max_shards:
    busy = {gpu for _, gpu, _, _, _ in running}
    free = [gpu for gpu in cfg.devices if gpu not in busy]
    while free and time.monotonic() < deadline and shard_idx < cfg.max_shards:
      gpu = free.pop(0)
      out = repair_dir / f"repairs_shard{shard_idx:03d}.pt"
      log = out_root / "logs" / f"collect_shard{shard_idx:03d}_gpu{gpu}.log"
      cmd = _repair_cmd(
        cfg,
        "collect",
        str(out),
        cfg.seed + 1000 + shard_idx,
        cfg.collect_batches_per_shard,
      )
      proc, handle = _popen(f"COLLECT {shard_idx}", cmd, log, gpu)
      running.append((shard_idx, gpu, out, proc, handle))
      shard_idx += 1

    time.sleep(10.0)
    still = []
    for idx, gpu, out, proc, handle in running:
      code = proc.poll()
      if code is None:
        still.append((idx, gpu, out, proc, handle))
        continue
      handle.close()
      if code == 0 and out.exists():
        print(f"[COLLECT] shard {idx} done: {out}", flush=True)
        shards.append(str(out))
      else:
        print(f"[COLLECT] shard {idx} failed with code {code}; see {out_root / 'logs'}", flush=True)
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
  return sorted(set(shards))


def _distill_and_eval(cfg: Cfg, out_root: Path, shards: list[str]) -> list[dict]:
  if not shards:
    raise RuntimeError("no repair shards available for distillation")
  results = []
  eval_gpu = cfg.devices[0]
  for scale in cfg.residual_scales:
    for bc_coef in cfg.base_bc_coefs:
      name = f"scale{scale:g}_bc{bc_coef:g}".replace(".", "p")
      ckpt = out_root / "distilled" / f"moe6_residual_{name}.pt"
      cmd = [
        sys.executable,
        "scripts/distill_moe6_residual_repairs.py",
        "--data",
        *shards,
        "--base",
        cfg.base,
        "--out",
        str(ckpt),
        "--epochs",
        str(cfg.distill_epochs),
        "--batch-size",
        str(cfg.batch_size),
        "--residual-scale",
        str(scale),
        "--base-bc-coef",
        str(bc_coef),
        "--device",
        "cuda:0",
      ]
      _run("DISTILL", cmd, out_root / "logs" / f"distill_{name}.log", gpu=eval_gpu)
      if not cfg.eval:
        results.append({"name": name, "checkpoint": str(ckpt), "score": -1.0})
        continue
      eval_json = out_root / "eval" / f"eval_{name}.json"
      eval_cmd = [
        sys.executable,
        "scripts/eval_goalkeeper_official_seeds.py",
        "--checkpoint",
        str(ckpt),
        "--trials-per-seed",
        str(cfg.official_trials_per_seed),
        "--parallel-seeds",
        "--seed-gpus",
        *[str(g) for g in cfg.devices[:3]],
        "--out",
        str(eval_json),
      ]
      code = _run("EVAL", eval_cmd, out_root / "logs" / f"eval_{name}.log", check=False)
      score = -1.0
      if code == 0 and eval_json.exists():
        data = json.loads(eval_json.read_text())
        score = float(data.get("mean_block_rate", -1.0))
      results.append({"name": name, "checkpoint": str(ckpt), "score": score, "eval": str(eval_json)})
  return results


def main(cfg: Cfg) -> None:
  start = time.monotonic()
  deadline = start + max(0.25, cfg.hours) * 3600.0
  out_root = Path(cfg.out_root)
  out_root.mkdir(parents=True, exist_ok=True)
  (out_root / "logs").mkdir(exist_ok=True)
  print(f"[INFO] base={cfg.base}", flush=True)
  print(f"[INFO] out_root={out_root}", flush=True)
  print(f"[INFO] devices={cfg.devices} hours={cfg.hours}", flush=True)

  if cfg.prove:
    cmd = _repair_cmd(cfg, "prove", "", cfg.seed, batches=2)
    prove_log = out_root / "logs" / "prove.log"
    _run("PROVE", cmd, prove_log, gpu=cfg.devices[0], check=False)
    prove_result = _read_prove_result(prove_log)
    if prove_result is None:
      msg = "[PROVE] could not parse proof result; refusing to collect blindly"
      if not cfg.continue_on_bad_prove and cfg.collect:
        print(msg, flush=True)
        return
      print(msg + " because --continue-on-bad-prove was set", flush=True)
    else:
      base_rate, repaired_rate, union_rate = prove_result
      gain = union_rate - base_rate
      print(
        f"[PROVE] base={100*base_rate:.1f}% repaired={100*repaired_rate:.1f}% "
        f"base_or_repair={100*union_rate:.1f}% gain={100*gain:+.1f}%",
        flush=True,
      )
      if cfg.collect and gain < cfg.prove_min_union_gain and not cfg.continue_on_bad_prove:
        print(
          "[PROVE] aborting before collection: proof did not show enough "
          f"base_or_repair gain (need {100*cfg.prove_min_union_gain:.1f}%).",
          flush=True,
        )
        print("[PROVE] rerun with --continue-on-bad-prove only if you intentionally want to burn time.", flush=True)
        return

  shards = _existing_shards(out_root)
  if cfg.collect:
    collect_deadline = deadline - 1.5 * 3600.0 if cfg.distill else deadline
    collect_deadline = max(time.monotonic() + 60.0, collect_deadline)
    shards = _collect(cfg, out_root, collect_deadline)
  print(f"[INFO] repair shards={len(shards)}", flush=True)
  for shard in shards:
    print(f"  {shard}", flush=True)

  results = []
  if cfg.distill:
    results = _distill_and_eval(cfg, out_root, shards)
    results.sort(key=lambda row: row["score"], reverse=True)
    summary = out_root / "summary.json"
    summary.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[INFO] wrote {summary}", flush=True)
    if results:
      best = results[0]
      print(
        f"[BEST] {best['name']} score={100.0 * best['score']:.2f}% "
        f"checkpoint={best['checkpoint']}",
        flush=True,
      )


if __name__ == "__main__":
  main(tyro.cli(Cfg, prog="run_keeper_big_repair"))
