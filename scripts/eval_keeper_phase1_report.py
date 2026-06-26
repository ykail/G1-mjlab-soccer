"""Run Phase-1 goalkeeper evaluation and write a Markdown report.

This is a thin wrapper around ``scripts/eval_goalkeeper_official_seeds.py``.
By default it evaluates the selected keeper for 50 trials on each of the three
fixed seeds, then reports ``success_times / 150`` as the pooled block rate.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = (
  "logs/keeper_h20_autorepair_leftmid_big/distilled/"
  "moe6_residual_regions3_scale0p008_bc1.pt"
)


def _score(rate: float, threshold: float, points: float) -> float:
  if threshold >= 1.0:
    raise ValueError("--score-threshold must be < 1.0")
  frac = (rate - threshold) / (1.0 - threshold)
  return max(0.0, min(1.0, frac)) * points


def _pct(rate: float) -> str:
  return f"{100.0 * rate:.2f}%"


def _quote(cmd: list[str]) -> str:
  return " ".join(shlex.quote(part) for part in cmd)


def _resolve_path(path: str | Path) -> Path:
  path = Path(path)
  return path if path.is_absolute() else _REPO_ROOT / path


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
  out_dir = _resolve_path(args.out_dir)
  stem = Path(args.checkpoint).stem
  suffix = f"{len(args.seeds)}seeds_{args.trials_per_seed}each"
  json_path = _resolve_path(args.json) if args.json else out_dir / f"{stem}_{suffix}.json"
  log_path = _resolve_path(args.log) if args.log else out_dir / f"{stem}_{suffix}.log"
  report_path = (
    _resolve_path(args.report)
    if args.report
    else out_dir / f"{stem}_{suffix}.md"
  )
  return json_path, log_path, report_path


def _eval_command(args: argparse.Namespace, json_path: Path, log_path: Path) -> list[str]:
  cmd = [
    args.python_bin,
    "scripts/eval_goalkeeper_official_seeds.py",
    "--checkpoint",
    args.checkpoint,
    "--seeds",
    *[str(seed) for seed in args.seeds],
    "--trials-per-seed",
    str(args.trials_per_seed),
    "--max-steps",
    str(args.max_steps),
    "--score-points",
    str(args.score_points),
    "--score-threshold",
    str(args.score_threshold),
    "--out",
    str(json_path),
    "--log",
    str(log_path),
  ]
  if args.task_id:
    cmd.extend(["--task-id", args.task_id])
  if args.parallel_seeds:
    cmd.append("--parallel-seeds")
  if args.seed_gpus:
    cmd.extend(["--seed-gpus", *[str(gpu) for gpu in args.seed_gpus]])
  if args.device:
    cmd.extend(["--device", args.device])
  return cmd


def _run_eval(args: argparse.Namespace, json_path: Path, log_path: Path) -> list[str]:
  cmd = _eval_command(args, json_path, log_path)
  json_path.parent.mkdir(parents=True, exist_ok=True)
  log_path.parent.mkdir(parents=True, exist_ok=True)

  env = os.environ.copy()
  env.setdefault("PYOPENGL_PLATFORM", "egl")
  env.setdefault("MUJOCO_GL", "egl")
  env.setdefault("WANDB_MODE", "disabled")

  print("[RUN] " + _quote(cmd), flush=True)
  proc = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, check=False)
  if proc.returncode != 0:
    tail = ""
    if log_path.exists():
      tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-80:])
    raise RuntimeError(
      f"evaluation failed with code {proc.returncode}; log={log_path}\n{tail}"
    )
  return cmd


def _load_summary(json_path: Path) -> dict:
  if not json_path.exists():
    raise FileNotFoundError(f"missing eval JSON: {json_path}")
  return json.loads(json_path.read_text())


def _summarize(summary: dict, args: argparse.Namespace) -> dict:
  per_seed = list(summary.get("per_seed", []))
  total_success = sum(int(row.get("blocked", 0)) for row in per_seed)
  total_trials = sum(int(row.get("trials", 0)) for row in per_seed)
  pooled_rate = total_success / max(1, total_trials)
  mean_rate = float(summary.get("mean_block_rate", pooled_rate))
  return {
    "checkpoint": summary.get("checkpoint") or args.checkpoint,
    "seeds": summary.get("seeds") or list(args.seeds),
    "trials_per_seed": int(summary.get("trials_per_seed", args.trials_per_seed)),
    "total_success": total_success,
    "total_trials": total_trials,
    "pooled_rate": pooled_rate,
    "mean_rate": mean_rate,
    "score_from_pooled": _score(pooled_rate, args.score_threshold, args.score_points),
    "score_from_mean": float(summary.get("score", _score(mean_rate, args.score_threshold, args.score_points))),
    "per_seed": per_seed,
  }


def _write_report(
  result: dict,
  args: argparse.Namespace,
  report_path: Path,
  json_path: Path,
  log_path: Path,
  command: list[str] | None,
) -> None:
  generated = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
  command_text = _quote(command) if command else "(loaded existing JSON with --skip-eval)"

  lines = [
    "# Phase 1 Goalkeeper Evaluation",
    "",
    f"- Generated: {generated}",
    f"- Checkpoint: `{result['checkpoint']}`",
    "- Eval task: `Eval-Goalkeeper`",
    f"- Seeds: `{', '.join(str(seed) for seed in result['seeds'])}`",
    f"- Trials per seed: `{result['trials_per_seed']}`",
    f"- Total trials: `{result['total_trials']}`",
    f"- Max steps per trial: `{args.max_steps}`",
    "",
    "## Metric",
    "",
    (
      "A goalkeeper trial is counted as successful when the ball does not enter "
      "the goal frame before timeout. The reported block rate here is the pooled "
      "success count divided by total trials."
    ),
    "",
    "```text",
    "block_rate = success_times / total_trials",
    "phase1_score = max(0, (block_rate - 0.8) / 0.2) * 30",
    "```",
    "",
    "## Result",
    "",
    (
      f"- Block rate: `{result['total_success']}/{result['total_trials']} = "
      f"{_pct(result['pooled_rate'])}`"
    ),
    f"- Mean block rate across seeds: `{_pct(result['mean_rate'])}`",
    f"- Phase 1 GK score from pooled block rate: `{result['score_from_pooled']:.2f}/30.00`",
    f"- Evaluator score from mean seed rate: `{result['score_from_mean']:.2f}/30.00`",
    "",
    "## Per-Seed Breakdown",
    "",
    "| Seed | Successes | Trials | Block rate |",
    "| ---: | ---: | ---: | ---: |",
  ]
  for row in result["per_seed"]:
    rate = float(row.get("rate", 0.0))
    lines.append(
      f"| {int(row.get('seed', 0))} | {int(row.get('blocked', 0))} | "
      f"{int(row.get('trials', 0))} | {_pct(rate)} |"
    )

  lines.extend(
    [
      "",
      "## Reproduction Command",
      "",
      "```bash",
      command_text,
      "```",
      "",
      "## Artifacts",
      "",
      f"- Raw JSON: `{json_path}`",
      f"- Full log: `{log_path}`",
    ]
  )

  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", default=_DEFAULT_CHECKPOINT)
  parser.add_argument("--python-bin", default=sys.executable)
  parser.add_argument("--out-dir", default="logs/keeper_phase1_eval")
  parser.add_argument("--json", default="", help="Optional explicit JSON output path.")
  parser.add_argument("--log", default="", help="Optional explicit eval log path.")
  parser.add_argument("--report", default="", help="Optional explicit Markdown output path.")
  parser.add_argument("--seeds", nargs="+", type=int, default=[42, 2810, 202686])
  parser.add_argument("--trials-per-seed", type=int, default=50)
  parser.add_argument("--max-steps", type=int, default=150)
  parser.add_argument("--score-points", type=float, default=30.0)
  parser.add_argument("--score-threshold", type=float, default=0.8)
  parser.add_argument("--task-id", default="Eval-Goalkeeper")
  parser.add_argument("--device", default="")
  parser.add_argument("--parallel-seeds", action="store_true")
  parser.add_argument("--seed-gpus", nargs="*", type=int, default=[])
  parser.add_argument(
    "--skip-eval",
    action="store_true",
    help="Read an existing JSON file and regenerate only the Markdown report.",
  )
  args = parser.parse_args()
  if args.trials_per_seed <= 0:
    raise ValueError("--trials-per-seed must be positive")
  if args.seed_gpus:
    args.parallel_seeds = True
  return args


def main() -> None:
  args = parse_args()
  json_path, log_path, report_path = _default_paths(args)
  command = None if args.skip_eval else _run_eval(args, json_path, log_path)
  summary = _load_summary(json_path)
  result = _summarize(summary, args)
  _write_report(result, args, report_path, json_path, log_path, command)

  print("", flush=True)
  print(
    f"[RESULT] block_rate={result['total_success']}/{result['total_trials']} "
    f"= {_pct(result['pooled_rate'])}",
    flush=True,
  )
  print(f"[RESULT] phase1_gk_score={result['score_from_pooled']:.2f}/30.00", flush=True)
  print(f"[INFO] report={report_path}", flush=True)
  print(f"[INFO] json={json_path}", flush=True)
  print(f"[INFO] log={log_path}", flush=True)


if __name__ == "__main__":
  main()
