# Goalkeeper Failure-Residual

This branch is a conservative alternative to training the HIMPPO/AMP keeper from
scratch.  It keeps the strongest available keeper as a frozen base policy and
trains only a small ballistic residual on the exact ball trajectories where that
base still concedes.

## Why this should work

The failed HIMPPO/AMP run learned a new keeper from scratch.  That is hard
because a successful save depends on precise timing, coordinated diving, and a
sparse eval objective.  Extra GPU can make that experiment faster, but it does
not fix the optimization problem.

This branch changes the problem:

- The base policy already knows how to stand, dive, and recover.
- The residual starts at zero, so iteration 0 behaves exactly like the base.
- Failure replay trains on the 5-10% of ball trajectories that matter instead of
  mostly seeing balls the base already saves.
- The residual receives explicit ballistic features, so it can learn timing and
  reach corrections without relearning perception from scratch.
- Rollback-after-eval keeps the best actor if PPO starts to damage the policy.

The goal is not to discover a keeper from scratch; it is to patch known misses
while preserving the existing high-success behavior.

## New code

- `scripts/collect_goalkeeper_failures.py`
  collects official eval failures for any checkpoint that
  `scripts/eval_naive_goalkeeper.py` can load.  The CSV contains ball start,
  velocity, region, crossing, and final-state diagnostics.

- `scripts/launch_keeper_failure_residual.py`
  launches many independent failure-replay residual runs across GPUs.  This is
  the preferred way to use 4 H20 cards because the policy network is small and
  MuJoCo/env stepping dominates more than neural-network memory.

- `scripts/train_ballistic_residual.py`
  already supported `--failure-csv`; this branch records the failure-replay
  settings in saved checkpoints.

## H20 setup

```bash
git clone -b codex/keeper-failure-residual <YOUR_GITHUB_REPO_URL> G1-mjlab-soccer
cd G1-mjlab-soccer

conda activate unitree_rl_mjlab
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
```

If the branch already exists locally:

```bash
git fetch origin
git switch codex/keeper-failure-residual
git pull --ff-only
```

## Step 1: collect failures from the current best keeper

Use your best checkpoint as `--checkpoint`.  The collector reuses the normal eval
loader, so it can collect failures from native MLP, ballistic-residual, LSTM,
HIMPPO/AMP, or MoE checkpoints.  If your best result is the bundled repaired
policy, start with:

```bash
python scripts/collect_goalkeeper_failures.py \
  --checkpoint src/assets/soccer/weight/model_repaired_lyk.pt \
  --out-csv logs/keeper_failure_residual/failures_model_repaired_lyk.csv \
  --num-envs 4096 \
  --batches 32 \
  --steps 150 \
  --device cuda:0
```

This samples `4096 * 32 = 131072` official eval episodes.  A 93% keeper should
produce roughly several thousand failures, enough to replay hard cases.

If you want to collect from another checkpoint:

```bash
python scripts/collect_goalkeeper_failures.py \
  --checkpoint /path/to/your_best_keeper.pt \
  --out-csv logs/keeper_failure_residual/failures_best.csv \
  --num-envs 4096 \
  --batches 32 \
  --device cuda:0
```

## Step 2: train residual patches on 4 H20 cards

Run independent seeds and hyperparameters in parallel.  `--init` must be a native
MLP keeper checkpoint or an existing ballistic-residual checkpoint, because this
training path freezes that actor as the base.  MoE/HIMPPO/LSTM checkpoints are
fine for failure collection, but should not be passed as `--init` here.

```bash
python scripts/launch_keeper_failure_residual.py \
  --init src/assets/soccer/weight/model_repaired_lyk.pt \
  --failure-csv logs/keeper_failure_residual/failures_model_repaired_lyk.csv \
  --devices 0 1 2 3 \
  --num-envs 8192 \
  --blocks 80 \
  --block-iters 15 \
  --eval-resets 6 \
  --max-runs 16
```

Outputs:

- checkpoints:
  `logs/keeper_failure_residual/checkpoints/*.pt`
- training logs:
  `logs/keeper_failure_residual/train_logs/*.log`
- exact commands:
  `logs/keeper_failure_residual/train_logs/*.cmd`

For a faster smoke test:

```bash
python scripts/launch_keeper_failure_residual.py \
  --init src/assets/soccer/weight/model_repaired_lyk.pt \
  --failure-csv logs/keeper_failure_residual/failures_model_repaired_lyk.csv \
  --devices 0 1 2 3 \
  --num-envs 4096 \
  --blocks 12 \
  --block-iters 8 \
  --eval-resets 3 \
  --max-runs 4
```

## Step 3: monitor training

```bash
for f in logs/keeper_failure_residual/train_logs/*.log; do
  echo "===== $f ====="
  grep -E "\[EVAL\]|\[INFO\] saved" "$f" | tail -n 8
done
```

Useful signals:

- `block` should not fall below the init baseline.
- `stable` is stricter than block rate and is useful for avoiding falling saves.
- `*best* (saved)` means the checkpoint was improved and written.
- `rollback` means a PPO block damaged eval score and the actor was restored.

## Step 4: official multi-seed evaluation

Evaluate promising checkpoints with the fixed seed protocol:

```bash
python scripts/eval_goalkeeper_official_seeds.py \
  --checkpoint logs/keeper_failure_residual/checkpoints/<RUN>.pt \
  --trials-per-seed 50 \
  --parallel-seeds \
  --seed-gpus 0 1 2 \
  --out logs/keeper_failure_residual/eval_<RUN>.json
```

For a quick single-script check:

```bash
python scripts/eval_naive_goalkeeper.py \
  --headless \
  --num-trials 500 \
  --checkpoint logs/keeper_failure_residual/checkpoints/<RUN>.pt \
  --device cuda:0
```

## If the first sweep does not improve

Try these in order:

1. Increase replay pressure:

```bash
python scripts/launch_keeper_failure_residual.py \
  --init src/assets/soccer/weight/model_repaired_lyk.pt \
  --failure-csv logs/keeper_failure_residual/failures_model_repaired_lyk.csv \
  --devices 0 1 2 3 \
  --failure-replay-ratios 0.6 0.75 \
  --residual-scale-values 0.12 0.18 \
  --lr-values 2e-5 3e-5 \
  --std-values 0.02 0.025 \
  --max-runs 16
```

2. Use a stronger init checkpoint if you have one:

```bash
python scripts/launch_keeper_failure_residual.py \
  --init /path/to/current_93_percent_keeper.pt \
  --failure-csv logs/keeper_failure_residual/failures_best.csv \
  --devices 0 1 2 3 \
  --max-runs 16
```

3. Collect more failures from the current best residual and run another round:

```bash
python scripts/collect_goalkeeper_failures.py \
  --checkpoint logs/keeper_failure_residual/checkpoints/<BEST>.pt \
  --out-csv logs/keeper_failure_residual/failures_round2.csv \
  --num-envs 4096 \
  --batches 32 \
  --device cuda:0

python scripts/launch_keeper_failure_residual.py \
  --init logs/keeper_failure_residual/checkpoints/<BEST>.pt \
  --failure-csv logs/keeper_failure_residual/failures_round2.csv \
  --devices 0 1 2 3 \
  --max-runs 16
```

## Expected outcome

This route should be judged by official eval, not by training reward.  It is
successful if it keeps the base policy's easy saves and improves the hard-case
tail.  Because the residual is bounded and the base is frozen, a bad run should
usually fail by doing nothing useful rather than destroying the keeper.

## MoE6 Checkpoints

PR #10's strong keeper is a MoE6 bundle, not a single actor.  Do not pass that
bundle directly to `launch_keeper_failure_residual.py --init`; that trainer is
for native MLP or ballistic-residual checkpoints.  For MoE6, keep the successful
architecture intact and continue training the six specialists separately.

First upload the 93% checkpoint:

```bash
mkdir -p checkpoints
scp /path/to/goalkeeper_moe6_hard3_default.pt \
  root@<server_ip>:/data/G1-mjlab-soccer/checkpoints/keeper_93_moe6.pt
```

On the H20 server:

```bash
cd /data/G1-mjlab-soccer
git fetch ykail
git switch codex/keeper-failure-residual
git pull --ff-only

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export WANDB_MODE=disabled
```

Collect failures from the MoE6 bundle:

```bash
python scripts/collect_goalkeeper_failures.py \
  --checkpoint checkpoints/keeper_93_moe6.pt \
  --out-csv logs/keeper_moe6_failure_replay/failures_keeper_93.csv \
  --num-envs 4096 \
  --batches 64 \
  --device cuda:0
```

For MoE6 checkpoints this CSV records both `true_region` and `used_region`.
The latter is the expert selected by the MoE gate; specialist replay uses
`used_region` when available.

Extract the six experts:

```bash
python scripts/extract_moe6_experts.py \
  --checkpoint checkpoints/keeper_93_moe6.pt \
  --out-dir logs/keeper_moe6_failure_replay/base_experts
```

Continue-train the specialists with region-specific failure replay:

```bash
python scripts/launch_moe6_failure_replay.py \
  --expert-dir logs/keeper_moe6_failure_replay/base_experts \
  --failure-csv logs/keeper_moe6_failure_replay/failures_keeper_93.csv \
  --out-dir logs/keeper_moe6_failure_replay/experts \
  --bundle-out logs/keeper_moe6_failure_replay/keeper_93_failure_replay_moe6.pt \
  --devices 0 1 2 3 \
  --num-envs 8192 \
  --blocks 60 \
  --block-iters 12 \
  --eval-resets 6
```

By default this launcher preserves each expert checkpoint's own residual scale.
Only pass `--residual-scale <value>` if you intentionally want to override that
metadata; overriding it can change the base MoE before training even starts.

Evaluate the bundled result:

```bash
python scripts/eval_goalkeeper_official_seeds.py \
  --checkpoint logs/keeper_moe6_failure_replay/keeper_93_failure_replay_moe6.pt \
  --trials-per-seed 50 \
  --parallel-seeds \
  --seed-gpus 0 1 2 \
  --out logs/keeper_moe6_failure_replay/eval_failure_replay_moe6.json
```

If region 3 still dominates failures, run a focused sweep:

```bash
python scripts/launch_moe6_failure_replay.py \
  --expert-dir logs/keeper_moe6_failure_replay/base_experts \
  --failure-csv logs/keeper_moe6_failure_replay/failures_keeper_93.csv \
  --out-dir logs/keeper_moe6_failure_replay/experts_region3 \
  --bundle-out logs/keeper_moe6_failure_replay/keeper_93_region3_moe6.pt \
  --devices 0 \
  --regions 3 \
  --num-envs 8192 \
  --failure-replay-ratio 0.75 \
  --lr 1.5e-5 \
  --std 0.018 \
  --residual-scale 0.14 \
  --blocks 80
```

If a failure-replay MoE6 bundle evaluates below the original 93% checkpoint,
stop training and run hybrid selection.  This checks whether any tuned expert is
useful by itself while leaving the other regions at the original base:

```bash
python scripts/select_moe6_hybrid.py \
  --base-dir logs/keeper_moe6_failure_replay/base_experts \
  --tuned-dir logs/keeper_moe6_failure_replay/experts \
  --out-root logs/keeper_moe6_failure_replay/hybrid_select \
  --final logs/keeper_moe6_failure_replay/keeper_93_hybrid_selected.pt \
  --regions 1 2 3 5 \
  --num-envs 512 \
  --batches 24 \
  --device cuda:0
```

Then run official eval on the selected hybrid:

```bash
python scripts/eval_goalkeeper_official_seeds.py \
  --checkpoint logs/keeper_moe6_failure_replay/keeper_93_hybrid_selected.pt \
  --trials-per-seed 50 \
  --parallel-seeds \
  --seed-gpus 0 1 2 \
  --out logs/keeper_moe6_failure_replay/eval_hybrid_selected.json
```

## Big Repair Run

When gate sweep and PPO-style failure replay do not clearly beat the 93% MoE6
base, use the repair-oracle pipeline.  This is the largest experiment in this
branch and is designed for an 8-hour, 4-GPU H20 window.

The pipeline:

1. runs a small CEM proof on hard regions;
2. stops immediately if the proof does not show a useful conservative gain;
3. collects independent CEM repair shards on all GPUs until the time budget is
   nearly exhausted;
4. distills the repaired actions into a frozen-base MoE6 residual policy;
5. evaluates several residual strengths and writes a ranked summary.

The proof log prints three rates:

- `base`: the frozen MoE6 base on this hard, biased proof distribution.  It is
  not the official uniform-seed 93% score.
- `repaired`: always use the CEM repair action.  This can be lower than base
  because some repair actions are intentionally aggressive.
- `base_or_repair`: keep base successes and only count repair wins on base
  failures.  This is the important upper-bound number for the conservative
  residual strategy.  The automation only continues collection if this beats
  `base` by at least `--prove-min-union-gain` (default: 2 percentage points).

Run:

```bash
python scripts/run_keeper_big_repair.py \
  --base checkpoints/keeper_93_moe6.pt \
  --out-root logs/keeper_big_repair \
  --devices 0 1 2 3 \
  --hours 8 \
  --regions 1 2 3 5 \
  --region-weights 1.2 1.4 2.0 1.2 \
  --G 16 \
  --P 48 \
  --iters 7 \
  --collect-batches-per-shard 4 \
  --prove-min-union-gain 0.02 \
  --distill-epochs 70 \
  --official-trials-per-seed 50
```

Outputs:

- repair shards: `logs/keeper_big_repair/repairs/repairs_shard*.pt`
- distilled checkpoints: `logs/keeper_big_repair/distilled/*.pt`
- eval JSON: `logs/keeper_big_repair/eval/*.json`
- ranked summary: `logs/keeper_big_repair/summary.json`

This experiment should be treated as an A/B candidate generator.  If the best
checkpoint in `summary.json` is not above the original MoE6 base under the same
official protocol, discard it and keep the base.
