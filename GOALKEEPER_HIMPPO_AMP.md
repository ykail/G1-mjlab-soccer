# Goalkeeper HIMPPO + AMP Training Path

This branch adds a new goalkeeper training direction:

```text
Unitree-G1-Goalkeeper-HIMPPO-AMP
```

The idea is to stop relying on a larger policy or branch-specific repair alone.
The actor still receives only deployable observations, but it now learns
supervised internal estimates of the ball interception target, ball velocity, and
goal region from privileged critic targets.  PPO then optimizes the save reward
while AMP keeps the motion close to the released region-specific goalkeeper
motions.

## What Changed

- `src/tasks/soccer/modules/gk_actor_critic.py`
  - HIMPPO-style actor-critic.
  - Actor input: current 96D frame + 16D history latent + 6D ball estimate + 1D region estimate.
  - Estimator heads are supervised during PPO, but inference still uses actor history only.
  - Fixes RSL-RL 5.x distribution API support and old checkpoint std-key migration.

- `src/tasks/soccer/modules/goalkeeper_ppo.py`
  - Extends PPO with estimator MSE/CE losses.
  - Adds policy/value smoothness on interpolated consecutive observations.
  - Adds AMP rewards and discriminator updates using next-observation pairs.

- `src/tasks/soccer/modules/goalkeeper_amp.py`
  - Six region-conditioned discriminators, one per goalkeeper motion region.
  - Produces a soft motion-prior reward for policy transitions.

- `src/tasks/soccer/modules/goalkeeper_motion_prior.py`
  - Loads `src/assets/soccer/motions/goalkeeper/{lefthand,righthand,leftjump,rightjump,leftstep,rightstep}.pt`.
  - Builds AMP states from 21 mapped G1 joint positions at `t` and `t+1`.

- `src/tasks/soccer/mdp/goalkeeper_ball_reset.py`
  - Caches the predicted keeper-plane ball crossing point in `_gk_ball_end_pos`.
  - The same cache is populated for normal random resets, forced scenarios, and failure replay.

- `src/tasks/soccer/mdp/goalkeeper_obs.py`
  - `goalkeeper_end_target_pos` now returns the cached interception target instead of the current ball position proxy.

- `src/tasks/soccer/config/g1/{rl_cfg.py,training_env_cfgs.py,__init__.py}`
  - Adds the new runner config, env config, and task registration.

## Why This Might Break the 93% Plateau

The current eval metric only cares whether the ball enters the goal.  The hard
cases are usually timing and region-selection errors: the policy sees ball
position history, but not true velocity or the intended crossing point.  A plain
PPO/LSTM policy has to discover that latent state indirectly from sparse outcome
feedback.

This branch makes that latent state explicit inside the actor:

1. The critic already has ball velocity, region, and the cached interception point.
2. The actor estimator heads learn those targets from actor-only history.
3. The policy consumes the learned estimates, so it can choose the correct dive/step earlier.
4. AMP discourages unstable or unnatural recoveries while still allowing PPO to optimize saves.

## Recommended 8-GPU Training

Single distributed run using all 8 GPUs:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python scripts/train.py Unitree-G1-Goalkeeper-HIMPPO-AMP \
  --gpu-ids all \
  --env.scene.num-envs=32768 \
  --agent.run-name himppo_amp_8gpu_32k \
  --agent.max-iterations=100000 \
  --agent.num-steps-per-env=100 \
  --agent.algorithm.num-mini-batches=16 \
  --agent.upload-model=False
```

If 32k envs is too high for your server, start with 16384:

```bash
python scripts/train.py Unitree-G1-Goalkeeper-HIMPPO-AMP \
  --gpu-ids all \
  --env.scene.num-envs=16384 \
  --agent.run-name himppo_amp_8gpu_16k \
  --agent.max-iterations=100000 \
  --agent.num-steps-per-env=100 \
  --agent.algorithm.num-mini-batches=16 \
  --agent.upload-model=False
```

Practical alternative: run 8 independent seeds, one per GPU.  This is often more
useful for a plateaued policy because you can select the best checkpoint by the
fixed official-seed eval:

```bash
for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python scripts/train.py Unitree-G1-Goalkeeper-HIMPPO-AMP \
    --gpu-ids 0 \
    --env.scene.num-envs=8192 \
    --agent.seed=$((42 + i * 1000)) \
    --agent.run-name himppo_amp_seed_$i \
    --agent.max-iterations=100000 \
    --agent.num-steps-per-env=100 \
    --agent.algorithm.num-mini-batches=8 \
    --agent.upload-model=False \
    > logs/himppo_amp_seed_$i.out 2>&1 &
done
wait
```

## Evaluation

Run the fixed official seed protocol with the official eval task:

```bash
python scripts/eval_goalkeeper_official_seeds.py \
  --checkpoint logs/rsl_rl/g1_goalkeeper_himppo_amp/<RUN>/model_<ITER>.pt \
  --task-id Eval-Goalkeeper \
  --parallel-seeds \
  --seed-gpus 0 1 2 \
  --out logs/himppo_amp_eval.json
```

For fast smoke checks while training:

```bash
python scripts/eval_goalkeeper_official_seeds.py \
  --checkpoint logs/rsl_rl/g1_goalkeeper_himppo_amp/<RUN>/model_<ITER>.pt \
  --task-id Eval-Goalkeeper \
  --seeds 42 \
  --trials-per-seed 20 \
  --device cuda:0
```

## What To Watch

- `estimator_ball` should drop steadily; if it stays high, reduce ball dropout/noise or increase estimator loss.
- `estimator_region` should get below random-class CE (`~1.79`) quickly.
- `amp_reward` should not dominate task reward.  If saves improve but posture collapses, raise `reward_coef` slightly.  If style is good but saves stall, lower it.
- Best model selection should use `eval_goalkeeper_official_seeds.py`, not training reward alone.
- If motion looks mirrored left/right, inspect `GOALKEEPER_REGION_MOTION_NAMES` in `goalkeeper_motion_prior.py`; that is the only explicit AMP region-to-motion mapping.
