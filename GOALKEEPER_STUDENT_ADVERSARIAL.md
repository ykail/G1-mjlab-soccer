# Single-Network Goalkeeper Adversarial Training

This branch keeps WZY's dual-robot adversarial environment, but avoids the
multi-expert keeper training path.  The trainable keeper is a single
`GoalkeeperStudentFiLMActor` checkpoint.  A frozen shooter is injected into the
same MuJoCo scene and acts every step; the compete-style task places a static
ball at `(3.0, 0.0, 0.1)` for the shooter to kick.  The keeper policy still
consumes its 964D student observation, so existing student checkpoints load
without changing input dimensions.

Why this exists:

- MoE/patch keepers are good deploy policies, but they are awkward PPO init
  points because routing, prepare behavior, and multiple expert checkpoints all
  need coordinated updates.
- A single student actor is slower to become excellent from scratch, but it is
  the right object to fine-tune adversarially.
- Phase 2 benefits from this setup because the keeper sees a real shooter
  policy distribution.  Phase 1 random parabolic-ball success is a different
  distribution; use the best Phase 1 keeper/patch checkpoint for that.

Find the shooter checkpoint:

```bash
find checkpoints logs/rsl_rl/g1_soccer -path '*model_*.pt' \
  | grep -E 'stage6|stage5|stage4|stage3|stage2' \
  | sort
```

The GitHub branch only archives `checkpoints/stage2/model_100000.pt`.  Stronger
Stage4/Stage6 shooters are usually in server logs, for example
`logs/rsl_rl/g1_soccer/<stage4-or-stage6-run>/model_*.pt`.

Smoke test:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

/data/mjlab-cu126/bin/python -m unittest \
  tests.test_adversarial_dual_robot_cfg \
  tests.test_goalkeeper_student_bc

/data/mjlab-cu126/bin/python scripts/train_goalkeeper_student_adversarial.py \
  --init <student_keeper.pt> \
  --shooter <strong_shooter.pt> \
  --shooter-task-id Eval-Shooter-Stage6 \
  --out-dir logs/adversarial_student_keeper_smoke \
  --gpu-ids 0 \
  --num-envs 128 \
  --max-iterations 1 \
  --dry-run
```

The script uses task `Unitree-G1-Goalkeeper-Student-Compete-Adversarial`, not
the older parabolic-ball `Unitree-G1-Goalkeeper-Student-Adversarial`.  This is
important: the frozen shooter must generate the incoming ball distribution.

Training command on 4 H20 GPUs:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MASTER_ADDR=127.0.0.1

/data/mjlab-cu126/bin/python scripts/train_goalkeeper_student_adversarial.py \
  --init <student_keeper.pt> \
  --shooter <strong_shooter.pt> \
  --shooter-task-id Eval-Shooter-Stage6 \
  --out-dir logs/adversarial_student_keeper \
  --run-name student_keeper_adv_h20 \
  --gpu-ids 0,1,2,3 \
  --num-envs 4096 \
  --max-iterations 3000 \
  --distill-coef 0.2 \
  --distill-final-coef 0.0 \
  --distill-anneal-updates 1500 \
  --learning-rate 1e-4 \
  --entropy-coef 5e-4 \
  --min-action-std 0.05 \
  --max-action-std 0.35 \
  --actor-mean-clip 2.0 \
  --mask-idle-actor-loss \
  --idle-deterministic-actions
```

Do not pass `checkpoints/keeper_93_moe6.pt`, a residual checkpoint, or a
patch-library checkpoint as `--init`; the script rejects them intentionally.
Use the latest repaired/patch keeper as a Phase 1 deploy candidate, and use a
single student checkpoint for adversarial PPO.

Only pass `--teacher` if the checkpoint is a MoE7-style bundle with an
`idle`/`prepare` expert.  A plain MoE6 bundle such as `keeper_93_moe6.pt` is not
accepted as the online student teacher.

If the shooter checkpoint is only `checkpoints/stage2/model_100000.pt`, use:

```bash
--shooter checkpoints/stage2/model_100000.pt --shooter-task-id Eval-Shooter
```

For a Stage4/Stage6 checkpoint, prefer the matching `Eval-Shooter-Stage4` or
`Eval-Shooter-Stage6` task.
