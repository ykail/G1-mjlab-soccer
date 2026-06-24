"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)
from .rl_cfg import (
  AdversarialGoalkeeperRunner,
  AdversarialSoccerRecurrentRunner,
  GoalkeeperRunner,
  GoalkeeperStudentRunner,
  MoE7PrepareGoalkeeperRunner,
  SoccerRecurrentRunner,
  unitree_g1_goalkeeper_adversarial_ppo_runner_cfg,
  unitree_g1_goalkeeper_moe7_prepare_ppo_runner_cfg,
  unitree_g1_goalkeeper_ppo_runner_cfg,
  unitree_g1_goalkeeper_student_ppo_runner_cfg,
  unitree_g1_soccer_ppo_runner_cfg,
  unitree_g1_soccer_adversarial_recurrent_runner_cfg,
  unitree_g1_soccer_recurrent_runner_cfg,
  unitree_g1_student_shooter_ppo_runner_cfg,
)
from .gk_train_cfg import goalkeeper_train_runner_cfg
from .training_env_cfgs import (
  unitree_g1_goalkeeper_adversarial_env_cfg,
  unitree_g1_goalkeeper_expert_adversarial_env_cfg,
  unitree_g1_goalkeeper_idle_adversarial_env_cfg,
  unitree_g1_goalkeeper_idle_training_env_cfg,
  unitree_g1_goalkeeper_student_adversarial_env_cfg,
  unitree_g1_goalkeeper_student_compete_adversarial_env_cfg,
  unitree_g1_goalkeeper_student_ppo_env_cfg,
  unitree_g1_goalkeeper_training_env_cfg,
  unitree_g1_shooter_adversarial_env_cfg,
  unitree_g1_stage1_env_cfg,
  unitree_g1_stage2_env_cfg,
  unitree_g1_stage3_env_cfg,
  unitree_g1_stage4_env_cfg,
  unitree_g1_stage5_env_cfg,
  unitree_g1_stage6_env_cfg,
  unitree_g1_student_shooter_env_cfg,
)

# -- Playable tasks ------------------------------------------------------------

register_mjlab_task(
  task_id="Unitree-G1-Shooter",
  env_cfg=unitree_g1_shooter_env_cfg(),
  play_env_cfg=unitree_g1_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper",
  env_cfg=unitree_g1_goalkeeper_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_ppo_runner_cfg(),
  runner_cls=None,
)

# -- Training tasks ------------------------------------------------------------

# Stage I: motion tracking (LSTM, 128-64-32 + LSTM 2×128, matching PAiD paper)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage1",
  env_cfg=unitree_g1_stage1_env_cfg(),
  play_env_cfg=unitree_g1_stage1_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage II: perception-guided kicking (LSTM, 128-64-32 + LSTM 2×128)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage2",
  env_cfg=unitree_g1_stage2_env_cfg(),
  play_env_cfg=unitree_g1_stage2_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage III: goal-plane accuracy + speed curriculum (LSTM, same architecture)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage3",
  env_cfg=unitree_g1_stage3_env_cfg(),
  play_env_cfg=unitree_g1_stage3_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage IV: high-speed goal-plane accuracy (LSTM, same architecture)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage4",
  env_cfg=unitree_g1_stage4_env_cfg(),
  play_env_cfg=unitree_g1_stage4_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage V: full-goal-width accuracy + quality control (LSTM, same architecture)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage5",
  env_cfg=unitree_g1_stage5_env_cfg(),
  play_env_cfg=unitree_g1_stage5_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Stage VI: high-speed compete-aligned kicking (LSTM, same architecture)
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Stage6",
  env_cfg=unitree_g1_stage6_env_cfg(),
  play_env_cfg=unitree_g1_stage6_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_recurrent_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

# Student shooter: motion-free observation, LSTM policy (BC→PPO).
# SoccerRecurrentRunner injects rnn_type/layers/hidden_dim into the config dict.
register_mjlab_task(
  task_id="Unitree-G1-Shooter-Student",
  env_cfg=unitree_g1_student_shooter_env_cfg(),
  play_env_cfg=unitree_g1_student_shooter_env_cfg(play=True),
  rl_cfg=unitree_g1_student_shooter_ppo_runner_cfg(),
  runner_cls=SoccerRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Train",
  env_cfg=unitree_g1_goalkeeper_training_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_training_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_ppo_runner_cfg(),
  runner_cls=GoalkeeperRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Student-PPO",
  env_cfg=unitree_g1_goalkeeper_student_ppo_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_student_ppo_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_student_ppo_runner_cfg(),
  runner_cls=GoalkeeperStudentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Student-Adversarial",
  env_cfg=unitree_g1_goalkeeper_student_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_student_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_student_ppo_runner_cfg(),
  runner_cls=GoalkeeperStudentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Student-Compete-Adversarial",
  env_cfg=unitree_g1_goalkeeper_student_compete_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_student_compete_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_student_ppo_runner_cfg(),
  runner_cls=GoalkeeperStudentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Idle-Train",
  env_cfg=unitree_g1_goalkeeper_idle_training_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_idle_training_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_ppo_runner_cfg(),
  runner_cls=GoalkeeperRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Shooter-Adversarial",
  env_cfg=unitree_g1_shooter_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_shooter_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_soccer_adversarial_recurrent_runner_cfg(),
  runner_cls=AdversarialSoccerRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Adversarial",
  env_cfg=unitree_g1_goalkeeper_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_adversarial_ppo_runner_cfg(),
  runner_cls=AdversarialGoalkeeperRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Expert-Adversarial",
  env_cfg=unitree_g1_goalkeeper_expert_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_expert_adversarial_env_cfg(play=True),
  rl_cfg=goalkeeper_train_runner_cfg(),
  runner_cls=None,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-Idle-Adversarial",
  env_cfg=unitree_g1_goalkeeper_idle_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_idle_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_adversarial_ppo_runner_cfg(),
  runner_cls=AdversarialGoalkeeperRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial",
  env_cfg=unitree_g1_goalkeeper_idle_adversarial_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_idle_adversarial_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_moe7_prepare_ppo_runner_cfg(),
  runner_cls=MoE7PrepareGoalkeeperRunner,
)
