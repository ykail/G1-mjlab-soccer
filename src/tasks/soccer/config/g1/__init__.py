"""Unitree G1 soccer task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  unitree_g1_shooter_env_cfg,
  unitree_g1_goalkeeper_env_cfg,
)
from .rl_cfg import unitree_g1_soccer_ppo_runner_cfg
from .rl_cfg import (
  GoalkeeperRunner,
  unitree_g1_goalkeeper_himppo_amp_runner_cfg,
)
from .gk_train_cfg import (
  GoalkeeperRecurrentRunner,
  goalkeeper_lstm_ppo_runner_cfg,
)
from .training_env_cfgs import (
  unitree_g1_goalkeeper_himppo_amp_env_cfg,
  unitree_g1_goalkeeper_lstm_block_env_cfg,
  unitree_g1_goalkeeper_lstm_dive_env_cfg,
  unitree_g1_goalkeeper_lstm_midup_env_cfg,
  unitree_g1_goalkeeper_lstm_ppo_env_cfg,
)

# -- Soccer tasks -------------------------------------------------------------

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

# -- Pure PPO recurrent goalkeeper experiments -------------------------------

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-LSTM-PPO",
  env_cfg=unitree_g1_goalkeeper_lstm_ppo_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_lstm_ppo_env_cfg(play=True),
  rl_cfg=goalkeeper_lstm_ppo_runner_cfg(),
  runner_cls=GoalkeeperRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-LSTM-MidUp",
  env_cfg=unitree_g1_goalkeeper_lstm_midup_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_lstm_midup_env_cfg(play=True),
  rl_cfg=goalkeeper_lstm_ppo_runner_cfg(),
  runner_cls=GoalkeeperRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-LSTM-Dive",
  env_cfg=unitree_g1_goalkeeper_lstm_dive_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_lstm_dive_env_cfg(play=True),
  rl_cfg=goalkeeper_lstm_ppo_runner_cfg(),
  runner_cls=GoalkeeperRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-LSTM-Block",
  env_cfg=unitree_g1_goalkeeper_lstm_block_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_lstm_block_env_cfg(play=True),
  rl_cfg=goalkeeper_lstm_ppo_runner_cfg(),
  runner_cls=GoalkeeperRecurrentRunner,
)

register_mjlab_task(
  task_id="Unitree-G1-Goalkeeper-HIMPPO-AMP",
  env_cfg=unitree_g1_goalkeeper_himppo_amp_env_cfg(),
  play_env_cfg=unitree_g1_goalkeeper_himppo_amp_env_cfg(play=True),
  rl_cfg=unitree_g1_goalkeeper_himppo_amp_runner_cfg(),
  runner_cls=GoalkeeperRunner,
)
