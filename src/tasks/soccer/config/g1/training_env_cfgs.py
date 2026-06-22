"""G1-specific wrappers for shooter training environment configs.

Wraps the generic Stage I / Stage II factories with G1 robot
placement, action scale, contact sensors, and CCD settings.
"""

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from src.assets.robots import G1_ACTION_SCALE, get_g1_robot_cfg
from src.assets.robots.unitree_g1.g1_constants import FULL_COLLISION, HOME_KEYFRAME
from src.tasks.soccer.config.soccer_settings import SETTINGS
from src.tasks.soccer.config.training.goalkeeper_env_cfg import make_goalkeeper_env_cfg
from src.tasks.soccer.config.training.stage1_env_cfg import make_stage1_env_cfg
from src.tasks.soccer.config.training.stage2_env_cfg import make_stage2_env_cfg
from src.tasks.soccer.mdp import MultiMotionSoccerCommandCfg

import math


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
  half = yaw / 2.0
  return (math.cos(half), 0.0, 0.0, math.sin(half))


def _g1_robot_at(pos: tuple[float, float, float], yaw: float = 0.0) -> dict:
  """Create G1 robot entity config dict at a specific position and yaw."""
  cfg = get_g1_robot_cfg()
  cfg.init_state = replace(HOME_KEYFRAME, pos=pos, rot=_yaw_to_quat(yaw))
  cfg.collisions = (FULL_COLLISION,)
  return cfg


def _setup_g1_training(cfg: ManagerBasedRlEnvCfg) -> None:
  """Apply G1-specific overrides for training."""
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  from mjlab.envs.mdp.actions import JointPositionActionCfg
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.viewer.body_name = "torso_link"

  # Feet-ground contact sensor.
  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="subtree", pattern="ground", entity="ground"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  # Ball-robot contact sensor for kick detection.
  ball_robot = ContactSensorCfg(
    name="ball_robot_contact",
    primary=ContactMatch(mode="subtree", pattern="ball", entity="ball"),
    secondary=ContactMatch(mode="subtree", pattern="torso_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=2,
    history_length=4,
  )

  # Full-body contact sensor for undesired_contacts penalty.
  # Matches all robot-ground contacts for penalizing non-foot/non-wrist touches.
  contact_forces = ContactSensorCfg(
    name="contact_forces",
    primary=ContactMatch(mode="subtree", pattern=r".*", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="ground", entity="ground"),
    fields=("found", "force"),
    reduce="none",
    num_slots=2,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, ball_robot, contact_forces)

  # Inject motion_dir into the command config at registration time.
  # The actual path is set by the training script via --motion-dir.
  motion_cmd = cfg.commands.get("motion")
  if isinstance(motion_cmd, MultiMotionSoccerCommandCfg):
    motion_cmd.motion_dir = "src/assets/soccer/motions/shooter"


def unitree_g1_stage1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 shooter Stage I: motion tracking on flat ground."""
  cfg = make_stage1_env_cfg()
  cfg.scene.entities["robot"] = _g1_robot_at(tuple(SETTINGS.scene.shooter_pos), 0.0)
  # Ball position is set dynamically by MultiMotionSoccerCommand.
  # Remove the penalty-spot init_state so the entity uses its own default.
  _setup_g1_training(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    # Remove training DR and motion-reference terminations for clean visualization.
    cfg.events.pop("push_robot", None)
    for key in ("anchor_pos_z", "anchor_ori", "ee_body_pos"):
      cfg.terminations.pop(key, None)
    from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommandCfg
    motion_cmd = cfg.commands.get("motion")
    if isinstance(motion_cmd, MultiMotionSoccerCommandCfg):
      motion_cmd.debug_vis = True
      motion_cmd.sampling_mode = "uniform"
      # Disable domain randomization for clean, repeatable playback.
      motion_cmd.pose_range = {}
      motion_cmd.velocity_range = {}
      motion_cmd.joint_position_range = (0.0, 0.0)

  return cfg


def unitree_g1_stage2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 shooter Stage II: perception-guided kicking."""
  cfg = make_stage2_env_cfg()
  cfg.scene.entities["robot"] = _g1_robot_at(tuple(SETTINGS.scene.shooter_pos), 0.0)
  # Ball position is set dynamically by MultiMotionSoccerCommand.
  _setup_g1_training(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    for key in ("anchor_pos_z", "anchor_ori", "ee_body_pos"):
      cfg.terminations.pop(key, None)
    from src.tasks.soccer.mdp.shooter_commands import MultiMotionSoccerCommandCfg
    motion_cmd = cfg.commands.get("motion")
    if isinstance(motion_cmd, MultiMotionSoccerCommandCfg):
      motion_cmd.debug_vis = True
      motion_cmd.sampling_mode = "uniform"
      motion_cmd.pose_range = {}
      motion_cmd.velocity_range = {}
      motion_cmd.joint_position_range = (0.0, 0.0)

  return cfg


def unitree_g1_goalkeeper_training_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """G1 goalkeeper training: single-stage perception-based interception.

  Matches the Humanoid-Goalkeeper paper's design with 10-frame history
  stacking, 6-region parabolic ball trajectories, and asymmetric actor-critic.

  Uses the GK reference default joint positions (goalkeeper stance) as the
  action offset base, and uniform action_scale=0.25 to match the reference
  PD controller. This ensures compatibility with pretrained checkpoints.
  """
  from src.tasks.soccer.mdp.goalkeeper_obs import (
    _GK_DEFAULT_JOINT_POS, get_gk_robot_cfg,
  )
  from mjlab.envs.mdp.actions import JointPositionActionCfg

  cfg = make_goalkeeper_env_cfg()

  # Robot with GK articulation (ref-matched PD gains) and GK stance.
  robot_cfg = get_gk_robot_cfg()
  robot_cfg.init_state = replace(
    robot_cfg.init_state,
    pos=tuple(SETTINGS.scene.goalkeeper_pos),
    joint_pos=_GK_DEFAULT_JOINT_POS,
  )
  robot_cfg.collisions = (FULL_COLLISION,)
  cfg.scene.entities["robot"] = robot_cfg

  _setup_g1_training(cfg)

  # Uniform action scale 0.25 (PD gains now match reference).
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("perturb_ball_vel", None)

  return cfg


def unitree_g1_goalkeeper_lstm_ppo_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Goalkeeper env for pure recurrent PPO with a learnable curriculum."""
  cfg = unitree_g1_goalkeeper_training_env_cfg(play=play)
  if play:
    return cfg

  # From-scratch recurrent PPO needs a clean curriculum before full eval-speed
  # randomization. Official scoring does not penalize falling after a save, so
  # keep episodes alive and optimize the ball outcome directly.
  cfg.events.pop("push_robot", None)
  cfg.events.pop("perturb_ball_vel", None)
  cfg.terminations.pop("fell_over", None)

  from mjlab.managers.reward_manager import RewardTermCfg
  from src.tasks.soccer.mdp.goalkeeper_rewards import (
    goalkeeper_body_intercept,
    goalkeeper_goal_conceded,
    goalkeeper_intercept_point,
  )

  reset_ball = cfg.events["reset_ball"]
  vel_cfg = reset_ball.params["vel_cfg"]
  vel_cfg.curriculum_warmup_calls = 120_000
  vel_cfg.speed_warmup_calls = 80_000
  vel_cfg.difficulty_min = 0.15
  vel_cfg.t_flight_slow_factor = 1.8

  cfg.rewards["ee_reach"] = replace(
    cfg.rewards["ee_reach"],
    weight=15.0,
    params={**cfg.rewards["ee_reach"].params, "std": 0.45, "planar": True},
  )
  cfg.rewards["stop_ball"] = replace(cfg.rewards["stop_ball"], weight=120.0)
  cfg.rewards["goal_conceded"] = RewardTermCfg(
    func=goalkeeper_goal_conceded,
    weight=-50.0,
  )
  cfg.rewards["intercept_point"] = RewardTermCfg(
    func=goalkeeper_intercept_point,
    weight=15.0,
    params={"std": 0.45},
  )
  cfg.rewards["body_intercept"] = RewardTermCfg(
    func=goalkeeper_body_intercept,
    weight=5.0,
    params={"std": 0.35},
  )
  cfg.rewards["posture_orientation"] = replace(
    cfg.rewards["posture_orientation"], weight=1.0
  )
  cfg.rewards["feet_slippage"] = replace(cfg.rewards["feet_slippage"], weight=-1.0)
  cfg.rewards["ang_vel_xy"] = replace(cfg.rewards["ang_vel_xy"], weight=-0.02)
  return cfg


def _set_goalkeeper_region_weights(
  cfg: ManagerBasedRlEnvCfg,
  weights: tuple[float, ...],
) -> None:
  reset_ball = cfg.events.get("reset_ball")
  if reset_ball is None:
    raise KeyError("goalkeeper env is missing reset_ball event")
  vel_cfg = reset_ball.params["vel_cfg"]
  vel_cfg.region_weights = weights


def unitree_g1_goalkeeper_lstm_midup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Over-sample the current weak Mid/Up regions for recurrent PPO."""
  cfg = unitree_g1_goalkeeper_lstm_ppo_env_cfg(play=play)
  if not play:
    # Regions: 0 Right-Mid, 1 Left-Mid, 2 Right-Up, 3 Left-Up, 4/5 Low.
    _set_goalkeeper_region_weights(cfg, (1.5, 1.5, 2.0, 2.0, 0.75, 0.75))
  return cfg


def unitree_g1_goalkeeper_lstm_dive_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Allow aggressive dives by reducing upright/slip pressure."""
  cfg = unitree_g1_goalkeeper_lstm_ppo_env_cfg(play=play)
  if not play:
    cfg.rewards["posture_orientation"] = replace(
      cfg.rewards["posture_orientation"], weight=0.5
    )
    cfg.rewards["feet_slippage"] = replace(cfg.rewards["feet_slippage"], weight=-0.5)
    cfg.rewards["ang_vel_xy"] = replace(cfg.rewards["ang_vel_xy"], weight=-0.02)
    cfg.rewards["intercept_point"] = replace(
      cfg.rewards["intercept_point"], weight=8.0
    )
  return cfg


def unitree_g1_goalkeeper_lstm_block_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Put more PPO pressure on the actual block/concede objective."""
  cfg = unitree_g1_goalkeeper_lstm_ppo_env_cfg(play=play)
  if not play:
    cfg.rewards["goal_conceded"] = replace(cfg.rewards["goal_conceded"], weight=-75.0)
    cfg.rewards["stop_ball"] = replace(cfg.rewards["stop_ball"], weight=150.0)
    cfg.rewards["intercept_point"] = replace(
      cfg.rewards["intercept_point"],
      weight=20.0,
      params={"std": 0.35},
    )
  return cfg


def unitree_g1_goalkeeper_himppo_amp_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Goalkeeper env tuned for HIMPPO estimator supervision plus AMP."""
  cfg = unitree_g1_goalkeeper_lstm_ppo_env_cfg(play=play)
  if play:
    return cfg

  reset_ball = cfg.events["reset_ball"]
  vel_cfg = reset_ball.params["vel_cfg"]
  vel_cfg.curriculum_warmup_calls = 160_000
  vel_cfg.speed_warmup_calls = 120_000
  vel_cfg.difficulty_min = 0.20
  vel_cfg.t_flight_slow_factor = 1.6

  # Keep weak high/mid regions represented while AMP provides motion style.
  _set_goalkeeper_region_weights(cfg, (1.4, 1.4, 1.8, 1.8, 0.8, 0.8))

  cfg.rewards["ee_reach"] = replace(
    cfg.rewards["ee_reach"],
    weight=12.0,
    params={**cfg.rewards["ee_reach"].params, "std": 0.45, "planar": True},
  )
  cfg.rewards["stop_ball"] = replace(cfg.rewards["stop_ball"], weight=120.0)
  cfg.rewards["goal_conceded"] = replace(cfg.rewards["goal_conceded"], weight=-70.0)
  cfg.rewards["intercept_point"] = replace(
    cfg.rewards["intercept_point"],
    weight=18.0,
    params={"std": 0.38},
  )
  cfg.rewards["body_intercept"] = replace(
    cfg.rewards["body_intercept"],
    weight=4.0,
    params={"std": 0.35},
  )
  cfg.rewards["posture_orientation"] = replace(
    cfg.rewards["posture_orientation"],
    weight=0.8,
  )
  cfg.rewards["feet_slippage"] = replace(cfg.rewards["feet_slippage"], weight=-0.6)
  cfg.rewards["ang_vel_xy"] = replace(cfg.rewards["ang_vel_xy"], weight=-0.02)
  return cfg
