"""Tests for dual-robot adversarial soccer task configuration."""

from __future__ import annotations

import unittest

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls


class AdversarialDualRobotCfgTest(unittest.TestCase):
  def test_shooter_adversarial_task_has_real_frozen_goalkeeper_opponent(self):
    self.assertIn("Unitree-G1-Shooter-Adversarial", list_tasks())

    cfg = load_env_cfg("Unitree-G1-Shooter-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Shooter-Adversarial")

    self.assertIn("robot", cfg.scene.entities)
    self.assertIn("opponent", cfg.scene.entities)
    self.assertIn("ball", cfg.scene.entities)
    self.assertIn("joint_pos", cfg.actions)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(cfg.actions["joint_pos"].entity_name, "robot")
    self.assertEqual(cfg.actions["opponent_joint_pos"].entity_name, "opponent")
    self.assertIsInstance(cfg.actions["joint_pos"], JointPositionActionCfg)
    self.assertIsInstance(cfg.actions["opponent_joint_pos"], JointPositionActionCfg)
    self.assertIn("opponent_root", cfg.observations["actor"].terms)
    self.assertIn("opponent_joints", cfg.observations["actor"].terms)
    self.assertIn("opponent_root", cfg.observations["critic"].terms)
    self.assertEqual(rl_cfg.obs_groups["actor"], ("actor",))
    self.assertEqual(rl_cfg.actor.class_name, "AdversarialRNNModel")
    self.assertEqual(load_runner_cls("Unitree-G1-Shooter-Adversarial").__name__, "AdversarialSoccerRecurrentRunner")

  def test_goalkeeper_adversarial_task_has_real_frozen_shooter_opponent(self):
    self.assertIn("Unitree-G1-Goalkeeper-Adversarial", list_tasks())

    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Adversarial")

    self.assertIn("robot", cfg.scene.entities)
    self.assertIn("opponent", cfg.scene.entities)
    self.assertIn("ball", cfg.scene.entities)
    self.assertIn("joint_pos", cfg.actions)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(cfg.actions["joint_pos"].entity_name, "robot")
    self.assertEqual(cfg.actions["opponent_joint_pos"].entity_name, "opponent")
    self.assertAlmostEqual(cfg.actions["joint_pos"].scale, 0.25)
    self.assertIn("opponent_root", cfg.observations["actor"].terms)
    self.assertIn("opponent_joints", cfg.observations["actor"].terms)
    self.assertIn("opponent_root", cfg.observations["critic"].terms)
    self.assertEqual(rl_cfg.actor.class_name, "AdversarialGoalkeeperActorCritic")
    self.assertEqual(load_runner_cls("Unitree-G1-Goalkeeper-Adversarial").__name__, "AdversarialGoalkeeperRunner")

  def test_goalkeeper_expert_adversarial_task_hides_shooter_state_from_regular_experts(self):
    self.assertIn("Unitree-G1-Goalkeeper-Expert-Adversarial", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Expert-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Expert-Adversarial")

    self.assertIn("opponent", cfg.scene.entities)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(cfg.actions["opponent_joint_pos"].entity_name, "opponent")
    self.assertNotIn("opponent_root", cfg.observations["actor"].terms)
    self.assertNotIn("opponent_joints", cfg.observations["actor"].terms)
    self.assertNotIn("opponent_root", cfg.observations["critic"].terms)
    self.assertIn("goal_conceded", cfg.rewards)
    self.assertLess(cfg.rewards["goal_conceded"].weight, 0.0)
    self.assertIn("intercept", cfg.rewards)
    self.assertIn("body", cfg.rewards)
    self.assertIn("stop_ball", cfg.rewards)
    self.assertGreater(cfg.rewards["stop_ball"].weight, 0.0)
    self.assertIn("action_rate", cfg.rewards)
    self.assertNotIn("ee_reach", cfg.rewards)
    self.assertNotIn("idle_fall_penalty", cfg.rewards)
    self.assertEqual(rl_cfg.actor.class_name, "MLPModel")
    self.assertIsNone(load_runner_cls("Unitree-G1-Goalkeeper-Expert-Adversarial"))

  def test_goalkeeper_idle_adversarial_task_trains_ready_stance_with_frozen_shooter(self):
    self.assertIn("Unitree-G1-Goalkeeper-Idle-Adversarial", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Idle-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Idle-Adversarial")

    self.assertIn("opponent", cfg.scene.entities)
    self.assertGreater(cfg.scene.entities["opponent"].init_state.pos[0], 3.0)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(cfg.actions["opponent_joint_pos"].entity_name, "opponent")
    self.assertIn("opponent_root", cfg.observations["actor"].terms)
    self.assertIn("opponent_joints", cfg.observations["critic"].terms)
    self.assertEqual(cfg.events["reset_ball"].func.__name__, "reset_ball_static_for_goalkeeper_idle")
    self.assertIn("goal_conceded", cfg.rewards)
    self.assertLess(cfg.rewards["goal_conceded"].weight, 0.0)
    self.assertIn("intercept", cfg.rewards)
    self.assertIn("body", cfg.rewards)
    self.assertIn("stop_ball", cfg.rewards)
    self.assertGreater(cfg.rewards["stop_ball"].weight, 0.0)
    self.assertIn("action_rate", cfg.rewards)
    self.assertNotIn("ee_reach", cfg.rewards)
    self.assertIn("idle_fall_penalty", cfg.rewards)
    self.assertNotIn("idle_joint_pose", cfg.rewards)
    self.assertNotIn("idle_base_still", cfg.rewards)
    self.assertNotIn("idle_alive", cfg.rewards)
    self.assertNotIn("idle_timeout_success", cfg.rewards)
    self.assertNotIn("stay_on_line", cfg.rewards)
    self.assertNotIn("posture_orientation", cfg.rewards)
    self.assertLessEqual(cfg.rewards["idle_fall_penalty"].weight, -1000.0)
    self.assertEqual(cfg.events["reset_ball"].params["idle_wait_range_s"], (8.0, 12.0))
    self.assertEqual(cfg.terminations["time_out"].func.__name__, "time_out")
    self.assertNotIn("ball_started", cfg.terminations)
    self.assertAlmostEqual(cfg.episode_length_s, 12.0)
    self.assertEqual(rl_cfg.actor.class_name, "AdversarialGoalkeeperActorCritic")
    self.assertEqual(load_runner_cls("Unitree-G1-Goalkeeper-Idle-Adversarial").__name__, "AdversarialGoalkeeperRunner")

  def test_goalkeeper_moe7_prepare_adversarial_task_runs_full_moe_actor(self):
    self.assertIn("Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial")

    self.assertIn("opponent", cfg.scene.entities)
    self.assertIn("opponent_root", cfg.observations["actor"].terms)
    self.assertIn("opponent_joints", cfg.observations["actor"].terms)
    self.assertIn("goal_conceded", cfg.rewards)
    self.assertIn("intercept", cfg.rewards)
    self.assertIn("stop_ball", cfg.rewards)
    self.assertIn("idle_fall_penalty", cfg.rewards)
    self.assertIn("idle_low_base_height", cfg.rewards)
    self.assertIn("idle_leg_ready_pose", cfg.rewards)
    self.assertEqual(rl_cfg.actor.class_name, "MoE7PrepareGoalkeeperActor")
    self.assertEqual(rl_cfg.critic.class_name, "AdversarialGoalkeeperActorCritic")
    self.assertEqual(rl_cfg.algorithm.entropy_coef, 0.0)
    self.assertEqual(rl_cfg.actor.distribution_cfg["idle_std_min"], 0.15)
    self.assertEqual(rl_cfg.actor.distribution_cfg["idle_std_max"], 0.15)
    self.assertEqual(cfg.rewards["action_rate"].weight, -0.5)
    self.assertGreater(cfg.rewards["posture"].weight, 0.0)
    self.assertEqual(cfg.rewards["ang_vel_xy"].weight, -0.5)
    self.assertEqual(cfg.rewards["idle_low_base_height"].weight, -20.0)
    self.assertEqual(cfg.rewards["idle_low_base_height"].params["target_z"], 0.73)
    self.assertEqual(cfg.rewards["idle_leg_ready_pose"].weight, -5.0)
    self.assertEqual(load_runner_cls("Unitree-G1-Goalkeeper-MoE7-Prepare-Adversarial").__name__, "MoE7PrepareGoalkeeperRunner")

  def test_goalkeeper_idle_train_task_uses_original_keeper_runner(self):
    self.assertIn("Unitree-G1-Goalkeeper-Idle-Train", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Idle-Train")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Idle-Train")

    self.assertIn("robot", cfg.scene.entities)
    self.assertNotIn("opponent", cfg.scene.entities)
    self.assertNotIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(cfg.events["reset_ball"].func.__name__, "reset_ball_static_for_goalkeeper_idle")
    self.assertEqual(cfg.events["reset_ball"].params["idle_wait_range_s"], (8.0, 12.0))
    self.assertEqual(cfg.terminations["time_out"].func.__name__, "goalkeeper_idle_timeout")
    self.assertAlmostEqual(cfg.episode_length_s, 12.0)
    self.assertIn("idle_fall_penalty", cfg.rewards)
    self.assertIn("idle_low_base_height", cfg.rewards)
    self.assertIn("idle_base_height_band", cfg.rewards)
    self.assertNotIn("idle_leg_ready_pose", cfg.rewards)
    self.assertNotIn("idle_joint_pose", cfg.rewards)
    self.assertNotIn("idle_timeout_success", cfg.rewards)
    self.assertEqual(cfg.rewards["idle_low_base_height"].weight, -20.0)
    self.assertEqual(cfg.rewards["idle_low_base_height"].params["target_z"], 0.72)
    self.assertEqual(cfg.rewards["idle_base_height_band"].weight, 5.0)
    self.assertEqual(cfg.rewards["idle_base_height_band"].params["target_z"], 0.72)
    self.assertEqual(rl_cfg.actor.class_name, "GoalkeeperActorCritic")
    self.assertEqual(load_runner_cls("Unitree-G1-Goalkeeper-Idle-Train").__name__, "GoalkeeperRunner")

  def test_goalkeeper_idle_rewards_do_not_affect_regular_adversarial_keeper(self):
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Adversarial")

    self.assertNotIn("idle_alive", cfg.rewards)
    self.assertNotIn("idle_timeout_success", cfg.rewards)
    self.assertNotIn("idle_low_base_height", cfg.rewards)
    self.assertNotIn("idle_leg_ready_pose", cfg.rewards)

  def test_goalkeeper_student_adversarial_task_trains_single_network_against_shooter(self):
    self.assertIn("Unitree-G1-Goalkeeper-Student-Adversarial", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Student-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Student-Adversarial")

    self.assertIn("opponent", cfg.scene.entities)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(list(cfg.observations["actor"].terms.keys()), ["student"])
    self.assertNotIn("opponent_root", cfg.observations["actor"].terms)
    self.assertNotIn("opponent_joints", cfg.observations["actor"].terms)
    self.assertIn("goal_conceded", cfg.rewards)
    self.assertIn("idle_fall_penalty", cfg.rewards)
    self.assertEqual(rl_cfg.actor.class_name, "GoalkeeperStudentFiLMActor")
    self.assertEqual(
      rl_cfg.algorithm.class_name,
      "src.tasks.soccer.modules.goalkeeper_student_ppo:GoalkeeperStudentPPO",
    )
    self.assertEqual(
      load_runner_cls("Unitree-G1-Goalkeeper-Student-Adversarial").__name__,
      "GoalkeeperStudentRunner",
    )

  def test_goalkeeper_student_compete_adversarial_task_uses_static_shooter_ball(self):
    self.assertIn("Unitree-G1-Goalkeeper-Student-Compete-Adversarial", list_tasks())
    cfg = load_env_cfg("Unitree-G1-Goalkeeper-Student-Compete-Adversarial")
    rl_cfg = load_rl_cfg("Unitree-G1-Goalkeeper-Student-Compete-Adversarial")

    self.assertIn("opponent", cfg.scene.entities)
    self.assertGreater(cfg.scene.entities["opponent"].init_state.pos[0], 3.0)
    self.assertIn("opponent_joint_pos", cfg.actions)
    self.assertEqual(list(cfg.observations["actor"].terms.keys()), ["student"])
    self.assertNotIn("opponent_root", cfg.observations["actor"].terms)
    self.assertEqual(cfg.events["reset_ball"].func.__name__, "reset_ball_static_for_goalkeeper_idle")
    self.assertEqual(cfg.events["reset_ball"].params["ball_pos"], (3.0, 0.0, 0.1))
    self.assertEqual(cfg.events["reset_ball"].params["idle_wait_range_s"], (8.0, 12.0))
    self.assertIn("idle_fall_penalty", cfg.rewards)
    self.assertIn("idle_leg_ready_pose", cfg.rewards)
    self.assertEqual(cfg.terminations["time_out"].func.__name__, "time_out")
    self.assertNotIn("ball_started", cfg.terminations)
    self.assertAlmostEqual(cfg.episode_length_s, 12.0)
    self.assertEqual(rl_cfg.actor.class_name, "GoalkeeperStudentFiLMActor")
    self.assertEqual(
      load_runner_cls("Unitree-G1-Goalkeeper-Student-Compete-Adversarial").__name__,
      "GoalkeeperStudentRunner",
    )


if __name__ == "__main__":
  unittest.main()
