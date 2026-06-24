"""Policy server for Phase 2 tournament — shooter teacher (160D LSTM).

Receives raw MuJoCo state from ``compete.py`` and computes the full 160D
training observation, filling motion-reference terms from real .npz data.

  POST /act    - receive raw state, return action
  POST /reset  - reset policy + motion + destination state

Usage:
  # Shooter server
  python scripts/api_server.py \\
      --checkpoint checkpoints/stage4/model_138985.pt \\
      --port 8000 --task shooter

  # With custom motion directory
  python scripts/api_server.py \\
      --checkpoint model.pt --port 8000 --task shooter \\
      --motion-dir src/assets/soccer/motions/shooter

Observation — 160D (9 terms, concatenated):
  command(ref_joint_pos+ref_joint_vel) 58 + projected_gravity 3 +
  motion_ref_ang_vel 3 + base_ang_vel 3 + joint_pos_rel 29 +
  joint_vel_rel 29 + last_action 29 + target_point_pos 3 +
  target_destination_pos 3                = 160
"""

from __future__ import annotations

import glob
import os
import random
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
import tyro
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_inv

from src.tasks.soccer.mdp.shooter_commands import _IL_TO_MJCF_JOINT, _IL_TO_MJCF_BODY
from src.tasks.soccer.modules.moe6_goalkeeper_policy import MoE6GoalkeeperPolicy as ApiMoE6Policy

_KEEPER_ADV_TASK_ID = "Unitree-G1-Goalkeeper-Adversarial"

# ---------------------------------------------------------------------------
# HOME_KEYFRAME default joint positions (29D, MJCF order)
# ---------------------------------------------------------------------------

_HOME_KEYFRAME_JOINT_POS = torch.tensor([
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,   # left  leg (6)
    -0.1,  0.0,  0.0,  0.3, -0.2,  0.0,   # right leg (6)
     0.0,  0.0,  0.0,                       # waist    (3)
     0.35, 0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # left  arm (7)
     0.35,-0.18, 0.0, 0.87, 0.0, 0.0, 0.0, # right arm (7)
])

_GK_DEFAULT_JOINT_POS = torch.tensor([
    0.0, 0.0, 0.0,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    -0.35, 0.7, -0.35, -0.25, 0.3, -0.1,
    0.0, 0.3, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
    0.8, 0.0, -1.6, 0.0, 0.5, 0.0, 0.0,
])

# ---------------------------------------------------------------------------
# Motion data
# ---------------------------------------------------------------------------

TORSO_MJCF_IDX = 15  # torso_link in MJCF body order
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_MOTION_DIR = "src/assets/soccer/motions/shooter"


@dataclass
class MotionData:
    joint_pos: torch.Tensor       # (T, 29) MJCF order
    joint_vel: torch.Tensor       # (T, 29) MJCF order
    anchor_ang_vel: torch.Tensor  # (T, 3)   torso ang vel
    name: str


def _load_motions(motion_dir: str, device: str) -> list[MotionData]:
    """Load all soccer-standard-*.npz, apply IL→MJCF permute, return list."""
    motion_path = motion_dir
    if not os.path.isabs(motion_path) and not os.path.isdir(motion_path):
        repo_relative = os.path.join(_REPO_ROOT, motion_path)
        if os.path.isdir(repo_relative):
            motion_path = repo_relative
    pattern = os.path.join(motion_path, "soccer-standard-*.npz")
    files = sorted(glob.glob(pattern))
    motions = []
    for f in files:
        data = np.load(f)
        jp = torch.tensor(data["joint_pos"][:, _IL_TO_MJCF_JOINT],
                          dtype=torch.float32, device=device)
        jv = torch.tensor(data["joint_vel"][:, _IL_TO_MJCF_JOINT],
                          dtype=torch.float32, device=device)
        bav = torch.tensor(data["body_ang_vel_w"][:, _IL_TO_MJCF_BODY, :],
                           dtype=torch.float32, device=device)
        anchor_av = bav[:, TORSO_MJCF_IDX, :]
        motions.append(MotionData(jp, jv, anchor_av, os.path.basename(f)))
    if not motions:
        raise FileNotFoundError(f"No soccer-standard-*.npz found under {motion_path}")
    print(f"[INFO] Loaded {len(motions)} motion files from {motion_path}")
    return motions


# ---------------------------------------------------------------------------
# Kick frame map — measured by analyze_kick_timing.py with 50 trials/motion.
# All motions kick consistently at frame 26-30 (0.52-0.60s), std=0.
# ---------------------------------------------------------------------------

_KICK_FRAME_MAP: dict[str, dict[str, int | float]] = {
    "soccer-standard-001_right.npz": {"total_frames": 260, "kick_frame": 27},
    "soccer-standard-002_left.npz":  {"total_frames": 214, "kick_frame": 29},
    "soccer-standard-003_left.npz":  {"total_frames": 289, "kick_frame": 30},
    "soccer-standard-004_right.npz": {"total_frames": 257, "kick_frame": 27},
    "soccer-standard-005_right.npz": {"total_frames": 230, "kick_frame": 26},
    "soccer-standard-006_right.npz": {"total_frames": 225, "kick_frame": 27},
    "soccer-standard-007_left.npz":  {"total_frames": 229, "kick_frame": 29},
    "soccer-standard-008_left.npz":  {"total_frames": 225, "kick_frame": 29},
    "soccer-standard-009_right.npz": {"total_frames": 205, "kick_frame": 27},
    "soccer-standard-010_right.npz": {"total_frames": 230, "kick_frame": 27},
}


# ---------------------------------------------------------------------------
# Observation computation  (CUSTOMIZE: match your training observation space)
# ---------------------------------------------------------------------------

def _plan_destination(gk_y: float) -> float:
    """基于守门员 y 位置决策射门目标 y。

    Strategy:
      - GK 在左 (y < -0.3) → 射右侧全区域
      - GK 在右 (y > +0.3) → 射左侧全区域
      - GK 在中间 (|y| ≤ 0.3) → 全门均匀随机

    射门覆盖范围 [0.3, 1.4] 覆盖球门 93% 宽度 (球门 ±1.5m)。
    """
    THRESHOLD = 0.3
    if gk_y < -THRESHOLD:
        return random.uniform(0.3, 1.4)
    elif gk_y > THRESHOLD:
        return random.uniform(-1.4, -0.3)
    else:
        return random.uniform(-1.4, 1.4)


def _select_motion_for_dest(dest_y: float, motions: list[MotionData]) -> MotionData:
    """根据目标位置 y 选最优 right-leg motion。

    映射来自 motion analysis 评测数据（5000 trials/pair）：
      - y < -0.7:        006_right — 速度 16.2m/s, err 0.10m  (左远角)
      - -0.7 ≤ y < -0.3: 009_right — 速度 14.2m/s, err 0.35m  (左中)
      - |y| ≤ 0.3:       010_right — 速度 16.4m/s, err 0.11m  (正中)
      - 0.3 < y ≤ 0.7:   005_right — 速度 12.1m/s, err 0.006m (右中)
      - y > 0.7:         005_right — 速度 15.7m/s, err 0.64m  (右远角)

    随 dest_y 扩展，函数确保覆盖 [-1.4, 1.4] 全范围。
    """
    PATTERNS: list[tuple[float, float, str]] = [
        (-1.5, -0.7, "006_right.npz"),   # 左远角
        (-0.7, -0.3, "009_right.npz"),   # 左中
        (-0.3,  0.3, "010_right.npz"),   # 正中
        ( 0.3,  0.7, "005_right.npz"),   # 右中
        ( 0.7,  1.5, "005_right.npz"),   # 右远角
    ]
    for lo, hi, suffix in PATTERNS:
        if lo <= dest_y <= hi:
            for m in motions:
                if m.name.endswith(suffix):
                    return m
    for m in motions:
        if "005_right" in m.name:
            return m
    return motions[0]


def _should_lock(gk_y_history: list[float], motion_name: str,
                 min_observe: int = 5, commit_threshold: float = 0.3) -> bool:
    """Returns True when we should lock the destination.

    Conditions:
      1. GK has visibly committed to one side for min_observe frames → lock early
      2. We've reached 60% of kick_frame deadline → lock by force
    """
    if len(gk_y_history) < min_observe:
        return False

    kick_info = _KICK_FRAME_MAP.get(motion_name, {"kick_frame": 27})
    deadline = int(kick_info["kick_frame"] * 0.6)

    if len(gk_y_history) >= deadline:
        return True

    recent = gk_y_history[-min_observe:]
    if abs(recent[-1]) >= commit_threshold:
        return True

    return False


def compute_shooter_obs(
    raw_state: dict,
    motion: MotionData,
    time_step: int,
    destination_world: tuple[float, float, float],
    device: str,
) -> torch.Tensor:
    """Compute 160D shooter observation from raw state + motion data.

    Concatenation order matches ``stage1_env_cfg.py`` actor_terms (9 terms).
    All tensors created on ``device`` to match the motion data device.
    """
    s = raw_state["shooter"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"], device=device)
    root_pos = torch.tensor(s["root_pos"], device=device)
    root_ang_vel = torch.tensor(s["root_ang_vel"], device=device)
    joint_pos = torch.tensor(s["joint_pos"], device=device)
    joint_vel = torch.tensor(s["joint_vel"], device=device)
    ball_pos = torch.tensor(ball["pos"], device=device)
    last_action = torch.tensor(s["last_action"], device=device)

    # Freeze at last frame when motion ends.
    T = motion.joint_pos.shape[0]
    t = min(time_step, T - 1)

    # 1. command — ref joint_pos + joint_vel (58D)
    command = torch.cat([motion.joint_pos[t], motion.joint_vel[t]])

    # 2. projected_gravity (3D)
    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=device)
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    # 3. motion_ref_ang_vel — reference anchor angular velocity (3D)
    motion_ref_ang_vel = motion.anchor_ang_vel[t]

    # 4. base_ang_vel — actual base angular velocity in body frame (3D)
    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel)

    # 5. joint_pos_rel — relative to HOME_KEYFRAME (29D)
    joint_pos_rel = joint_pos - _HOME_KEYFRAME_JOINT_POS.to(device)

    # 6. joint_vel_rel (29D, default_vel = 0)
    joint_vel_rel = joint_vel

    # 7. last_action (29D) — already have it

    # 8. target_point_pos — ball in robot pelvis frame (3D)
    ball_body = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    # 9. target_destination_pos — destination in robot pelvis frame (3D)
    dest_w = torch.tensor(destination_world, device=device)
    dest_body = quat_apply(quat_inv(root_quat), dest_w - root_pos)

    obs = torch.cat([
        command,             # 58
        projected_gravity,   #  3
        motion_ref_ang_vel,  #  3
        base_ang_vel,        #  3
        joint_pos_rel,       # 29
        joint_vel_rel,       # 29
        last_action,         # 29
        ball_body,           #  3
        dest_body,           #  3
    ])  # → 160

    return obs.unsqueeze(0)  # (1, 160)


def compute_goalkeeper_obs(raw_state: dict) -> torch.Tensor:
    """Compute goalkeeper observation tensor from raw state (single frame).

    Default: matches ``eval_goalkeeper_cfg`` per-frame terms (96-D).
    Replace with your own obs terms, scaling, and concatenation order.
    """
    s = raw_state["goalkeeper"]
    ball = raw_state["ball"]

    root_quat = torch.tensor(s["root_quat"])
    root_ang_vel = torch.tensor(s["root_ang_vel"])
    joint_pos = torch.tensor(s["joint_pos"])
    joint_vel = torch.tensor(s["joint_vel"])
    ball_pos = torch.tensor(ball["pos"])
    root_pos = torch.tensor(s["root_pos"])
    last_action = torch.tensor(s["last_action"])

    gravity_w = torch.tensor([0.0, 0.0, -1.0])
    projected_gravity = quat_apply(quat_inv(root_quat), gravity_w)

    base_ang_vel = quat_apply(quat_inv(root_quat), root_ang_vel) * 0.25
    joint_pos_rel = (joint_pos - _GK_DEFAULT_JOINT_POS) * 1.0
    joint_vel_scaled = joint_vel * 0.05
    ball_pos_local = quat_apply(quat_inv(root_quat), ball_pos - root_pos)

    obs = torch.cat([
        ball_pos_local,         # 3
        base_ang_vel,           # 3
        projected_gravity,      # 3
        joint_pos_rel,          # 29
        joint_vel_scaled,       # 29
        last_action,            # 29
    ])
    return obs.unsqueeze(0)  # (1, 96)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ActResponse(BaseModel):
    action: list[list[float]]


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------

def _action_dim(env: Any) -> int:
    """Return action dimension across mjlab env/wrapper versions."""
    if hasattr(env, "num_actions"):
        return int(env.num_actions)
    return int(env.action_manager.total_action_dim)


def _as_batch(values: list[float], device: str | torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).view(1, -1)


def _has_scene_entity(scene: Any, name: str) -> bool:
    if hasattr(scene, "entities"):
        return name in scene.entities
    try:
        scene[name]
    except KeyError:
        return False
    return True


def _checkpoint_has_actor_key(checkpoint_path: str, prefix: str, device: str) -> bool:
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor_state = loaded.get("actor_state_dict") if isinstance(loaded, dict) else None
    return isinstance(actor_state, dict) and any(key.startswith(prefix) for key in actor_state)


def _moe_bundle_from_checkpoint(loaded: Any) -> dict[str, Any] | None:
    if not isinstance(loaded, dict):
        return None
    if "sr" in loaded:
        return loaded
    actor_state = loaded.get("actor_state_dict")
    if isinstance(actor_state, dict) and "sr" in actor_state:
        return actor_state
    return None


def _moe_bundle_has_adversarial_idle(bundle: dict[str, Any]) -> bool:
    idle = next((bundle[key] for key in ("idle", "prepare", "idle_expert") if key in bundle), None)
    if not isinstance(idle, dict):
        return False
    actor_state = idle.get("actor_state_dict")
    if not isinstance(actor_state, dict):
        return False
    return any(key.startswith(("actor_residual.", "critic_residual.")) for key in actor_state)


def _set_action_term(action_manager: Any, term_name: str, action: torch.Tensor) -> None:
    if not hasattr(action_manager, "_terms"):
        action_manager._action[:, :] = action
        if hasattr(action_manager, "get_term"):
            term = action_manager.get_term(term_name)
            if hasattr(term, "_raw_actions"):
                term._raw_actions[:, :] = action
        return

    offset = 0
    for name, term in action_manager._terms.items():
        next_offset = offset + term.action_dim
        if name == term_name:
            action_manager._action[:, offset:next_offset] = action.to(action_manager._action.device)
            if hasattr(term, "_raw_actions"):
                term._raw_actions[:, :] = action
            return
        offset = next_offset
    raise KeyError(f"Action term not found: {term_name}")


class GoalkeeperEvalObsAdapter:
    """Compute goalkeeper API observations through the Eval-Goalkeeper env."""

    def __init__(self, env: Any, device: str | torch.device):
        self.env = env
        self.base_env = env.unwrapped
        self.device = torch.device(device)
        self.action_term = self.base_env.action_manager.get_term("joint_pos")

    def reset(self) -> None:
        self.env.reset()

    def compute_obs(self, raw_state: dict) -> torch.Tensor:
        self._sync_raw_state(raw_state)
        self._sync_last_action(raw_state["goalkeeper"]["last_action"])
        return self.base_env.observation_manager.compute_group(
            "actor", update_history=True,
        )

    def _sync_raw_state(self, raw_state: dict) -> None:
        goalkeeper = raw_state["goalkeeper"]
        shooter = raw_state["shooter"]
        ball = raw_state["ball"]
        robot = self.base_env.scene["robot"]
        ball_entity = self.base_env.scene["ball"]

        root_state = torch.cat([
            _as_batch(goalkeeper["root_pos"], self.device),
            _as_batch(goalkeeper["root_quat"], self.device),
            _as_batch(goalkeeper["root_lin_vel"], self.device),
            _as_batch(goalkeeper["root_ang_vel"], self.device),
        ], dim=-1)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(
            _as_batch(goalkeeper["joint_pos"], self.device),
            _as_batch(goalkeeper["joint_vel"], self.device),
        )
        robot.clear_state()

        if _has_scene_entity(self.base_env.scene, "opponent"):
            opponent = self.base_env.scene["opponent"]
            opponent_root_state = torch.cat([
                _as_batch(shooter["root_pos"], self.device),
                _as_batch(shooter["root_quat"], self.device),
                _as_batch(shooter["root_lin_vel"], self.device),
                _as_batch(shooter["root_ang_vel"], self.device),
            ], dim=-1)
            opponent.write_root_state_to_sim(opponent_root_state)
            opponent.write_joint_state_to_sim(
                _as_batch(shooter["joint_pos"], self.device),
                _as_batch(shooter["joint_vel"], self.device),
            )
            opponent.clear_state()

        ball_state = torch.cat([
            _as_batch(ball["pos"], self.device),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.device),
            _as_batch(ball["vel"], self.device),
            torch.zeros(1, 3, device=self.device),
        ], dim=-1)
        ball_entity.write_root_state_to_sim(ball_state)
        ball_entity.clear_state()

        self.base_env.scene.write_data_to_sim()
        self.base_env.sim.forward()
        self.base_env.sim.sense()

    def _sync_last_action(self, last_action: list[float]) -> None:
        action = _as_batch(last_action, self.device)
        _set_action_term(self.base_env.action_manager, "joint_pos", action)


def _load_policy(checkpoint_path: str, task_id: str, device: str) -> tuple[Any, Any, str]:
    """Build env from task config, load checkpoint, return inference policy."""
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()

    if task_id == "Eval-Goalkeeper" and _checkpoint_has_actor_key(checkpoint_path, "actor_residual.", device):
        task_id = _KEEPER_ADV_TASK_ID
        print(f"[INFO] Detected adversarial goalkeeper checkpoint; using task: {task_id}")

    env_cfg = load_env_cfg(task_id, play=False)
    env_cfg.scene.num_envs = 1
    env_base = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env_base, clip_actions=100.0)

    actor_terms = list(env_cfg.observations["actor"].terms.keys())
    history_len = env_cfg.observations["actor"].history_length
    print(f"[INFO] Task: {task_id}")
    print(f"[INFO] Actor obs  ({len(actor_terms)} terms × {history_len} history): {actor_terms}")
    print(f"[INFO] Action dim: {_action_dim(env)}")

    if task_id in ("Eval-Goalkeeper", _KEEPER_ADV_TASK_ID):
        from src.tasks.soccer.config.g1.rl_cfg import (
            AdversarialGoalkeeperRunner,
            GoalkeeperRunner,
            unitree_g1_goalkeeper_adversarial_ppo_runner_cfg,
            unitree_g1_goalkeeper_ppo_runner_cfg,
        )
        from src.tasks.soccer.adversarial import SingleRobotActionEnvView

        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        moe_bundle = _moe_bundle_from_checkpoint(loaded)
        if moe_bundle is not None:
            print("[INFO] Detected MoE region-specialist bundle.")
            idle_env = None
            if _moe_bundle_has_adversarial_idle(moe_bundle):
                idle_cfg = load_env_cfg(_KEEPER_ADV_TASK_ID, play=False)
                idle_cfg.scene.num_envs = 1
                idle_base = ManagerBasedRlEnv(cfg=idle_cfg, device=device)
                idle_env = SingleRobotActionEnvView(
                    RslRlVecEnvWrapper(idle_base, clip_actions=100.0),
                )
                print("[INFO] Detected adversarial prepare expert; using adversarial idle env.")
            policy = ApiMoE6Policy(moe_bundle, env, device, idle_env=idle_env)
            print(f"[INFO] Policy loaded from: {checkpoint_path}")
            return policy, env, task_id

        from scripts.eval_naive_goalkeeper import (
            _is_goalkeeper_student_checkpoint,
            _load_goalkeeper_student_policy,
        )

        if _is_goalkeeper_student_checkpoint(loaded):
            print("[INFO] Detected position-conditioned goalkeeper student checkpoint.")
            policy = _load_goalkeeper_student_policy(loaded, env, device)
            print(f"[INFO] Policy loaded from: {checkpoint_path}")
            return policy, env, task_id

        if task_id == _KEEPER_ADV_TASK_ID:
            print("[INFO] Detected adversarial goalkeeper ActorCritic checkpoint.")
            agent_cfg = unitree_g1_goalkeeper_adversarial_ppo_runner_cfg()
            runner = AdversarialGoalkeeperRunner(
                SingleRobotActionEnvView(env),
                asdict(agent_cfg),
                device=device,
            )
            runner.load(
                checkpoint_path,
                load_cfg={"actor": True, "critic": False, "optimizer": False, "iteration": False},
                map_location=device,
            )
        elif isinstance(loaded, dict) and "model_state_dict" in loaded:
            print("[INFO] Detected HIMPPO ActorCritic checkpoint — loading directly.")
            agent_cfg = unitree_g1_goalkeeper_ppo_runner_cfg()
            runner = GoalkeeperRunner(env, asdict(agent_cfg), device=device)
            actor_state = {
                k: v
                for k, v in loaded["model_state_dict"].items()
                if not k.startswith("critic.")
            }
            runner.alg.actor.load_state_dict(actor_state, strict=False)
        else:
            from mjlab.rl import MjlabOnPolicyRunner
            from src.tasks.soccer.config.g1.gk_train_cfg import goalkeeper_train_runner_cfg

            print("[INFO] Detected native goalkeeper MLP checkpoint.")
            runner = MjlabOnPolicyRunner(
                env,
                asdict(goalkeeper_train_runner_cfg()),
                device=device,
            )
            runner.load(checkpoint_path, load_cfg={"actor": True})
    else:
        from src.tasks.soccer.config.g1.rl_cfg import (
            SoccerRecurrentRunner,
            unitree_g1_soccer_recurrent_runner_cfg,
        )
        agent_cfg = unitree_g1_soccer_recurrent_runner_cfg()
        runner = SoccerRecurrentRunner(
            env, asdict(agent_cfg), log_dir=None, device=device,
        )
        runner.load(checkpoint_path)

    policy = runner.get_inference_policy(device=device)
    print(f"[INFO] Policy loaded from: {checkpoint_path}")
    return policy, env, task_id


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    checkpoint_path: str,
    task_id: str,
    device: str,
    motion_dir: str = _DEFAULT_MOTION_DIR,
    strategy: str = "gk-aware",
) -> FastAPI:
    """Build the FastAPI app with a loaded policy and motion data."""

    policy, env, task_id = _load_policy(checkpoint_path, task_id, device)
    is_gk = task_id in ("Eval-Goalkeeper", _KEEPER_ADV_TASK_ID)
    history_len = 10 if is_gk else 1
    goalkeeper_adapter = GoalkeeperEvalObsAdapter(env, device) if is_gk else None
    idle_env = getattr(policy, "idle_env", None)
    idle_goalkeeper_adapter = (
        GoalkeeperEvalObsAdapter(idle_env, device) if is_gk and idle_env is not None else None
    )

    # History buffer for observation stack.
    history: deque[torch.Tensor] = deque(maxlen=history_len)

    # Motion + destination state (re-sampled per episode).
    motions: list[MotionData] = []
    motion_ctx: dict[str, Any] = {
        "motion": None,
        "time_step": 0,
        "destination": (0.0, 0.0, 0.0),
    }

    if not is_gk:
        motions = _load_motions(motion_dir, device)
        motion_ctx["motion"] = motions[0]  # placeholder, overwritten on first /act

    # Track whether destination has been planned for current episode.
    shoot_planned: dict[str, bool] = {"done": False}
    gk_y_history: list[float] = []  # gk-aware observation buffer

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print(f"[INFO] Server ready — {task_id} policy on {device}  (strategy={strategy})")
        yield
        env.close()
        print("[INFO] Server shutting down.")

    app = FastAPI(title=f"CS2810 Phase 2 — {task_id}", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/act", response_model=ActResponse)
    async def act(req: dict):
        nonlocal shoot_planned, gk_y_history
        raw_state = req
        if is_gk:
            assert goalkeeper_adapter is not None
            stacked = goalkeeper_adapter.compute_obs(raw_state)
            if idle_goalkeeper_adapter is not None:
                idle_goalkeeper_adapter.compute_obs(raw_state)
        else:
            if strategy == "gk-aware" and not shoot_planned["done"]:
                gk_root = raw_state["goalkeeper"]["root_pos"]
                gk_y_history.append(gk_root[1])
                if _should_lock(gk_y_history, motion_ctx["motion"].name):
                    avg_gk_y = sum(gk_y_history) / len(gk_y_history)
                    dest_y = _plan_destination(avg_gk_y)
                    motion_ctx["destination"] = (-0.5, dest_y, 0.11)
                    shoot_planned["done"] = True
            m = motion_ctx["motion"]
            frame = compute_shooter_obs(raw_state, m, motion_ctx["time_step"], motion_ctx["destination"], device)
            motion_ctx["time_step"] += 1

            if len(history) == 0:
                for _ in range(history_len):
                    history.append(frame.clone())

            history.append(frame)
            stacked = torch.cat(list(history), dim=-1)

        with torch.inference_mode():
            if isinstance(policy, ApiMoE6Policy):
                action = policy({"actor": stacked}, raw_state)
            else:
                action = policy({"actor": stacked})

        return ActResponse(action=action.cpu().tolist())

    @app.post("/reset")
    async def reset():
        nonlocal shoot_planned, gk_y_history
        if is_gk:
            policy.reset()
            assert goalkeeper_adapter is not None
            goalkeeper_adapter.reset()
            if idle_goalkeeper_adapter is not None:
                idle_goalkeeper_adapter.reset()
        else:
            motion_ctx["time_step"] = 0
            if strategy == "uniform":
                motion_ctx["motion"] = random.choice(motions)
                dest_y = random.uniform(-1.4, 1.4)
                motion_ctx["destination"] = (-0.5, dest_y, 0.11)
                shoot_planned["done"] = True
            elif strategy == "best-motion":
                dest_y = random.uniform(-1.4, 1.4)
                motion_ctx["destination"] = (-0.5, dest_y, 0.11)
                motion_ctx["motion"] = _select_motion_for_dest(dest_y, motions)
                shoot_planned["done"] = True
            else:  # gk-aware
                right_motions = [m for m in motions if "right" in m.name]
                motion_ctx["motion"] = random.choice(right_motions) if right_motions else motions[0]
                motion_ctx["destination"] = (-0.5, 0.0, 0.11)
                shoot_planned["done"] = False
                gk_y_history.clear()
            policy.reset()
        history.clear()
        return {"status": "ok"}

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@dataclass
class ServerConfig:
    checkpoint: str
    """Path to the policy checkpoint (.pt)."""
    port: int = 8000
    """Port to listen on."""
    task: str = "shooter"
    """Task type: 'shooter' or 'goalkeeper'."""
    host: str = "0.0.0.0"
    """Host to bind to."""
    device: str | None = None
    """Torch device (auto-detected if omitted)."""
    motion_dir: str = _DEFAULT_MOTION_DIR
    """Directory containing soccer-standard-*.npz motion files."""
    strategy: Literal["uniform", "best-motion", "gk-aware"] = "gk-aware"
    """Shooting strategy:
       - uniform:    随机 motion + 随机目标 (原始行为)
       - best-motion: 随机目标 + 该目标最优 right-leg motion
       - gk-aware:    根据 GK 位置避开射远角 + 最优 motion (默认)"""


def main():
    import src.tasks  # noqa: F401  — register eval tasks

    args = tyro.cli(ServerConfig, prog="api_server")

    task_id = "Eval-Shooter-Stage6" if args.task == "shooter" else "Eval-Goalkeeper"
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    app = create_app(args.checkpoint, task_id, device, args.motion_dir, args.strategy)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
