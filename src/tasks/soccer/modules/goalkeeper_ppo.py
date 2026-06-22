"""Goalkeeper PPO with estimator supervision and AMP motion-prior rewards."""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from src.tasks.soccer.modules.goalkeeper_amp import GoalkeeperAMP
from src.tasks.soccer.modules.goalkeeper_motion_prior import (
  GoalkeeperMotionPrior,
  _region_from_observations,
  build_amp_state_from_observations,
)


_CRITIC_BALL_VEL = slice(99, 102)
_CRITIC_END_TARGET = slice(109, 112)
_CRITIC_REGION = 112


class GoalkeeperPPO(PPO):
  """PPO variant that trains deployable estimator heads plus AMP."""

  def __init__(
    self,
    *args,
    estimator_ball_loss_coef: float = 1.0,
    estimator_region_loss_coef: float = 1.0,
    value_smoothness_coef: float = 0.1,
    smoothness_upper_bound: float = 1.0,
    smoothness_lower_bound: float = 0.1,
    amp_cfg: dict | None = None,
    **kwargs,
  ) -> None:
    super().__init__(*args, **kwargs)
    self.estimator_ball_loss_coef = estimator_ball_loss_coef
    self.estimator_region_loss_coef = estimator_region_loss_coef
    self.value_smoothness_coef = value_smoothness_coef
    self.smoothness_upper_bound = smoothness_upper_bound
    self.smoothness_lower_bound = smoothness_lower_bound
    self.amp_cfg = amp_cfg or {"enabled": False}
    self.amp: GoalkeeperAMP | None = None
    self.amp_motion_prior: GoalkeeperMotionPrior | None = None
    self.amp_reward_mode = "mix"
    self.amp_reward_coef = 0.0
    self.amp_reward_dt = 1.0
    self.amp_disc_loss_coef = 1.0
    self.amp_num_reward_samples = 20
    self.amp_reward_sigma = 0.3
    self.amp_reward_scale = 0.5
    self._pending_next_obs = None
    self._pending_amp_not_done = None
    self._last_amp_reward_mean = 0.0
    if self.amp_cfg.get("enabled", bool(self.amp_cfg)):
      self._init_amp(self.amp_cfg)

  def _uses_smoothness(self) -> bool:
    return self.smoothness_upper_bound > 0.0 and self.smoothness_lower_bound > 0.0

  def _init_amp(self, amp_cfg: dict) -> None:
    motion_dir = amp_cfg.get("motion_dir")
    if not motion_dir:
      raise ValueError("amp_cfg.motion_dir must point to goalkeeper motion priors")
    self.amp_motion_prior = GoalkeeperMotionPrior(motion_dir, device=self.device)
    self.amp = GoalkeeperAMP(
      num_regions=self.amp_motion_prior.num_regions,
      state_dim=self.amp_motion_prior.transition_dim,
      hidden_dims=tuple(amp_cfg.get("hidden_dims", (512, 256))),
      activation=amp_cfg.get("activation", "relu"),
      grad_penalty_coef=float(amp_cfg.get("grad_penalty_coef", 0.5)),
      device=self.device,
    )
    self.optimizer.add_param_group(
      {
        "params": self.amp.parameters(),
        "lr": float(amp_cfg.get("learning_rate", self.learning_rate)),
        "weight_decay": float(amp_cfg.get("weight_decay", 1.0e-4)),
        "is_amp": True,
      }
    )
    self.amp_reward_coef = float(amp_cfg.get("reward_coef", 0.3))
    self.amp_reward_dt = float(amp_cfg.get("reward_dt", 1.0))
    self.amp_reward_mode = str(amp_cfg.get("reward_mode", "mix"))
    self.amp_disc_loss_coef = float(amp_cfg.get("disc_loss_coef", 1.0))
    self.amp_num_reward_samples = int(amp_cfg.get("num_reward_samples", 20))
    self.amp_reward_sigma = float(amp_cfg.get("reward_sigma", 0.3))
    self.amp_reward_scale = float(amp_cfg.get("reward_scale", 0.5))

  @staticmethod
  def _extract_tensor(obs, group: str) -> torch.Tensor:
    if isinstance(obs, dict):
      return obs[group]
    if hasattr(obs, "get"):
      return obs.get(group)
    return getattr(obs, group)

  def _estimator_targets_from_observations(
    self,
    observations,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ball(pos+vel) and region labels from the current critic obs."""
    critic_obs = self._extract_tensor(observations, "critic")
    ball_pos = critic_obs[:, _CRITIC_END_TARGET]
    ball_vel = critic_obs[:, _CRITIC_BALL_VEL]
    region = torch.clamp(torch.round(critic_obs[:, _CRITIC_REGION] * 3.0), 0, 5).long()
    return torch.cat([ball_pos, ball_vel], dim=-1), region

  def _estimator_loss(self, observations) -> dict[str, torch.Tensor]:
    actor_obs = self._extract_tensor(observations, "actor")
    ball_target, region_target = self._estimator_targets_from_observations(observations)
    return self.actor.compute_estimator_loss(
      actor_obs,
      ball_target=ball_target,
      region_target=region_target,
      ball_loss_coef=self.estimator_ball_loss_coef,
      region_loss_coef=self.estimator_region_loss_coef,
    )

  def process_env_step(
    self,
    obs,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    self._pending_next_obs = obs
    needs_next_obs = self.amp is not None or self._uses_smoothness()
    if needs_next_obs and not hasattr(self.storage, "next_observations"):
      self.storage.next_observations = self.storage.observations.clone()
    if self.amp is not None and not hasattr(self.storage, "amp_not_done"):
      self.storage.amp_not_done = torch.zeros_like(
        self.storage.dones,
        dtype=torch.float32,
      )
    if self.amp is not None:
      amp_not_done = (1.0 - dones.float()).view(-1)
      self._pending_amp_not_done = amp_not_done
      self.storage.amp_not_done[self.storage.step].copy_(amp_not_done.view(-1, 1))
    if needs_next_obs:
      self.storage.next_observations[self.storage.step].copy_(obs)

    self.transition.rewards = rewards.clone()
    self._augment_transition_with_amp_reward()
    rewards = self.transition.rewards
    super().process_env_step(obs, rewards, dones, extras)
    self._pending_next_obs = None
    self._pending_amp_not_done = None

  def _build_amp_state(self, observations, next_observations) -> torch.Tensor:
    if self.amp_motion_prior is None:
      raise RuntimeError("AMP motion prior is not initialized")
    return build_amp_state_from_observations(
      observations,
      next_observations,
      joint_indices=self.amp_motion_prior.full_joint_indices,
      default_joint_pos=self.amp_motion_prior.full_default_joint_pos,
    )

  def _augment_transition_with_amp_reward(self) -> None:
    if (
      self.amp is None
      or self._pending_next_obs is None
      or self.transition.observations is None
      or self.transition.rewards is None
    ):
      return
    amp_state = self._build_amp_state(self.transition.observations, self._pending_next_obs)
    regions = _region_from_observations(self.transition.observations)
    amp_reward = self.amp.predict_reward(
      amp_state,
      regions,
      num_samples=self.amp_num_reward_samples,
      sigma=self.amp_reward_sigma,
    ).squeeze(-1) * self.amp_reward_scale
    if self.transition.rewards.ndim == 2 and self.transition.rewards.shape[-1] == 1:
      amp_reward = amp_reward.unsqueeze(-1)
    valid_mask = None
    if self._pending_amp_not_done is not None:
      valid_mask = self._pending_amp_not_done.to(
        device=amp_reward.device,
        dtype=amp_reward.dtype,
      )
      if amp_reward.ndim == 2:
        valid_mask = valid_mask.unsqueeze(-1)
      amp_reward = amp_reward * valid_mask
    if self.amp_reward_mode == "mix":
      raw_rewards = self.transition.rewards
      mixed_rewards = (
        (1.0 - self.amp_reward_coef) * raw_rewards
        + self.amp_reward_coef * amp_reward
      )
      if valid_mask is not None:
        self.transition.rewards = torch.where(valid_mask > 0.5, mixed_rewards, raw_rewards)
      else:
        self.transition.rewards = mixed_rewards
    else:
      self.transition.rewards += self.amp_reward_coef * self.amp_reward_dt * amp_reward
    self._last_amp_reward_mean = float(amp_reward.mean().item())

  def _mini_batch_generator_with_next_obs(
    self,
    num_mini_batches: int,
    num_epochs: int = 8,
  ):
    """Yield RSL-RL-style mini-batches while preserving next-observation pairs."""
    batch_size = self.storage.num_envs * self.storage.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(
      num_mini_batches * mini_batch_size,
      requires_grad=False,
      device=self.storage.device,
    )

    observations = self.storage.observations.flatten(0, 1)
    next_observations = self.storage.next_observations.flatten(0, 1)
    amp_not_done = getattr(self.storage, "amp_not_done", None)
    if amp_not_done is not None:
      amp_not_done = amp_not_done.flatten(0, 1)
    actions = self.storage.actions.flatten(0, 1)
    values = self.storage.values.flatten(0, 1)
    returns = self.storage.returns.flatten(0, 1)
    old_actions_log_prob = self.storage.actions_log_prob.flatten(0, 1)
    advantages = self.storage.advantages.flatten(0, 1)
    old_distribution_params = tuple(
      p.flatten(0, 1) for p in self.storage.distribution_params
    )

    storage_cls = self.storage.__class__
    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        batch = storage_cls.Batch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
        )
        batch.next_observations = next_observations[batch_idx]
        if amp_not_done is not None:
          batch.amp_not_done = amp_not_done[batch_idx]
        yield batch

  def _amp_discriminator_loss(self, observations, next_observations, amp_not_done=None):
    if self.amp is None or self.amp_motion_prior is None:
      zero = torch.tensor(0.0, device=self.device)
      return {"total": zero, "expert": zero, "policy": zero, "grad_penalty": zero}
    if amp_not_done is not None:
      valid = amp_not_done.view(-1).to(device=self.device) > 0.5
      if not torch.any(valid):
        zero = torch.tensor(0.0, device=self.device)
        return {"total": zero, "expert": zero, "policy": zero, "grad_penalty": zero}
      observations = observations[valid]
      next_observations = next_observations[valid]
    policy_state = self._build_amp_state(observations, next_observations)
    regions = _region_from_observations(observations)
    expert_state = self.amp_motion_prior.sample_expert_transitions(regions)
    return self.amp.compute_loss(policy_state.detach(), expert_state, regions)

  def _smoothness_loss(
    self,
    observations,
    next_observations,
    action_mean: torch.Tensor,
    values: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """HIMPPO policy/value smoothness on interpolated observations."""
    actor_obs = self._extract_tensor(observations, "actor")
    next_actor_obs = self._extract_tensor(next_observations, "actor")
    critic_obs = self._extract_tensor(observations, "critic")
    next_critic_obs = self._extract_tensor(next_observations, "critic")
    if valid_mask is not None:
      valid = valid_mask.view(-1).to(device=actor_obs.device) > 0.5
      if torch.any(valid):
        actor_obs = actor_obs[valid]
        next_actor_obs = next_actor_obs[valid]
        critic_obs = critic_obs[valid]
        next_critic_obs = next_critic_obs[valid]
        action_mean = action_mean[valid]
        values = values[valid]
      else:
        zero = torch.tensor(0.0, device=actor_obs.device)
        return {"total": zero, "policy": zero, "value": zero}

    denom = max(
      self.smoothness_upper_bound - self.smoothness_lower_bound,
      1.0e-6,
    )
    epsilon = self.smoothness_lower_bound / denom
    policy_coef = self.smoothness_upper_bound * epsilon
    value_coef = self.value_smoothness_coef * policy_coef

    mix_weights = (torch.rand_like(actor_obs[:, :1]) - 0.5) * 2.0
    mix_obs = actor_obs + mix_weights * (next_actor_obs - actor_obs)
    mix_critic_obs = critic_obs + mix_weights * (next_critic_obs - critic_obs)
    mixed_observations = {
      "actor": mix_obs,
      "critic": mix_critic_obs,
    }
    mixed_mean = self.actor.act_inference(mix_obs)
    mixed_values = self.critic(mixed_observations)
    policy_loss = torch.square(torch.norm(action_mean - mixed_mean, dim=-1)).mean()
    value_loss = torch.square(torch.norm(values - mixed_values, dim=-1)).mean()
    total = policy_coef * policy_loss + value_coef * value_loss
    return {"total": total, "policy": policy_loss, "value": value_loss}

  @staticmethod
  def _batch_hidden(batch, index: int):
    hidden = getattr(batch, "hidden_states", (None, None))
    if hidden is None:
      return None
    return hidden[index]

  def update(self) -> dict[str, float]:
    """Run PPO update plus estimator, smoothness, and AMP losses."""
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_estimator_ball_loss = 0.0
    mean_estimator_region_loss = 0.0
    mean_estimator_loss = 0.0
    mean_amp_discriminator_loss = 0.0
    mean_amp_expert_loss = 0.0
    mean_amp_policy_loss = 0.0
    mean_amp_grad_penalty = 0.0
    mean_policy_smoothness_loss = 0.0
    mean_value_smoothness_loss = 0.0
    mean_smoothness_loss = 0.0
    amp_update_count = 0
    mean_rnd_loss = 0.0 if self.rnd else None
    mean_symmetry_loss = 0.0 if self.symmetry else None

    if (
      (self.amp is not None or self._uses_smoothness())
      and not (self.actor.is_recurrent or self.critic.is_recurrent)
      and hasattr(self.storage, "next_observations")
    ):
      generator = self._mini_batch_generator_with_next_obs(
        self.num_mini_batches,
        self.num_learning_epochs,
      )
    elif self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches,
        self.num_learning_epochs,
      )

    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]

      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            (batch.advantages - batch.advantages.mean())
            / (batch.advantages.std() + 1.0e-8)
          )

      if self.symmetry:
        self.symmetry.augment_batch(batch, original_batch_size)

      self.actor(
        batch.observations,
        masks=getattr(batch, "masks", None),
        hidden_state=self._batch_hidden(batch, 0),
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations,
        masks=getattr(batch, "masks", None),
        hidden_state=self._batch_hidden(batch, 1),
      )
      distribution_params = tuple(
        p[:original_batch_size] for p in self.actor.output_distribution_params
      )
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params,
            distribution_params,
          )
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()
          for param_group in self.optimizer.param_groups:
            if not param_group.get("is_amp", False):
              param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio,
        1.0 - self.clip_param,
        1.0 + self.clip_param,
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param,
          self.clip_param,
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      estimator_losses = self._estimator_loss(batch.observations[:original_batch_size])
      loss = loss + estimator_losses["total"]

      smoothness_losses = None
      next_observations_for_smoothness = getattr(batch, "next_observations", None)
      if next_observations_for_smoothness is not None:
        smoothness_valid = getattr(batch, "amp_not_done", None)
        smoothness_losses = self._smoothness_loss(
          batch.observations[:original_batch_size],
          next_observations_for_smoothness[:original_batch_size],
          distribution_params[0],
          values[:original_batch_size],
          valid_mask=(
            smoothness_valid[:original_batch_size]
            if smoothness_valid is not None
            else None
          ),
        )
        loss = loss + smoothness_losses["total"]

      amp_losses = None
      if self.amp is not None:
        next_observations = getattr(batch, "next_observations", None)
        if next_observations is not None:
          amp_not_done = getattr(batch, "amp_not_done", None)
          amp_losses = self._amp_discriminator_loss(
            batch.observations[:original_batch_size],
            next_observations[:original_batch_size],
            amp_not_done=(
              amp_not_done[:original_batch_size]
              if amp_not_done is not None
              else None
            ),
          )
          loss = loss + self.amp_disc_loss_coef * amp_losses["total"]

      rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None
      if self.symmetry:
        symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
        if self.symmetry.use_mirror_loss:
          loss = loss + self.symmetry.mirror_loss_coeff * symmetry_loss

      self.optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        rnd_loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      if self.amp is not None:
        nn.utils.clip_grad_norm_(self.amp.parameters(), self.max_grad_norm)
      self.optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_estimator_ball_loss += estimator_losses["ball"].item()
      mean_estimator_region_loss += estimator_losses["region"].item()
      mean_estimator_loss += estimator_losses["total"].item()
      if smoothness_losses is not None:
        mean_policy_smoothness_loss += smoothness_losses["policy"].item()
        mean_value_smoothness_loss += smoothness_losses["value"].item()
        mean_smoothness_loss += smoothness_losses["total"].item()
      if amp_losses is not None:
        mean_amp_discriminator_loss += amp_losses["total"].item()
        mean_amp_expert_loss += amp_losses["expert"].item()
        mean_amp_policy_loss += amp_losses["policy"].item()
        mean_amp_grad_penalty += amp_losses["grad_penalty"].item()
        amp_update_count += 1
      if mean_rnd_loss is not None:
        mean_rnd_loss += rnd_loss.item()
      if mean_symmetry_loss is not None:
        mean_symmetry_loss += symmetry_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_estimator_ball_loss /= num_updates
    mean_estimator_region_loss /= num_updates
    mean_estimator_loss /= num_updates
    mean_policy_smoothness_loss /= num_updates
    mean_value_smoothness_loss /= num_updates
    mean_smoothness_loss /= num_updates
    if mean_rnd_loss is not None:
      mean_rnd_loss /= num_updates
    if mean_symmetry_loss is not None:
      mean_symmetry_loss /= num_updates
    if amp_update_count > 0:
      mean_amp_discriminator_loss /= amp_update_count
      mean_amp_expert_loss /= amp_update_count
      mean_amp_policy_loss /= amp_update_count
      mean_amp_grad_penalty /= amp_update_count

    self.storage.clear()

    loss_dict = {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "estimator_ball": mean_estimator_ball_loss,
      "estimator_region": mean_estimator_region_loss,
      "estimator": mean_estimator_loss,
      "policy_smoothness": mean_policy_smoothness_loss,
      "value_smoothness": mean_value_smoothness_loss,
      "smoothness": mean_smoothness_loss,
    }
    if self.rnd:
      loss_dict["rnd"] = mean_rnd_loss
    if self.symmetry:
      loss_dict["symmetry"] = mean_symmetry_loss
    if self.amp is not None:
      loss_dict["amp_discriminator"] = mean_amp_discriminator_loss
      loss_dict["amp_expert"] = mean_amp_expert_loss
      loss_dict["amp_policy"] = mean_amp_policy_loss
      loss_dict["amp_grad_penalty"] = mean_amp_grad_penalty
      loss_dict["amp_reward"] = self._last_amp_reward_mean
    return loss_dict

  def train_mode(self) -> None:
    super().train_mode()
    if self.amp is not None:
      self.amp.train()

  def eval_mode(self) -> None:
    super().eval_mode()
    if self.amp is not None:
      self.amp.eval()

  def save(self) -> dict:
    saved = super().save()
    if self.amp is not None:
      saved["amp_state_dict"] = self.amp.state_dict()
    return saved

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    iteration = super().load(loaded_dict, load_cfg, strict)
    load_amp = load_cfg is None or load_cfg.get("amp", True)
    if load_amp and self.amp is not None and "amp_state_dict" in loaded_dict:
      self.amp.load_state_dict(loaded_dict["amp_state_dict"], strict=strict)
    return iteration

  def broadcast_parameters(self) -> None:
    super().broadcast_parameters()
    if self.amp is None:
      return
    amp_state = [self.amp.state_dict()]
    torch.distributed.broadcast_object_list(amp_state, src=0)
    self.amp.load_state_dict(amp_state[0])

  def reduce_parameters(self) -> None:
    super().reduce_parameters()
    if self.amp is None:
      return
    params = [param for param in self.amp.parameters() if param.grad is not None]
    if not params:
      return
    all_grads = torch.cat([param.grad.view(-1) for param in params])
    torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
    all_grads /= self.gpu_world_size
    offset = 0
    for param in params:
      numel = param.numel()
      param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
      offset += numel
