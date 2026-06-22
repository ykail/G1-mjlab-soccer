"""Position-conditioned AMP discriminator for goalkeeper training."""

from __future__ import annotations

import torch
import torch.nn as nn


DISC_LOGIT_INIT_SCALE = 1.0


def _activation(name: str) -> nn.Module:
  if name == "elu":
    return nn.ELU()
  if name == "selu":
    return nn.SELU()
  if name == "lrelu":
    return nn.LeakyReLU()
  if name == "tanh":
    return nn.Tanh()
  if name == "sigmoid":
    return nn.Sigmoid()
  return nn.ReLU()


class RunningNormalizer(nn.Module):
  """Torch-native running normalizer for AMP states."""

  def __init__(self, dim: int, eps: float = 1.0e-5) -> None:
    super().__init__()
    self.eps = eps
    self.register_buffer("count", torch.tensor(eps, dtype=torch.float32))
    self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
    self.register_buffer("var", torch.ones(dim, dtype=torch.float32))

  @torch.no_grad()
  def update(self, values: torch.Tensor) -> None:
    if values.numel() == 0:
      return
    values = values.detach()
    batch_mean = values.mean(dim=0)
    batch_var = values.var(dim=0, unbiased=False)
    batch_count = torch.tensor(
      values.shape[0],
      dtype=self.count.dtype,
      device=self.count.device,
    )

    delta = batch_mean - self.mean
    total_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / total_count
    m_a = self.var * self.count
    m_b = batch_var * batch_count
    correction = delta.square() * self.count * batch_count / total_count
    new_var = (m_a + m_b + correction) / total_count

    self.mean.copy_(new_mean)
    self.var.copy_(torch.clamp(new_var, min=self.eps))
    self.count.copy_(total_count)

  def normalize(self, values: torch.Tensor) -> torch.Tensor:
    return (values - self.mean.to(values.device)) / torch.sqrt(
      self.var.to(values.device) + self.eps
    )


def _make_discriminator(
  state_dim: int,
  hidden_dims: tuple[int, ...],
  activation: str,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  input_dim = state_dim
  act = _activation(activation)
  for layer_index, hidden_dim in enumerate(hidden_dims):
    layer = nn.Linear(input_dim, hidden_dim)
    if layer_index > 0:
      nn.init.uniform_(layer.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
      nn.init.zeros_(layer.bias)
    layers.append(layer)
    layers.append(act)
    input_dim = hidden_dim
  out = nn.Linear(input_dim, 1)
  nn.init.uniform_(out.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
  nn.init.zeros_(out.bias)
  layers.append(out)
  return nn.Sequential(*layers)


class GoalkeeperAMP(nn.Module):
  """Six region-specific AMP discriminators."""

  def __init__(
    self,
    num_regions: int,
    state_dim: int,
    hidden_dims: tuple[int, ...] = (512, 256),
    activation: str = "relu",
    grad_penalty_coef: float = 0.5,
    device: str | torch.device = "cpu",
  ) -> None:
    super().__init__()
    self.num_regions = num_regions
    self.state_dim = state_dim
    self.grad_penalty_coef = grad_penalty_coef
    self.discriminators = nn.ModuleList(
      [
        _make_discriminator(state_dim, hidden_dims, activation)
        for _ in range(num_regions)
      ]
    )
    self.normalizer = RunningNormalizer(state_dim)
    self.to(device)

  def _logits_by_region(
    self,
    states: torch.Tensor,
    regions: torch.Tensor,
  ) -> torch.Tensor:
    regions = torch.clamp(
      regions.to(device=states.device, dtype=torch.long),
      0,
      self.num_regions - 1,
    )
    logits = torch.empty(states.shape[0], 1, dtype=states.dtype, device=states.device)
    for region in torch.unique(regions):
      mask = regions == region
      logits[mask] = self.discriminators[int(region.item())](states[mask])
    return logits

  def forward(self, states: torch.Tensor, regions: torch.Tensor) -> torch.Tensor:
    return self._logits_by_region(states, regions)

  def compute_loss(
    self,
    policy_states: torch.Tensor,
    expert_states: torch.Tensor,
    regions: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    policy_norm = self.normalizer.normalize(policy_states)
    expert_norm = self.normalizer.normalize(expert_states)

    expert_terms = []
    policy_terms = []
    grad_terms = []
    active_regions = torch.unique(regions.to(device=policy_norm.device, dtype=torch.long))
    for region in active_regions:
      mask = regions.to(device=policy_norm.device, dtype=torch.long) == region
      region_id = int(region.item())
      policy_logits = self.discriminators[region_id](policy_norm[mask])
      expert_logits = self.discriminators[region_id](expert_norm[mask])
      expert_terms.append((expert_logits - 1.0).square().mean())
      policy_terms.append((policy_logits + 1.0).square().mean())
      grad_terms.append(self._expert_grad_penalty_for_region(expert_norm[mask], region_id))

    expert_loss = torch.stack(expert_terms).sum()
    policy_loss = torch.stack(policy_terms).sum()
    grad_penalty = torch.stack(grad_terms).sum()
    total = expert_loss + policy_loss + self.grad_penalty_coef * grad_penalty
    self.normalizer.update(torch.cat([policy_states.detach(), expert_states.detach()], dim=0))
    return {
      "total": total,
      "expert": expert_loss,
      "policy": policy_loss,
      "grad_penalty": grad_penalty,
    }

  def _expert_grad_penalty_for_region(
    self,
    expert_states: torch.Tensor,
    region_id: int,
  ) -> torch.Tensor:
    expert_states = expert_states.detach().clone().requires_grad_(True)
    logits = self.discriminators[region_id](expert_states)
    grad = torch.autograd.grad(
      outputs=logits,
      inputs=expert_states,
      grad_outputs=torch.ones_like(logits),
      create_graph=True,
      retain_graph=True,
      only_inputs=True,
    )[0]
    return grad.square().sum(dim=-1).mean()

  @torch.no_grad()
  def predict_reward(
    self,
    states: torch.Tensor,
    regions: torch.Tensor,
    num_samples: int = 20,
    sigma: float = 0.3,
  ) -> torch.Tensor:
    was_training = self.training
    self.eval()
    norm = self.normalizer.normalize(states)
    if num_samples <= 1 or sigma <= 0.0:
      logits = self._logits_by_region(norm, regions)
      reward = torch.clamp(1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
      self.train(was_training)
      return reward

    noise = torch.randn(
      norm.shape[0],
      num_samples,
      norm.shape[1],
      dtype=norm.dtype,
      device=norm.device,
    ) * sigma
    perturbed = norm.unsqueeze(1) + noise
    flat_states = perturbed.reshape(-1, norm.shape[1])
    flat_regions = regions.to(device=norm.device).repeat_interleave(num_samples)
    logits = self._logits_by_region(flat_states, flat_regions).reshape(
      norm.shape[0],
      num_samples,
    )
    best_error = (logits - 1.0).square().min(dim=1).values
    reward = torch.clamp(1.0 - 0.25 * best_error, min=0.0).unsqueeze(-1)
    self.train(was_training)
    return reward
