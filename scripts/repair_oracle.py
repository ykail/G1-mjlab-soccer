"""CEM repair oracle for the goalkeeper.

Freezes a competent closed-loop BASE policy (the reference teacher) and searches,
per failing scenario, an OPEN-LOOP residual action sequence r_t such that
    a_t = clip(base(o_t) + r_t)
blocks the ball. The base provides stabilizing closed-loop control; the residual
nudges timing/reach onto the ball. Because the teacher is bottlenecked by high
balls (see diagnose_gk.py), search can EXCEED it — that's the whole point.

Layout: N = G scenarios x P population. All P envs of a scenario share an
identical ball trajectory + pinned robot init, but get P different residual
sequences = the CEM population. Cost is dense (min blocking-link distance to the
ball) so the search has gradient even before the first block.

Run modes:
  --mode prove   : optimize a set of regions in-sample, report base vs repaired.
  --mode collect : optimize a large scenario sweep, dump (obs, action) dataset.
"""
from __future__ import annotations
import copy, os, sys
from dataclasses import dataclass
from pathlib import Path
import torch, tyro

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

import mjlab.tasks  # noqa: F401
import src.tasks.soccer.config.eval  # noqa: F401

_REGION = ["Right-Mid", "Left-Mid", "Right-Up", "Left-Up", "Right-Low", "Left-Low"]
_BLOCK_LINKS = (
  "left_wrist_yaw_link", "right_wrist_yaw_link", "left_elbow_link", "right_elbow_link",
  "left_shoulder_roll_link", "right_shoulder_roll_link", "left_ankle_roll_link",
  "right_ankle_roll_link", "left_knee_link", "right_knee_link", "torso_link", "pelvis",
)


@dataclass
class Cfg:
  checkpoint: str = "src/assets/soccer/weight/goalkeeper.pt"  # base policy (teacher)
  mode: str = "prove"
  regions: tuple[int, ...] = (3, 2, 1, 0)   # which regions to repair (default: hard ones)
  region_weights: tuple[float, ...] = ()
  G: int = 16                # scenarios per batch
  P: int = 64                # CEM population per scenario  (N = G*P)
  iters: int = 8             # CEM iterations
  elites: int = 8
  knots: int = 12
  knot_span: int = 80        # knots concentrated in steps [0, knot_span] (save window)
  release_steps: int = 20    # fade residual back to base after knot_span
  horizon: int = 150
  clip: float = 0.45         # residual clamp (rad)
  init_std: float = 0.25
  smooth: float = 0.7        # CEM mean/std update smoothing
  batches: int = 1           # how many G-scenario batches
  seed: int = 0
  device: str = "cuda:0"
  out: str = ""              # collect: path to save dataset
  w_dist: float = 60.0       # cost weight on min blocking-link distance
  w_goal: float = 1000.0     # cost weight on conceding
  w_res: float = 0.2         # cost weight on residual L2
  w_upright: float = 5.0     # cost weight on not-upright at the contact window
  w_stable: float = 20.0     # cost weight on whole-episode tilt after aggressive saves
  w_final_upright: float = 20.0
  collect_pre_steps: int = 35
  collect_post_steps: int = 12
  require_final_upright: bool = False
  final_upright_gz: float = -0.25


def _gen_scenarios(env, regions, region_weights, G, gen, device):
  """Sample G ball scenarios (region pinned per scenario) exactly as the env does."""
  rb = env.unwrapped.cfg.events["reset_ball"]
  vc = rb.params["vel_cfg"]
  region_ids = torch.tensor(regions, dtype=torch.long, device=device)
  if region_weights:
    if len(region_weights) != len(regions):
      raise ValueError(
        f"region_weights length {len(region_weights)} must match regions length {len(regions)}"
      )
    weights = torch.tensor(region_weights, dtype=torch.float32, device=device)
    if torch.any(weights < 0.0) or float(weights.sum()) <= 0.0:
      raise ValueError("region_weights must be non-negative with positive sum")
    pick = torch.multinomial(weights, G, replacement=True, generator=gen)
  else:
    pick = torch.randint(0, len(regions), (G,), generator=gen, device=device)
  reg = region_ids[pick]
  u = lambda lo, hi: lo + torch.rand(G, generator=gen, device=device) * (hi - lo)
  start_x = u(*vc.ball_start_x_range)
  start_y = u(*vc.ball_start_y_range)
  start_z = u(*vc.ball_start_z_range)
  hl = torch.tensor([vc.regions[int(r)]["height"][0] for r in reg], device=device)
  hh = torch.tensor([vc.regions[int(r)]["height"][1] for r in reg], device=device)
  wl = torch.tensor([vc.regions[int(r)]["width"][0] for r in reg], device=device)
  wh = torch.tensor([vc.regions[int(r)]["width"][1] for r in reg], device=device)
  end_x = -u(*vc.ball_end_x_range)
  end_y = wl + torch.rand(G, generator=gen, device=device) * (wh - wl)
  end_z = hl + torch.rand(G, generator=gen, device=device) * (hh - hl)
  t_flight = u(*vc.t_flight_range)
  start = torch.stack([start_x, start_y, start_z], -1)
  end = torch.stack([end_x, end_y, end_z], -1)
  return reg, start, end, t_flight


def _apply(env, robot, ball, start, vel, P, region):
  """Force the (expanded) ball scenario THROUGH a normal env.reset() so the
  observation history is filled correctly (no stale-frame transient). Robot init
  uses the env's normal reset (eval-matched). Returns fresh obs."""
  N = env.unwrapped.num_envs
  org = env.unwrapped.scene.env_origins
  start_w = start.repeat_interleave(P, 0) + org
  vel_e = vel.repeat_interleave(P, 0)
  reg_e = region.repeat_interleave(P, 0)
  env.unwrapped._gk_forced = {"start": start_w, "vel": vel_e, "region": reg_e}
  obs, _ = env.reset()
  return obs


def _reset_policy(policy) -> None:
  reset = getattr(policy, "reset", None)
  if reset is None:
    return
  try:
    reset()
  except TypeError:
    reset(None)


def _ball_vel(start, end, t_flight):
  g = 9.81
  vxy = (end[:, :2] - start[:, :2]) / t_flight[:, None]
  vz = ((end[:, 2] - start[:, 2]) + 0.5 * g * t_flight ** 2) / t_flight
  return torch.cat([vxy, vz[:, None]], -1)


def main(cfg: Cfg):
  configure_torch_backends()
  dev = cfg.device
  N = cfg.G * cfg.P
  env_cfg = load_env_cfg("Eval-Goalkeeper", play=False)
  env_cfg.scene.num_envs = N
  if "fell_over" in env_cfg.terminations:
    env_cfg.terminations["fell_over"] = None
  env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=dev), clip_actions=100.0)

  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from eval_naive_goalkeeper import _load_policy
  base = _load_policy(cfg.checkpoint, env, dev)

  robot = env.unwrapped.scene["robot"]; ball = env.unwrapped.scene["ball"]
  blk = torch.as_tensor(robot.find_bodies(_BLOCK_LINKS, preserve_order=True)[0], device=dev)
  gen = torch.Generator(device=dev); gen.manual_seed(cfg.seed)

  T, K, J = cfg.horizon, cfg.knots, 29
  # Concentrate knots in the early SAVE window [0, knot_span] (the ball arrives at
  # step ~25-50); steps beyond the last knot hold its value. Uniform knots over
  # 150 steps are too coarse for explosive high saves.
  span = min(cfg.knot_span, T - 1)
  kt = torch.linspace(0, span, K, device=dev)
  st = torch.arange(T, device=dev).float().clamp(max=span)
  idx = torch.clamp(torch.bucketize(st, kt) - 1, 0, K - 2)
  t0 = kt[idx]; t1 = kt[idx + 1]; frac = (st - t0) / (t1 - t0).clamp(min=1e-6)
  W = torch.zeros(T, K, device=dev)
  W[torch.arange(T), idx] = 1 - frac; W[torch.arange(T), idx + 1] = frac
  release = max(1, cfg.release_steps)
  fade = torch.ones(T, device=dev)
  after_span = torch.arange(T, device=dev).float() - float(span)
  fade = torch.where(after_span > 0.0, torch.clamp(1.0 - after_span / release, 0.0, 1.0), fade)

  keep = torch.arange(0, N, cfg.P, device=dev)   # 1 env per scenario (all P identical)

  def collect_window(scen_start, scen_vel):
    """Frame mask around keeper-plane crossing; avoids teaching post-save falls."""
    vx = torch.minimum(
      scen_vel[:, 0],
      torch.full((scen_start.shape[0],), -1.0e-3, device=dev),
    )
    t_keeper = torch.clamp(-scen_start[:, 0] / vx, min=0.0, max=T * 0.02)
    center = torch.round(t_keeper / 0.02).long().clamp(0, T - 1)
    frame = torch.arange(T, device=dev).unsqueeze(0)
    lo = (center - cfg.collect_pre_steps).unsqueeze(1)
    hi = (center + cfg.collect_post_steps).unsqueeze(1)
    return (frame >= lo) & (frame <= hi)

  def rollout(knots, scen_start, scen_vel, scen_region, collect=False):
    """knots:(N,K,J). Returns cost(N), blocked(N), and optionally (obs,act) seq
    for the `keep` envs only (stored on CPU to avoid OOM)."""
    R = torch.einsum('tk,nkj->ntj', W, knots).clamp(-cfg.clip, cfg.clip)  # (N,T,J)
    R = R * fade[None, :, None]
    obs = _apply(env, robot, ball, scen_start, scen_vel, cfg.P, scen_region)
    _reset_policy(base)
    entered = torch.zeros(N, dtype=torch.bool, device=dev)
    min_d = torch.full((N,), 1e9, device=dev)
    upr_bad = torch.zeros(N, device=dev)
    stable_bad = torch.zeros(N, device=dev)
    obs_buf = [] if collect else None
    act_buf = [] if collect else None
    base_act_buf = [] if collect else None
    for t in range(T):
      with torch.inference_mode():
        ba = base(obs)
      a = (ba + R[:, t]).clamp(-100, 100)
      if collect:
        obs_buf.append(obs["actor"][keep].to("cpu")); act_buf.append(a[keep].to("cpu"))
        base_act_buf.append(ba[keep].to("cpu"))
      res = env.step(a); obs = res[0]
      bp = ball.data.root_link_pos_w
      bx = bp[:, 0] - env.unwrapped.scene.env_origins[:, 0]
      by = bp[:, 1] - env.unwrapped.scene.env_origins[:, 1]
      entered |= (bx <= -0.5) & (by.abs() <= 1.5) & (bp[:, 2] <= 1.8)
      body = robot.data.body_link_pos_w[:, blk]
      d = (body - bp.unsqueeze(1)).pow(2).sum(-1).min(1).values.sqrt()
      min_d = torch.minimum(min_d, d)
      # upright penalty only while ball is near the keeper plane (contact window)
      near = bx.abs() < 0.6
      gz = robot.data.projected_gravity_b[:, 2]   # ~ -1 upright
      upr_bad += near.float() * torch.clamp(gz + 0.3, min=0.0)
      stable_bad += torch.clamp(gz + 0.5, min=0.0)
    final_gz = robot.data.projected_gravity_b[:, 2]
    cost = (cfg.w_goal * entered.float() + cfg.w_dist * min_d
            + cfg.w_res * R.pow(2).mean((1, 2)) + cfg.w_upright * upr_bad / T
            + cfg.w_stable * stable_bad / T
            + cfg.w_final_upright * torch.clamp(final_gz + 0.5, min=0.0))
    if collect:
      return (
        cost,
        ~entered,
        min_d,
        torch.stack(obs_buf, 1),
        torch.stack(act_buf, 1),
        torch.stack(base_act_buf, 1),
        final_gz[keep].to("cpu"),
      )
    return cost, ~entered, min_d

  all_data_obs, all_data_act, all_data_base_act, all_data_blk = [], [], [], []
  agg_base, agg_rep, agg_union, agg_n = 0, 0, 0, 0
  for b in range(cfg.batches):
    reg, start, end, tf = _gen_scenarios(env, cfg.regions, cfg.region_weights, cfg.G, gen, dev)
    vel = _ball_vel(start, end, tf)
    # baseline (residual 0). In collect mode also record the base trajectory so
    # scenarios the base ALREADY blocks are taught with residual=0 (pure base
    # action) — this protects the easy-ball behavior from CEM residual noise.
    z = torch.zeros(N, K, J, device=dev)
    if cfg.mode == "collect":
      _, base_blk, _, base_ob, base_ac, base_ba, base_gz = rollout(z, start, vel, reg, collect=True)
    else:
      _, base_blk, _ = rollout(z, start, vel, reg)
    base_rate = base_blk.view(cfg.G, cfg.P).float().mean(1)   # per scenario
    # CEM (iCEM: keep-best-ever per scenario + elite carryover). The final
    # repair uses the best sampled candidate, not the elite mean: blocking is a
    # threshold event, so averaging two good dive timings can create a bad one.
    mu = torch.zeros(cfg.G, K, J, device=dev)
    sig = torch.full((cfg.G, K, J), cfg.init_std, device=dev)
    best_cost = torch.full((cfg.G,), float("inf"), device=dev)
    best_knots = torch.zeros(cfg.G, K, J, device=dev)
    carry = None  # (G, nkeep, K, J) elites carried from previous iter
    nkeep = max(1, cfg.elites // 4)
    for it in range(cfg.iters):
      eps = torch.randn(N, K, J, generator=gen, device=dev)
      knots = (mu.repeat_interleave(cfg.P, 0) + sig.repeat_interleave(cfg.P, 0) * eps)
      knots = knots.clamp(-cfg.clip, cfg.clip)
      knots_g = knots.view(cfg.G, cfg.P, K, J)
      if carry is not None:                       # inject carried elites
        knots_g[:, :nkeep] = carry
        knots = knots_g.view(N, K, J)
      cost, blk_, _ = rollout(knots, start, vel, reg)
      cost_g = cost.view(cfg.G, cfg.P)
      best_val, best_idx = cost_g.min(1)
      improved = best_val < best_cost
      if improved.any():
        best_cost[improved] = best_val[improved]
        best_knots[improved] = knots_g[
          improved,
          best_idx[improved],
        ]
      ei = cost_g.topk(cfg.elites, largest=False).indices   # (G,E)
      el = torch.gather(knots_g, 1, ei[:, :, None, None].expand(-1, -1, K, J))
      mu = cfg.smooth * el.mean(1) + (1 - cfg.smooth) * mu
      sig = cfg.smooth * el.std(1) + (1 - cfg.smooth) * sig
      sig = sig.clamp(min=0.03)
      carry = el[:, :nkeep].clone()               # best nkeep elites for next iter
    # final deterministic eval of the repair (best candidate per scenario).
    mu_e = best_knots.repeat_interleave(cfg.P, 0)
    if cfg.mode == "collect":
      _, rep_blk, _, rep_ob, rep_ac, rep_ba, rep_gz = rollout(mu_e, start, vel, reg, collect=True)
      bbk = base_blk[keep].cpu()          # (G,) base blocked at keep env
      rbk = rep_blk[keep].cpu()           # (G,) repair blocked at keep env
      if cfg.require_final_upright:
        rbk = rbk & (rep_gz < cfg.final_upright_gz)
        bbk = bbk & (base_gz < cfg.final_upright_gz)
      use_base = bbk[:, None, None]       # prefer pure base action where it works
      ob = torch.where(use_base, base_ob, rep_ob)
      ac = torch.where(use_base, base_ac, rep_ac)
      ba = torch.where(use_base, base_ba, rep_ba)
      blocked = (bbk | rbk)[:, None] & collect_window(start, vel).cpu()
      all_data_obs.append(ob.reshape(-1, ob.shape[-1]))
      all_data_act.append(ac.reshape(-1, ac.shape[-1]))
      all_data_base_act.append(ba.reshape(-1, ba.shape[-1]))
      all_data_blk.append(blocked.reshape(-1))
    else:
      _, rep_blk, _ = rollout(mu_e, start, vel, reg)
    rep_rate = rep_blk.view(cfg.G, cfg.P).float().mean(1)
    union_rate = (base_blk | rep_blk).view(cfg.G, cfg.P).float().mean(1)
    agg_base += float(base_rate.mean()) * cfg.G
    agg_rep += float(rep_rate.mean()) * cfg.G
    agg_union += float(union_rate.mean()) * cfg.G
    agg_n += cfg.G
    print(f"  batch {b+1}/{cfg.batches}: base {100*float(base_rate.mean()):.1f}% -> "
          f"repaired {100*float(rep_rate.mean()):.1f}% ; "
          f"base_or_repair {100*float(union_rate.mean()):.1f}%  (n={cfg.G})", flush=True)
  print(f"\n{'='*60}\n  REGIONS {[_REGION[r] for r in cfg.regions]}\n"
        f"  base {100*agg_base/agg_n:.1f}%  ->  repaired {100*agg_rep/agg_n:.1f}%   "
        f"; base_or_repair {100*agg_union/agg_n:.1f}%   "
        f"(over {agg_n} scenarios)\n{'='*60}")
  if cfg.mode == "collect" and cfg.out:
    obs = torch.cat(all_data_obs); act = torch.cat(all_data_act); bk = torch.cat(all_data_blk)
    base_act = torch.cat(all_data_base_act)
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"obs": obs, "act": act, "base_act": base_act, "blocked": bk}, cfg.out)
    print(f"  saved {obs.shape[0]} (obs,act) pairs to {cfg.out}  "
          f"(repaired-blocked frac {float(bk.float().mean()):.2f})")
  env.close()


if __name__ == "__main__":
  import mjlab.tasks, src.tasks  # noqa
  main(tyro.cli(Cfg, prog="repair_oracle"))
