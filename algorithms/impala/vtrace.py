"""
V-trace Implementation for IMPALA.

V-trace is an off-policy actor-critic algorithm that uses importance
sampling to correct for the difference between the behavior policy
(used to collect experience) and the target policy (being learned).

Key features:
- Truncated importance sampling for stability
- Multi-step off-policy correction
- Unbiased value estimation

Per DEC-0039: Vectorized tensor operations.
"""

from typing import Tuple
import torch
from torch import Tensor


def compute_vtrace(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    log_pi: Tensor,
    log_mu: Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute V-trace targets and advantages.

    V-trace corrects for off-policy data using truncated importance sampling:

    v_s = V(s) + sum_{t>=s} gamma^(t-s) (prod_{i=s}^{t-1} c_i) * delta_t
    where delta_t = rho_t * (r_t + gamma*V(s_{t+1}) - V(s_t))

    rho_t = min(rho_bar, pi(a_t|s_t) / mu(a_t|s_t))
    c_t = min(c_bar, pi(a_t|s_t) / mu(a_t|s_t))

    Args:
        rewards: (T, B) rewards at each timestep
        dones: (T, B) done flags (True if episode ended)
        values: (T, B) value estimates V(s_t)
        bootstrap_value: (B,) value estimate for s_T (state after last action)
        log_pi: (T, B) log probability under target policy
        log_mu: (T, B) log probability under behavior policy
        gamma: Discount factor
        rho_bar: Truncation threshold for rho (importance weights in TD error)
        c_bar: Truncation threshold for c (importance weights in trace)

    Returns:
        Tuple of:
        - vs: (T, B) V-trace value targets
        - advantages: (T, B) policy gradient advantages (rho * (r + gamma*v' - V))
    """
    T, B = rewards.shape
    device = rewards.device

    # Compute importance weights
    # log(pi/mu) = log(pi) - log(mu)
    log_rho = log_pi - log_mu
    rho = torch.exp(log_rho)

    # Truncate importance weights
    rho_clipped = torch.clamp(rho, max=rho_bar)
    c_clipped = torch.clamp(rho, max=c_bar)

    # Compute TD errors: delta_t = rho_t * (r_t + gamma * V(s_{t+1}) - V(s_t))
    # For the last step, V(s_{t+1}) = bootstrap_value
    # For done states, V(s_{t+1}) = 0

    # Create V(s_{t+1}) tensor
    values_next = torch.zeros(T, B, device=device)
    values_next[:-1] = values[1:]  # V(s_{t+1}) for t < T-1
    values_next[-1] = bootstrap_value  # Bootstrap for last step

    # Zero out next values for terminal states
    values_next = torch.where(dones, torch.zeros_like(values_next), values_next)

    # TD errors
    delta = rho_clipped * (rewards + gamma * values_next - values)

    # Compute V-trace targets backwards
    # v_s = V(s) + sum of discounted, c-weighted deltas
    vs = torch.zeros(T, B, device=device)
    vs_next = bootstrap_value.clone()

    for t in reversed(range(T)):
        # Reset vs_next on episode boundary
        vs_next = torch.where(dones[t], torch.zeros_like(vs_next), vs_next)

        # V-trace update: v_t = V(s_t) + delta_t + gamma * c_t * (v_{t+1} - V(s_{t+1}))
        vs[t] = values[t] + delta[t] + gamma * c_clipped[t] * (vs_next - values_next[t])
        vs_next = vs[t]

    # Policy gradient advantages: rho * (r + gamma * v' - V)
    # Use vs as the target value (not values_next)
    vs_shifted = torch.zeros(T, B, device=device)
    vs_shifted[:-1] = vs[1:]
    vs_shifted[-1] = bootstrap_value

    # Zero out for terminal states
    vs_shifted = torch.where(dones, torch.zeros_like(vs_shifted), vs_shifted)

    # Advantages for policy gradient
    advantages = rho_clipped * (rewards + gamma * vs_shifted - values)

    return vs, advantages


def compute_vtrace_from_logits(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    bootstrap_value: Tensor,
    target_logits: Tensor,
    behavior_logits: Tensor,
    actions: Tensor,
    valid_masks: Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[Tensor, Tensor]:
    """Compute V-trace from logits instead of log probabilities.

    This is a convenience function that computes log probabilities from
    logits and valid masks, then calls compute_vtrace.

    Args:
        rewards: (T, B) rewards
        dones: (T, B) done flags
        values: (T, B) value estimates
        bootstrap_value: (B,) bootstrap value
        target_logits: (T, B, A) logits from target policy
        behavior_logits: (T, B, A) logits from behavior policy
        actions: (T, B) actions taken
        valid_masks: (T, B, A) valid action masks
        gamma: Discount factor
        rho_bar: rho truncation threshold
        c_bar: c truncation threshold

    Returns:
        Tuple of (vs, advantages)
    """
    import torch.nn.functional as F

    T, B, A = target_logits.shape

    # Mask invalid actions before computing log probabilities
    masked_target = target_logits.clone()
    masked_target[~valid_masks] = float('-inf')

    masked_behavior = behavior_logits.clone()
    masked_behavior[~valid_masks] = float('-inf')

    # Log softmax for numerical stability
    log_pi_all = F.log_softmax(masked_target, dim=-1)  # (T, B, A)
    log_mu_all = F.log_softmax(masked_behavior, dim=-1)  # (T, B, A)

    # Get log prob of taken action
    log_pi = log_pi_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)
    log_mu = log_mu_all.gather(2, actions.unsqueeze(-1)).squeeze(-1)  # (T, B)

    return compute_vtrace(
        rewards, dones, values, bootstrap_value,
        log_pi, log_mu, gamma, rho_bar, c_bar
    )
