import numpy as np
import torch
from torch.distributions.normal import Normal


def _policy_dist(agent, states_t: torch.Tensor):
    """
    Editor-aware policy forward for analysis/eval.

    Uses agent._forward_policy(...) if available, otherwise falls back to
    agent.policy(...). This makes KL/localization correct for repedit too.
    """
    if hasattr(agent, "_forward_policy"):
        mean, std, _, _ = agent._forward_policy(states_t)
    else:
        mean, std, _, _ = agent.policy(states_t)
    return Normal(mean, std)


def gaussian_policy_kl(agent_a, agent_b, states):
    """
    Compute KL divergence between two Gaussian policies:
        KL(pi_a || pi_b)
    """
    states = np.asarray(states, dtype=np.float32)
    if len(states) == 0:
        return 0.0

    with torch.no_grad():
        states_t = torch.as_tensor(
            states, dtype=torch.float32, device=agent_a.device
        )

        dist_a = _policy_dist(agent_a, states_t)
        dist_b = _policy_dist(agent_b, states_t)

        kl = torch.distributions.kl_divergence(dist_a, dist_b)
        return kl.mean().item()


def collect_states(env, agent, n_steps: int = 5000):
    """
    Collect states from environment rollout.
    """
    states = []

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for _ in range(n_steps):
        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        step_out = env.step(act)
        obs, rew, cost, term, trunc, info = step_out

        states.append(np.asarray(obs, dtype=np.float32))

        if bool(term) or bool(trunc):
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    return np.asarray(states, dtype=np.float32)


def extract_hazard_distance(obs):
    """
    Approximate hazard distance from Safety-Gym style lidar observations.

    For SafetyPointGoal-style observations, hazard lidar is commonly in the
    first channels. Higher lidar activation means closer obstacle, so:
        distance ~= 1 - max(hazard_lidar)
    """
    obs = np.asarray(obs, dtype=np.float32)

    if obs.shape[-1] < 16:
        raise ValueError(
            f"Observation last-dim too small for hazard lidar heuristic: {obs.shape}"
        )

    hazard_lidar = obs[..., :16]
    dist = 1.0 - hazard_lidar.max(axis=-1)
    return dist


def behaviour_localization(agent_old, agent_new, states):
    """
    Compare policy change near hazards vs far away using KL.

    Returns:
        kl_near, kl_far
    """
    states = np.asarray(states, dtype=np.float32)
    if len(states) == 0:
        return 0.0, 0.0

    hazard_dist = extract_hazard_distance(states)
    threshold = np.percentile(hazard_dist, 50)

    near_mask = hazard_dist < threshold
    far_mask = hazard_dist >= threshold

    states_near = states[near_mask]
    states_far = states[far_mask]

    kl_near = gaussian_policy_kl(agent_old, agent_new, states_near)
    kl_far = gaussian_policy_kl(agent_old, agent_new, states_far)

    return kl_near, kl_far