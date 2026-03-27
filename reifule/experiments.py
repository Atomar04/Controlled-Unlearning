import torch
import numpy as np
from torch.distributions.normal import Normal


def gaussian_policy_kl(agent_a, agent_b, states):

    """
    Compute KL divergence between two Gaussian policies
    KL(π_a || π_b)
    """

    with torch.no_grad():

        states = torch.as_tensor(states, dtype=torch.float32, device=agent_a.device)

        mean_a, std_a, _, _ = agent_a.policy(states)
        mean_b, std_b, _, _ = agent_b.policy(states)

        dist_a = Normal(mean_a, std_a)
        dist_b = Normal(mean_b, std_b)

        kl = torch.distributions.kl_divergence(dist_a, dist_b)

        return kl.mean().item()


def collect_states(env, agent, n_steps=5000):

    """
    Collect states from environment rollout
    """

    states = []

    obs, info = env.reset()

    for _ in range(n_steps):

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        obs, rew, cost, term, trunc, info = env.step(act)

        states.append(obs)

        if term or trunc:
            obs, info = env.reset()

    return np.array(states)


def extract_hazard_distance(obs):

    """
    Safety-Gym observation format contains lidar features.
    Hazard lidar usually appears in the first channels.
    """

    # This approximation works for SafetyPointGoal
    hazard_lidar = obs[..., :16]

    dist = 1.0 - hazard_lidar.max(axis=-1)

    return dist


def behaviour_localization(agent_old, agent_new, states):

    """
    Compare policy change near hazards vs far away
    """

    hazard_dist = extract_hazard_distance(states)

    threshold = np.percentile(hazard_dist, 50)

    near_mask = hazard_dist < threshold
    far_mask = hazard_dist >= threshold

    states_near = states[near_mask]
    states_far = states[far_mask]

    kl_near = gaussian_policy_kl(agent_old, agent_new, states_near)

    kl_far = gaussian_policy_kl(agent_old, agent_new, states_far) \
        if len(states_far) > 0 else 0.0

    return kl_near, kl_far
