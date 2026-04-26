import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch.distributions.normal import Normal
import wandb

from reifule.algorithm import PPOUnlearner
from reifule.utils import (
    make_env,
    set_seed,
    extract_cforget,
    model_checkpoint_path,
)


@dataclass
class Episode:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    cforget: np.ndarray
    has_forget: bool


def discounted_returns_with_bootstrap(
    rewards: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    bootstrap_value: float,
    gamma: float,
):
    """
    Returns with proper handling of truncation.

    Important:
    - bootstrap_value is injected as the initial running value
    - if episode ended by truncation, final return includes V(s_T)
    - if episode truly terminated, running is reset to 0 at that step
    """
    rets = np.zeros_like(rewards, dtype=np.float32)
    running = float(bootstrap_value)

    for t in reversed(range(len(rewards))):
        if terminated[t]:
            running = 0.0
        running = float(rewards[t]) + gamma * running
        rets[t] = running

    return rets


def collect_unsafe_episodes(
    unsafe_agent: PPOUnlearner,
    env_id: str,
    seed: int,
    n_episodes: int,
):
    """
    Roll out the unsafe expert and collect full episodes.
    """
    env = make_env(env_id, n_envs=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    episodes: List[Episode] = []
    env_steps = 0

    obs, info = env.reset()

    cur_states = []
    cur_actions = []
    cur_rewards = []
    cur_terminated = []
    cur_truncated = []
    cur_cforget = []

    while len(episodes) < n_episodes:
        act, _, _, _ = unsafe_agent.act(obs, deterministic=False)
        act = act[0] if act.ndim == 2 else act

        next_obs, rew, cost, term, trunc, info = env.step(act)
        env_steps += 1

        c_forget = extract_cforget(cost, info)
        c_forget = float(np.asarray(c_forget).reshape(-1)[0])

        cur_states.append(np.asarray(obs, dtype=np.float32))
        cur_actions.append(np.asarray(act, dtype=np.float32))
        cur_rewards.append(float(rew))
        cur_terminated.append(bool(term))
        cur_truncated.append(bool(trunc))
        cur_cforget.append(c_forget)

        obs = next_obs

        if term or trunc:
            ep = Episode(
                states=np.asarray(cur_states, dtype=np.float32),
                actions=np.asarray(cur_actions, dtype=np.float32),
                rewards=np.asarray(cur_rewards, dtype=np.float32),
                terminated=np.asarray(cur_terminated, dtype=np.bool_),
                truncated=np.asarray(cur_truncated, dtype=np.bool_),
                cforget=np.asarray(cur_cforget, dtype=np.float32),
                has_forget=bool(np.sum(cur_cforget) > 0),
            )
            episodes.append(ep)

            obs, info = env.reset()
            cur_states = []
            cur_actions = []
            cur_rewards = []
            cur_terminated = []
            cur_truncated = []
            cur_cforget = []

    env.close()
    return episodes, env_steps


def flatten_retain_episodes(
    episodes: List[Episode],
    bootstrap_agent: PPOUnlearner,
    gamma: float,
):
    """
    Flatten retained full-safe episodes into a transition dataset.
    """
    states_all = []
    actions_all = []
    returns_all = []

    device = bootstrap_agent.device

    for ep in episodes:
        if ep.has_forget:
            continue

        last_state = ep.states[-1]
        last_term = bool(ep.terminated[-1])
        last_trunc = bool(ep.truncated[-1])

        bootstrap_value = 0.0
        if last_trunc and not last_term:
            with torch.no_grad():
                s = torch.as_tensor(
                    last_state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                _, _, vr, _ = bootstrap_agent.policy(s)
                bootstrap_value = float(vr.squeeze(0).item())

        rets = discounted_returns_with_bootstrap(
            rewards=ep.rewards,
            terminated=ep.terminated,
            truncated=ep.truncated,
            bootstrap_value=bootstrap_value,
            gamma=gamma,
        )

        states_all.append(ep.states)
        actions_all.append(ep.actions)
        returns_all.append(rets)

    if len(states_all) == 0:
        return None

    return {
        "states": np.concatenate(states_all, axis=0),
        "actions": np.concatenate(actions_all, axis=0),
        "returns": np.concatenate(returns_all, axis=0),
    }


def offline_decremental_update(
    agent: PPOUnlearner,
    retain_data: Dict[str, np.ndarray],
    epochs: int = 10,
    batch_size: int = 256,
    value_coef: float = 0.5,
    bc_coef: float = 1.0,
    ent_coef: float = 0.01,
):
    """
    Retain-only decremental-style update.

    Objective:
      - behavior cloning on retained actions
      - reward value regression on retained returns
      - entropy bonus to reduce memorization / collapse

    No KL to unsafe policy.
    """
    device = agent.device

    states_np = retain_data["states"]
    actions_np = retain_data["actions"]
    returns_np = retain_data["returns"]

    actions_np = np.clip(actions_np, -1.0 + 1e-6, 1.0 - 1e-6)

    states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions_np, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns_np, dtype=torch.float32, device=device)

    N = states.shape[0]
    idx = np.arange(N)

    total_loss = 0.0
    total_entropy = 0.0
    n_batches = 0

    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, N, batch_size):
            mb = idx[start:start + batch_size]

            s = states[mb]
            a = actions[mb]
            R = returns[mb]

            mean, std, vr, _ = agent.policy(s)

            u = agent._unsquash(a)
            logp = agent._logp(mean, std, u, a)
            bc_loss = -logp.mean()

            value_loss = 0.5 * ((vr - R) ** 2).mean()

            entropy = Normal(mean, std).entropy().sum(dim=-1).mean()

            loss = (
                (bc_coef * bc_loss)
                + (value_coef * value_loss)
                - (ent_coef * entropy)
            )

            agent.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.policy.parameters(), agent.max_grad_norm
            )
            agent.opt.step()

            total_loss += float(loss.item())
            total_entropy += float(entropy.item())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_entropy = total_entropy / max(n_batches, 1)
    return avg_loss, avg_entropy


def train_trajectory_decremental(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_trajectory_decremental_{args.env}_Seed{args.seed}",
        group=f"Unlearning_trajectory_decremental_{args.env}",
        config=vars(args),
    )

    set_seed(args.seed)

    env = make_env(args.env, n_envs=1)

    agent = PPOUnlearner(
        env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )
    agent.load(model_checkpoint_path("unsafe", args.env))

    unsafe_ref = PPOUnlearner(
        env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )
    unsafe_ref.load(model_checkpoint_path("unsafe", args.env))
    unsafe_ref.policy.eval()
    for p in unsafe_ref.policy.parameters():
        p.requires_grad_(False)

    save_points = set(args.save_points)

    env_steps_total = args.unsafe_updates * args.horizon * args.n_envs

    print("\n[TRAJECTORY_DECREMENTAL] Collecting unsafe expert episodes...")
    episodes, steps_used = collect_unsafe_episodes(
        unsafe_agent=unsafe_ref,
        env_id=args.env,
        seed=args.seed,
        n_episodes=args.dataset_episodes,
    )
    env_steps_total += steps_used

    log_step = env_steps_total

    n_forget = sum(int(ep.has_forget) for ep in episodes)
    n_retain = len(episodes) - n_forget

    print(f"[TRAJECTORY_DECREMENTAL] Total episodes collected: {len(episodes)}")
    print(f"[TRAJECTORY_DECREMENTAL] Forget episodes: {n_forget}")
    print(f"[TRAJECTORY_DECREMENTAL] Retain episodes: {n_retain}")

    if n_retain == 0:
        raise RuntimeError(
            "No fully safe retained episodes were collected. "
            "Increase dataset_episodes or change environment difficulty."
        )

    retain_data = flatten_retain_episodes(
        episodes=episodes,
        bootstrap_agent=unsafe_ref,
        gamma=agent.gamma,
    )
    if retain_data is None:
        raise RuntimeError("Retain dataset is empty after filtering.")

    print(
        f"[TRAJECTORY_DECREMENTAL] Retained transitions: "
        f"{len(retain_data['states'])}"
    )

    current_retain_episodes = n_retain
    current_forget_episodes = n_forget
    current_retained_transitions = len(retain_data["states"])

    for update in range(1, args.updates + 1):
        if update > 1 and (update - 1) % args.recollect_every == 0:
            print(
                f"\n[TRAJECTORY_DECREMENTAL] Recollecting retain dataset "
                f"at update {update}..."
            )
            episodes, steps_used = collect_unsafe_episodes(
                unsafe_agent=unsafe_ref,
                env_id=args.env,
                seed=args.seed + update,
                n_episodes=args.dataset_episodes,
            )
            env_steps_total += steps_used

            n_forget = sum(int(ep.has_forget) for ep in episodes)
            n_retain = len(episodes) - n_forget

            if n_retain == 0:
                raise RuntimeError(
                    f"No retained episodes after recollection at update {update}."
                )

            retain_data = flatten_retain_episodes(
                episodes=episodes,
                bootstrap_agent=unsafe_ref,
                gamma=agent.gamma,
            )
            if retain_data is None:
                raise RuntimeError("Retain dataset became empty after recollection.")

            current_retain_episodes = n_retain
            current_forget_episodes = n_forget
            current_retained_transitions = len(retain_data["states"])

            print(
                f"[TRAJECTORY_DECREMENTAL] Recollected retain episodes: "
                f"{current_retain_episodes}"
            )
            print(
                f"[TRAJECTORY_DECREMENTAL] Recollected retained transitions: "
                f"{current_retained_transitions}"
            )

        loss, entropy = offline_decremental_update(
            agent=agent,
            retain_data=retain_data,
            epochs=args.decremental_epochs_per_update,
            batch_size=args.batch_size,
            value_coef=args.decremental_value_coef,
            bc_coef=args.decremental_bc_coef,
            ent_coef=args.decremental_ent_coef,
        )

        log_step += args.horizon

        wandb.log(
            {
                "Train/Update": update,
                "Train/Loss": loss,
                "Train/Entropy": entropy,
                "Data/RetainEpisodes": current_retain_episodes,
                "Data/ForgetEpisodes": current_forget_episodes,
                "Data/RetainedTransitions": current_retained_transitions,
                "Compute/EnvStepsTotal": env_steps_total,
                "Compute/LogStep": log_step,
            },
            step=log_step,
        )

        print(
            f"[TRAJECTORY_DECREMENTAL] Update {update}/{args.updates} | "
            f"Loss {loss:.4f} | Entropy {entropy:.4f}"
        )

        if update in save_points:
            path = model_checkpoint_path("trajectory_decremental", args.env, update)
            agent.save(path)
            wandb.save(path)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--unsafe_updates", type=int, default=300)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--dataset_episodes", type=int, default=200)
    p.add_argument("--decremental_epochs_per_update", type=int, default=10)
    p.add_argument("--decremental_bc_coef", type=float, default=1.0)
    p.add_argument("--decremental_value_coef", type=float, default=0.5)
    p.add_argument("--decremental_ent_coef", type=float, default=0.01)
    p.add_argument("--recollect_every", type=int, default=10)

    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    args = p.parse_args()
    train_trajectory_decremental(args)