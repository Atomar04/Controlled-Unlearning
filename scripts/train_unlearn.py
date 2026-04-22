import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import random
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch
from torch.distributions.normal import Normal
import wandb

from reifule.algorithm import PPOUnlearner
from reifule.computation_amnesiac import PIDLagrangian
from scripts.train_unsafe import make_env


# Utilities


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_cforget(cost, info):
    """
    Binary forget concept: hazard contact.
    Uses cost_hazards if present, otherwise falls back to cost > 0.
    Supports single-env and vector-env outputs.
    """
    if isinstance(info, dict) and "cost_hazards" in info:
        ch = np.asarray(info["cost_hazards"], dtype=np.float32)
        return (ch > 0).astype(np.float32)
    return (np.asarray(cost, dtype=np.float32) > 0).astype(np.float32)


@dataclass
class Episode:
    states: np.ndarray          # [T, obs]
    actions: np.ndarray         # [T, act]
    rewards: np.ndarray         # [T]
    terminated: np.ndarray      # [T] true terminal only
    truncated: np.ndarray       # [T] timeout/truncation only
    cforget: np.ndarray         # [T]
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
    - `bootstrap_value` is injected as the initial running value.
    - Therefore, if the episode ended by truncation, the final return
      correctly includes V(s_T).
    - If the episode truly terminated, running is reset to 0 at that step.
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
    Roll out the UNSAFE expert and collect full episodes.

    Scientific note:
    This deliberately keeps the retain/forget dataset defined by the
    unsafe expert's experience distribution. Recollection with new seeds
    increases sample diversity from the SAME fixed unsafe policy; it does
    not make targets on-policy for the evolving decremental agent.

    Returns:
      episodes, env_steps_consumed
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

    Scientific choice:
    We bootstrap using the fixed unsafe reference policy as part of the
    fixed-distribution decremental baseline. This is consistent with the
    data-level framing, though it introduces target mismatch as the
    evolving agent drifts.
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
                s = torch.as_tensor(last_state, dtype=torch.float32, device=device).unsqueeze(0)
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
      - behaviour cloning on retained actions
      - reward value regression on retained returns
      - entropy bonus to reduce memorization / collapse

    No KL to unsafe policy.
    """
    device = agent.device

    states_np = retain_data["states"]
    actions_np = retain_data["actions"]
    returns_np = retain_data["returns"]

    # Safe fallback clipping for tanh-squashed policies.
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

            loss = (bc_coef * bc_loss) + (value_coef * value_loss) - (ent_coef * entropy)

            agent.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
            agent.opt.step()

            total_loss += float(loss.item())
            total_entropy += float(entropy.item())
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_entropy = total_entropy / max(n_batches, 1)
    return avg_loss, avg_entropy


# =========================================================
# Main training
# =========================================================

def train_unlearning(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_{args.mode}_{args.env}_Seed{args.seed}",
        group=f"Unlearning_{args.mode}_{args.env}",
        config=vars(args),
    )

    set_seed(args.seed)

    # -----------------------------------------------------
    # MODE 1: concept-level post-hoc constrained fine-tuning
    # -----------------------------------------------------
    if args.mode == "concept":
        env = make_env(args.env, n_envs=args.n_envs)

        agent = PPOUnlearner(
            env,
            lr=args.lr,
            ent_coef=args.ent_coef,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        agent.load(f"unsafe_expert_{args.env}.pt")

        pid = PIDLagrangian(
            goal_cost=args.target_cost,
            Kp=args.kp,
            Ki=args.ki,
            Kd=args.kd,
            lambda_init=args.lambda_init,
            lambda_max=args.lambda_max,
            integral_max=args.integral_max,
        )

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        env_steps_total = args.unsafe_updates * args.horizon * args.n_envs
        save_points = set(args.save_points)

        for update in range(1, args.updates + 1):
            buf = {
                "states": [],
                "actions": [],
                "logp": [],
                "rewards": [],
                "costs": [],
                "terminated": [],
            }

            batch_rew_sum = 0.0
            batch_cforget_sum = 0.0

            for _ in range(args.horizon):
                act, logp, _, _ = agent.act(obs, deterministic=False)
                next_obs, rew, cost, term, trunc, info = env.step(act)

                c_forget = extract_cforget(cost, info)

                buf["states"].append(np.asarray(obs, dtype=np.float32))
                buf["actions"].append(np.asarray(act, dtype=np.float32))
                buf["logp"].append(np.asarray(logp, dtype=np.float32))
                buf["rewards"].append(np.asarray(rew, dtype=np.float32))
                buf["costs"].append(np.asarray(c_forget, dtype=np.float32))
                buf["terminated"].append(np.asarray(term, dtype=np.bool_))

                batch_rew_sum += float(np.asarray(rew, dtype=np.float32).mean())
                batch_cforget_sum += float(np.asarray(c_forget, dtype=np.float32).mean())

                obs = next_obs
                env_steps_total += args.n_envs

            mean_reward_step = batch_rew_sum / args.horizon
            mean_cost_rate = batch_cforget_sum / args.horizon
            lambda_val = pid.update(mean_cost_rate)

            states = np.asarray(buf["states"], dtype=np.float32)
            actions = np.asarray(buf["actions"], dtype=np.float32)
            logp = np.asarray(buf["logp"], dtype=np.float32)
            rewards = np.asarray(buf["rewards"], dtype=np.float32)
            costs = np.asarray(buf["costs"], dtype=np.float32)
            terminated = np.asarray(buf["terminated"], dtype=np.bool_)

            batch = {
                "states": states,
                "actions": actions,
                "log_probs": logp,
                "rewards": rewards,
                "costs": costs,
                "terminated": terminated,
                "last_state": np.asarray(obs, dtype=np.float32),
                "last_terminated": np.asarray(terminated[-1], dtype=np.bool_),
            }

            loss = agent.update(batch, lambda_val=lambda_val)

            wandb.log(
                {
                    "Train/Update": update,
                    "Train/Lambda": lambda_val,
                    "Train/Batch_MeanRewardStep": mean_reward_step,
                    "Train/Batch_Cost_Rate": mean_cost_rate,
                    "Train/PID_Error": mean_cost_rate - args.target_cost,
                    "Train/Loss": loss,
                    "Compute/EnvStepsTotal": env_steps_total,
                },
                step=env_steps_total,
            )

            print(
                f"[CONCEPT] Update {update}/{args.updates} | "
                f"Rew {mean_reward_step:.4f} | Cost {mean_cost_rate:.4f} | "
                f"Lambda {lambda_val:.4f}"
            )

            if update in save_points:
                path = f"safe_concept_{args.env}_{update}.pt"
                agent.save(path)
                wandb.save(path)

        wandb.finish()
        return

    # -----------------------------------------------------
    # MODE 2: trajectory-level decremental retain-only baseline
    # -----------------------------------------------------
    elif args.mode == "trajectory_decremental":
        env = make_env(args.env, n_envs=1)

        agent = PPOUnlearner(
            env,
            lr=args.lr,
            ent_coef=args.ent_coef,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        agent.load(f"unsafe_expert_{args.env}.pt")

        # Fixed unsafe reference policy.
        unsafe_ref = PPOUnlearner(
            env,
            lr=args.lr,
            ent_coef=args.ent_coef,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
        )
        unsafe_ref.load(f"unsafe_expert_{args.env}.pt")
        unsafe_ref.policy.eval()
        for p in unsafe_ref.policy.parameters():
            p.requires_grad_(False)

        save_points = set(args.save_points)

        # Match concept-mode offset so both methods share the same starting timeline.
        env_steps_total = args.unsafe_updates * args.horizon * args.n_envs

        print("\n[TRAJECTORY_DECREMENTAL] Collecting unsafe expert episodes...")
        episodes, steps_used = collect_unsafe_episodes(
            unsafe_agent=unsafe_ref,
            env_id=args.env,
            seed=args.seed,
            n_episodes=args.dataset_episodes,
        )
        env_steps_total += steps_used

        # Separate monotonic plotting step from actual env steps.
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

        print(f"[TRAJECTORY_DECREMENTAL] Retained transitions: {len(retain_data['states'])}")

        current_retain_episodes = n_retain
        current_forget_episodes = n_forget
        current_retained_transitions = len(retain_data["states"])

        for update in range(1, args.updates + 1):
            if update > 1 and (update - 1) % args.recollect_every == 0:
                print(f"\n[TRAJECTORY_DECREMENTAL] Recollecting retain dataset at update {update}...")
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

                print(f"[TRAJECTORY_DECREMENTAL] Recollected retain episodes: {current_retain_episodes}")
                print(f"[TRAJECTORY_DECREMENTAL] Recollected retained transitions: {current_retained_transitions}")

            loss, entropy = offline_decremental_update(
                agent=agent,
                retain_data=retain_data,
                epochs=args.decremental_epochs_per_update,
                batch_size=args.batch_size,
                value_coef=args.decremental_value_coef,
                bc_coef=args.decremental_bc_coef,
                ent_coef=args.decremental_ent_coef,
            )

            # Advance monotonic plotting step even without new env interaction.
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
                path = f"safe_trajectory_decremental_{args.env}_{update}.pt"
                agent.save(path)
                wandb.save(path)

        env.close()
        wandb.finish()
        return

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument(
        "--mode",
        type=str,
        default="concept",
        choices=["concept", "trajectory_decremental"],
    )

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed", type=int, default=0)

    # concept-mode PID params
    p.add_argument("--kp", type=float, default=2.0)
    p.add_argument("--ki", type=float, default=0.01)
    p.add_argument("--kd", type=float, default=0.1)
    p.add_argument("--lambda_init", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=200.0)
    p.add_argument("--integral_max", type=float, default=5.0)
    p.add_argument("--target_cost", type=float, default=0.03)

    # shared training params
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--unsafe_updates", type=int, default=300)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)

    # trajectory_decremental params
    p.add_argument("--dataset_episodes", type=int, default=200)
    p.add_argument("--decremental_epochs_per_update", type=int, default=10)
    p.add_argument("--decremental_bc_coef", type=float, default=1.0)
    p.add_argument("--decremental_value_coef", type=float, default=0.5)
    p.add_argument("--decremental_ent_coef", type=float, default=0.01)
    p.add_argument("--recollect_every", type=int, default=10)

    # checkpointing
    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    args = p.parse_args()
    train_unlearning(args)
