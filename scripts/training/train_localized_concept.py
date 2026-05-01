"""
Localized Concept Unlearning.

This is the locality-regularized version of vanilla train_concept.py.

Narrative role:
    Vanilla concept unlearning:
        reduce hazard/contact cost using PID-Lagrangian PPO.

    Localized concept unlearning:
        reduce hazard/contact cost while explicitly preserving the unsafe expert's
        policy away from hazard/contact states.

Objective intuition:
    PPO reward objective
    - lambda * hazard-cost advantage
    + beta * KL(pi_current || pi_unsafe) on non-hazard states

This should increase KL_Locality_Ratio because far/safe states are explicitly
regularized to stay close to the original unsafe expert.

Example:
    python -m scripts.training.train_localized_concept \
        --env SafetyPointGoal1-v0 \
        --updates 150 \
        --horizon 1024 \
        --n_envs 8 \
        --locality_beta 0.05 \
        --save_points 50 100 150
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.distributions.normal import Normal

from reifule.algorithm import PPOUnlearner
from reifule.computation_amnesiac import PIDLagrangian, compute_gae_te
from reifule.utils import (
    make_env,
    set_seed,
    extract_cforget,
    model_checkpoint_path,
    artifact_dir,
)


# =========================================================
# Checkpoint helper
# =========================================================

def localized_concept_checkpoint_path(env: str, update: int) -> str:
    """
    Separate checkpoint family for localized concept unlearning.

    NOTE:
        eval_policy_suite.py must also be updated to discover this family:

            ("localized_concept", r"safe_localized_concept_.*_(\\d+)\\.pt$")

        or equivalent simplified logic.
    """
    folder = artifact_dir("models", "localized_concept")
    return str(folder / f"safe_localized_concept_{env}_{update}.pt")


# =========================================================
# Tensor / environment helpers
# =========================================================

def reset_env_with_seed(env, seed: int):
    try:
        reset_out = env.reset(seed=seed)
    except TypeError:
        reset_out = env.reset()

    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    return obs


def as_float32_array(x):
    return np.asarray(x, dtype=np.float32)


def as_bool_array(x):
    return np.asarray(x, dtype=np.bool_)


def is_vector_env(args) -> bool:
    return args.n_envs > 1


# =========================================================
# KL helper
# =========================================================

def gaussian_kl_diag(mean_p, std_p, mean_q, std_q):
    """
    KL[N(mean_p, std_p) || N(mean_q, std_q)] for diagonal Gaussians.

    Shape:
        mean_p/std_p: [B, act_dim]
        mean_q/std_q: [B, act_dim]

    Returns:
        kl: [B]
    """
    var_p = std_p.pow(2)
    var_q = std_q.pow(2)

    kl = (
        torch.log(std_q + 1e-8)
        - torch.log(std_p + 1e-8)
        + (var_p + (mean_p - mean_q).pow(2)) / (2.0 * var_q + 1e-8)
        - 0.5
    )

    return kl.sum(dim=-1)


# =========================================================
# Localized PPO update
# =========================================================

def localized_ppo_update(
    agent: PPOUnlearner,
    ref_agent: PPOUnlearner,
    batch,
    lambda_val: float,
    locality_beta: float,
    safe_cost_threshold: float,
    normalize_locality_loss: bool = True,
):
    """
    PPO update with far-state KL preservation.

    batch contains unflattened vector-env rollout:
        states:        [T, E, obs_dim]
        actions:       [T, E, act_dim]
        log_probs:     [T, E]
        rewards:       [T, E]
        costs:         [T, E]    # c_forget / hazard contact
        terminated:    [T, E]
        last_state:    [E, obs_dim]
        last_terminated: [E]

    Locality term:
        safe_mask = 1[cost <= safe_cost_threshold]

        L_locality = E_safe[ KL(pi_current(.|s) || pi_unsafe(.|s)) ]

    This penalizes policy movement on non-hazard states.
    """
    device = agent.device

    states = torch.as_tensor(
        batch["states"],
        dtype=torch.float32,
        device=device,
    )
    actions = torch.as_tensor(
        batch["actions"],
        dtype=torch.float32,
        device=device,
    )
    old_logp = torch.as_tensor(
        batch["log_probs"],
        dtype=torch.float32,
        device=device,
    )

    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    costs = np.asarray(batch["costs"], dtype=np.float32)
    terminated = np.asarray(batch["terminated"], dtype=np.bool_)

    T, E = rewards.shape

    # -----------------------------------------------------
    # Compute reward/cost values and GAE
    # -----------------------------------------------------
    with torch.no_grad():
        flat_states = states.reshape(T * E, -1)

        _, _, v_r_flat, v_c_flat = agent._forward_policy(flat_states)

        v_r = v_r_flat.reshape(T, E)
        v_c = v_c_flat.reshape(T, E)

        last_state = torch.as_tensor(
            batch["last_state"],
            dtype=torch.float32,
            device=device,
        )

        _, _, last_vr, last_vc = agent._forward_policy(last_state)

        last_vr = last_vr.detach().cpu().numpy().astype(np.float32)
        last_vc = last_vc.detach().cpu().numpy().astype(np.float32)

        last_term = np.asarray(batch["last_terminated"], dtype=np.bool_)

        next_vr = np.where(last_term, 0.0, last_vr)
        next_vc = np.where(last_term, 0.0, last_vc)

        adv_r = compute_gae_te(
            rewards,
            v_r,
            next_vr,
            agent.gamma,
            agent.gae_lambda,
            terminated,
        ).to(device)

        adv_c = compute_gae_te(
            costs,
            v_c,
            next_vc,
            agent.gamma,
            agent.gae_lambda,
            terminated,
        ).to(device)

        ret_r = adv_r + v_r
        ret_c = adv_c + v_c

        adv_r_n = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)

        adv_c_std = adv_c.std()
        if adv_c_std > 1e-8:
            adv_c_n = (adv_c - adv_c.mean()) / (adv_c_std + 1e-8)
        else:
            adv_c_n = adv_c - adv_c.mean()

        adv = adv_r_n - float(lambda_val) * adv_c_n

    # -----------------------------------------------------
    # Flatten batch
    # -----------------------------------------------------
    states_f = states.reshape(T * E, -1)
    actions_f = actions.reshape(T * E, -1)
    old_logp_f = old_logp.reshape(T * E)

    adv_f = adv.reshape(T * E)
    ret_r_f = ret_r.reshape(T * E)
    ret_c_f = ret_c.reshape(T * E)

    costs_t = torch.as_tensor(
        costs,
        dtype=torch.float32,
        device=device,
    )
    safe_mask_f = (costs_t.reshape(T * E) <= safe_cost_threshold).float()

    N = states_f.shape[0]
    idx = np.arange(N)

    last_loss = 0.0
    last_pi_loss = 0.0
    last_vf_loss = 0.0
    last_entropy = 0.0
    last_locality_kl = 0.0
    last_approx_kl = 0.0
    stop_early = False

    # -----------------------------------------------------
    # PPO optimization
    # -----------------------------------------------------
    for _ in range(agent.ppo_epochs):
        np.random.shuffle(idx)

        for start in range(0, N, agent.batch_size):
            mb = idx[start:start + agent.batch_size]

            s = states_f[mb]
            a = actions_f[mb]
            oldlp = old_logp_f[mb]
            A = adv_f[mb]
            Rr = ret_r_f[mb]
            Rc = ret_c_f[mb]
            safe_w = safe_mask_f[mb]

            mean, std, vr_new, vc_new = agent._forward_policy(s)

            u = agent._unsquash(a)
            newlp = agent._logp(mean, std, u, a)

            ratio = torch.exp(newlp - oldlp)

            surr1 = ratio * A
            surr2 = torch.clamp(
                ratio,
                1.0 - agent.clip_eps,
                1.0 + agent.clip_eps,
            ) * A

            pi_loss = -torch.min(surr1, surr2).mean()

            vf_loss = (
                0.5 * ((vr_new - Rr) ** 2).mean()
                + 0.5 * ((vc_new - Rc) ** 2).mean()
            )

            entropy = Normal(mean, std).entropy().sum(dim=-1).mean()

            # -------------------------------------------------
            # Locality regularizer:
            # preserve unsafe expert on safe / far states.
            # -------------------------------------------------
            with torch.no_grad():
                ref_mean, ref_std, _, _ = ref_agent._forward_policy(s)

            kl_cur_ref = gaussian_kl_diag(
                mean_p=mean,
                std_p=std,
                mean_q=ref_mean,
                std_q=ref_std,
            )

            if normalize_locality_loss:
                locality_kl = (safe_w * kl_cur_ref).sum() / (safe_w.sum() + 1e-8)
            else:
                locality_kl = (safe_w * kl_cur_ref).mean()

            loss = (
                pi_loss
                + agent.vf_coef * vf_loss
                - agent.ent_coef * entropy
                + float(locality_beta) * locality_kl
            )

            agent.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.max_grad_norm)
            agent.opt.step()

            with torch.no_grad():
                approx_kl = (oldlp - newlp).mean().item()

            last_loss = float(loss.item())
            last_pi_loss = float(pi_loss.item())
            last_vf_loss = float(vf_loss.item())
            last_entropy = float(entropy.item())
            last_locality_kl = float(locality_kl.item())
            last_approx_kl = float(approx_kl)

            if abs(approx_kl) > 1.5 * agent.target_kl:
                stop_early = True
                break

        if stop_early:
            break

    return {
        "loss": last_loss,
        "pi_loss": last_pi_loss,
        "vf_loss": last_vf_loss,
        "entropy": last_entropy,
        "locality_kl": last_locality_kl,
        "approx_kl": last_approx_kl,
        "safe_fraction": float(safe_mask_f.mean().detach().cpu().item()),
    }


# =========================================================
# Training loop
# =========================================================

def train_localized_concept(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_localized_concept_{args.env}_Seed{args.seed}",
        group=f"Unlearning_localized_concept_{args.env}",
        config=vars(args),
    )

    set_seed(args.seed)

    env = make_env(args.env, n_envs=args.n_envs)

    # -----------------------------------------------------
    # Trainable agent initialized from unsafe expert
    # -----------------------------------------------------
    agent = PPOUnlearner(
        env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        target_kl=args.target_kl,
    )
    unsafe_path = model_checkpoint_path("unsafe", args.env)
    agent.load(unsafe_path)

    # -----------------------------------------------------
    # Frozen unsafe reference for far-state preservation
    # -----------------------------------------------------
    ref_agent = PPOUnlearner(env)
    ref_agent.load(unsafe_path)
    ref_agent.freeze_policy()

    pid = PIDLagrangian(
        goal_cost=args.target_cost,
        Kp=args.kp,
        Ki=args.ki,
        Kd=args.kd,
        lambda_init=args.lambda_init,
        lambda_max=args.lambda_max,
        integral_max=args.integral_max,
    )

    obs = reset_env_with_seed(env, args.seed)

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

        # -------------------------------------------------
        # Rollout collection
        # -------------------------------------------------
        for _ in range(args.horizon):
            act, logp, _, _ = agent.act(obs, deterministic=False)

            step_act = act if is_vector_env(args) else act[0]

            next_obs, rew, cost, term, trunc, info = env.step(step_act)

            c_forget = extract_cforget(cost, info)

            # Make single-env case look like vector env with E=1.
            obs_arr = as_float32_array(obs)
            act_arr = as_float32_array(act)
            logp_arr = as_float32_array(logp)
            rew_arr = as_float32_array(rew).reshape(-1)
            cforget_arr = as_float32_array(c_forget).reshape(-1)
            term_arr = as_bool_array(term).reshape(-1)

            if obs_arr.ndim == 1:
                obs_arr = obs_arr[None, :]

            if act_arr.ndim == 1:
                act_arr = act_arr[None, :]

            if logp_arr.ndim == 0:
                logp_arr = logp_arr.reshape(1)

            buf["states"].append(obs_arr)
            buf["actions"].append(act_arr)
            buf["logp"].append(logp_arr)
            buf["rewards"].append(rew_arr)
            buf["costs"].append(cforget_arr)
            buf["terminated"].append(term_arr)

            batch_rew_sum += float(rew_arr.mean())
            batch_cforget_sum += float(cforget_arr.mean())

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

        last_state = as_float32_array(obs)
        if last_state.ndim == 1:
            last_state = last_state[None, :]

        batch = {
            "states": states,
            "actions": actions,
            "log_probs": logp,
            "rewards": rewards,
            "costs": costs,
            "terminated": terminated,
            "last_state": last_state,
            "last_terminated": np.asarray(terminated[-1], dtype=np.bool_),
        }

        update_info = localized_ppo_update(
            agent=agent,
            ref_agent=ref_agent,
            batch=batch,
            lambda_val=lambda_val,
            locality_beta=args.locality_beta,
            safe_cost_threshold=args.safe_cost_threshold,
            normalize_locality_loss=not args.no_normalize_locality_loss,
        )

        wandb.log(
            {
                "Train/Update": update,
                "Train/Lambda": lambda_val,
                "Train/Batch_MeanRewardStep": mean_reward_step,
                "Train/Batch_Cost_Rate": mean_cost_rate,
                "Train/PID_Error": mean_cost_rate - args.target_cost,
                "Train/Loss": update_info["loss"],
                "Train/PiLoss": update_info["pi_loss"],
                "Train/VfLoss": update_info["vf_loss"],
                "Train/Entropy": update_info["entropy"],
                "Train/ApproxKL": update_info["approx_kl"],
                "Localized/LocalityKL": update_info["locality_kl"],
                "Localized/SafeFraction": update_info["safe_fraction"],
                "Localized/Beta": args.locality_beta,
                "Compute/EnvStepsTotal": env_steps_total,
            },
            step=env_steps_total,
        )

        print(
            f"[LOCALIZED_CONCEPT] Update {update}/{args.updates} | "
            f"Rew {mean_reward_step:.4f} | "
            f"Cost {mean_cost_rate:.4f} | "
            f"Lambda {lambda_val:.4f} | "
            f"LocKL {update_info['locality_kl']:.6f} | "
            f"Loss {update_info['loss']:.4f}"
        )

        if update in save_points:
            path = localized_concept_checkpoint_path(args.env, update)

            payload = {
                "state_dict": agent.policy.state_dict(),
                "meta": {
                    "method": "localized_concept",
                    "env": args.env,
                    "seed": args.seed,
                    "update": update,
                    "unsafe_reference": unsafe_path,
                    "locality_beta": args.locality_beta,
                    "safe_cost_threshold": args.safe_cost_threshold,
                    "target_cost": args.target_cost,
                    "lambda_val": float(lambda_val),
                    "mean_reward_step": float(mean_reward_step),
                    "mean_cost_rate": float(mean_cost_rate),
                },
            }

            torch.save(payload, path)
            wandb.save(path)

    env.close()
    wandb.finish()


# =========================================================
# CLI
# =========================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed", type=int, default=0)

    # PID parameters.
    # Slightly gentler defaults than vanilla concept to reduce reward collapse.
    p.add_argument("--kp", type=float, default=1.0)
    p.add_argument("--ki", type=float, default=0.003)
    p.add_argument("--kd", type=float, default=0.03)
    p.add_argument("--lambda_init", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=50.0)
    p.add_argument("--integral_max", type=float, default=5.0)
    p.add_argument("--target_cost", type=float, default=0.03)

    # Locality regularization.
    p.add_argument(
        "--locality_beta",
        type=float,
        default=0.05,
        help=(
            "Strength of KL preservation on non-hazard states. "
            "Try 0.01, 0.03, 0.05, 0.1, 0.3."
        ),
    )
    p.add_argument(
        "--safe_cost_threshold",
        type=float,
        default=0.0,
        help=(
            "States with c_forget <= threshold are treated as safe/far "
            "for KL preservation."
        ),
    )
    p.add_argument(
        "--no_normalize_locality_loss",
        action="store_true",
        help=(
            "If set, use mean weighted KL instead of normalizing by number "
            "of safe states."
        ),
    )

    # Rollout/training parameters.
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--unsafe_updates", type=int, default=300)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--target_kl", type=float, default=0.02)

    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    args = p.parse_args()
    train_localized_concept(args)