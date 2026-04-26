import argparse
from typing import Dict

import numpy as np
import torch
import wandb

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.utils import (
    make_env,
    set_seed,
    extract_cforget,
    model_checkpoint_path,
    probe_artifact_path,
    save_repedit_checkpoint,
)


def _to_float(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def load_repedit_artifact(path: str):
    """
    Loads a hazard concept direction from either:
      - .pt / .pth torch file
      - .npz numpy archive

    Supported keys:
      direction, u, vector, probe_weight, w

    Optional metadata:
      tau, alpha, beta
    """
    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        keys = ["direction", "u", "vector", "probe_weight", "w"]

        direction = None
        for k in keys:
            if k in data:
                direction = np.asarray(data[k], dtype=np.float32)
                break

        if direction is None:
            raise KeyError(
                f"No direction key found in {path}. "
                f"Expected one of {keys}."
            )

        meta = {}
        for k in ["tau", "alpha", "beta"]:
            if k in data:
                meta[k] = float(np.asarray(data[k]).reshape(-1)[0])

        return direction, meta

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported repedit artifact format in {path}")

    keys = ["direction", "u", "vector", "probe_weight", "w"]
    direction = None

    for k in keys:
        if k in obj:
            val = obj[k]
            if isinstance(val, torch.Tensor):
                direction = val.detach().cpu().float().numpy()
            else:
                direction = np.asarray(val, dtype=np.float32)
            break

    if direction is None:
        raise KeyError(
            f"No direction key found in {path}. "
            f"Expected one of {keys}."
        )

    meta = {}
    for k in ["tau", "alpha", "beta"]:
        if k in obj:
            meta[k] = _to_float(obj[k])

    return direction, meta


def collect_repedit_calibration_states(
    unsafe_agent: PPOUnlearner,
    env_id: str,
    seed: int,
    n_steps: int,
):
    """
    Collect states from the fixed unsafe expert to calibrate the editor threshold.
    """
    env = make_env(env_id, n_envs=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    obs, info = env.reset()
    states = []
    labels = []

    steps = 0
    while steps < n_steps:
        act, _, _, _ = unsafe_agent.act(obs, deterministic=False)
        act = act[0] if act.ndim == 2 else act

        next_obs, rew, cost, term, trunc, info = env.step(act)
        c_forget = extract_cforget(cost, info)
        c_forget = float(np.asarray(c_forget).reshape(-1)[0])

        states.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        labels.append(c_forget)

        obs = next_obs
        steps += 1

        if term or trunc:
            obs, info = env.reset()

    env.close()

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
    )


def calibrate_repedit_tau(
    agent: PPOUnlearner,
    direction: np.ndarray,
    states_np: np.ndarray,
    labels_np: np.ndarray,
    quantile: float = 0.80,
    score_batch_size: int = 4096,
):
    """
    Computes latent scores h·u on calibration states and sets tau by quantile.

    Preference:
      - use hazard-positive states if enough exist
      - otherwise fall back to all states
    """
    device = agent.device

    d = torch.as_tensor(direction, dtype=torch.float32, device=device)
    d = d / (torch.norm(d) + 1e-8)

    scores = []
    with torch.no_grad():
        for start in range(0, len(states_np), score_batch_size):
            s = torch.as_tensor(
                states_np[start:start + score_batch_size],
                dtype=torch.float32,
                device=device,
            )
            feat = agent.policy.encode(s)
            sc = feat @ d
            scores.append(sc.detach().cpu().numpy())

    scores = np.concatenate(scores, axis=0)

    pos_mask = labels_np > 0.5
    if np.sum(pos_mask) >= 32:
        ref_scores = scores[pos_mask]
    else:
        ref_scores = scores

    tau = float(np.quantile(ref_scores, quantile))
    stats = {
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "tau": tau,
        "n_states": int(len(states_np)),
        "n_positive": int(np.sum(pos_mask)),
    }
    return tau, stats


def latent_score_stats(
    agent: PPOUnlearner,
    direction: np.ndarray,
    obs_batch: np.ndarray,
):
    device = agent.device
    d = torch.as_tensor(direction, dtype=torch.float32, device=device)
    d = d / (torch.norm(d) + 1e-8)

    with torch.no_grad():
        s = torch.as_tensor(obs_batch, dtype=torch.float32, device=device)
        feat = agent.policy.encode(s)
        scores = feat @ d

    return {
        "Latent/ScoreMean": float(scores.mean().item()),
        "Latent/ScoreStd": float(scores.std(unbiased=False).item()),
        "Latent/ScoreMax": float(scores.max().item()),
        "Latent/ScoreMin": float(scores.min().item()),
    }


def train_repedit(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_repedit_{args.env}_Seed{args.seed}",
        group=f"Unlearning_repedit_{args.env}",
        config=vars(args),
    )

    set_seed(args.seed)

    env = make_env(args.env, n_envs=args.n_envs)

    agent = PPOUnlearner(
        env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )
    agent.load(model_checkpoint_path("unsafe", args.env))

    direction, artifact_meta = load_repedit_artifact(args.repedit_artifact)

    if args.repedit_tau is not None:
        tau = float(args.repedit_tau)
        tau_stats = {
            "tau": tau,
            "score_mean": float("nan"),
            "score_std": float("nan"),
            "score_min": float("nan"),
            "score_max": float("nan"),
            "n_states": 0,
            "n_positive": 0,
        }
        print(f"[REPEDIT] Using manual tau = {tau:.6f}")

    elif args.repedit_use_artifact_tau and ("tau" in artifact_meta):
        tau = float(artifact_meta["tau"])
        tau_stats = {
            "tau": tau,
            "score_mean": float("nan"),
            "score_std": float("nan"),
            "score_min": float("nan"),
            "score_max": float("nan"),
            "n_states": 0,
            "n_positive": 0,
        }
        print(f"[REPEDIT] Using artifact tau = {tau:.6f}")

    else:
        print("\n[REPEDIT] Collecting calibration states from unsafe expert...")
        calib_states, calib_labels = collect_repedit_calibration_states(
            unsafe_agent=agent,
            env_id=args.env,
            seed=args.seed,
            n_steps=args.repedit_calib_steps,
        )
        tau, tau_stats = calibrate_repedit_tau(
            agent=agent,
            direction=direction,
            states_np=calib_states,
            labels_np=calib_labels,
            quantile=args.repedit_tau_quantile,
        )
        print(
            f"[REPEDIT] Calibrated tau = {tau:.6f} | "
            f"states = {tau_stats['n_states']} | "
            f"hazard positives = {tau_stats['n_positive']}"
        )

    editor = RepresentationEditor(
        direction=direction,
        alpha=args.repedit_alpha,
        tau=tau,
        beta=args.repedit_beta,
    )
    agent.set_editor(editor)
    agent.freeze_policy()
    agent.editor.eval()

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    env_steps_total = args.unsafe_updates * args.horizon * args.n_envs
    save_points = set(args.save_points)

    # Static edit: no optimizer updates, just rollout/log/save.
    for update in range(1, args.updates + 1):
        batch_rew_sum = 0.0
        batch_cforget_sum = 0.0
        obs_buf = []

        for _ in range(args.horizon):
            act, logp, _, _ = agent.act(
                obs,
                deterministic=args.repedit_deterministic,
            )
            next_obs, rew, cost, term, trunc, info = env.step(act)

            c_forget = extract_cforget(cost, info)

            batch_rew_sum += float(np.asarray(rew, dtype=np.float32).mean())
            batch_cforget_sum += float(np.asarray(c_forget, dtype=np.float32).mean())
            obs_buf.append(np.asarray(obs, dtype=np.float32))

            obs = next_obs
            env_steps_total += args.n_envs

        mean_reward_step = batch_rew_sum / args.horizon
        mean_cost_rate = batch_cforget_sum / args.horizon

        obs_batch = np.concatenate(obs_buf, axis=0)
        score_logs = latent_score_stats(agent, direction, obs_batch)

        wandb.log(
            {
                "Train/Update": update,
                "Train/Batch_MeanRewardStep": mean_reward_step,
                "Train/Batch_Cost_Rate": mean_cost_rate,
                "RepEdit/Alpha": float(agent.editor.alpha.detach().cpu().item()),
                "RepEdit/Tau": float(agent.editor.tau.detach().cpu().item()),
                "RepEdit/Beta": float(agent.editor.beta),
                "Calibration/Tau": float(tau_stats["tau"]),
                "Calibration/ScoreMean": float(tau_stats["score_mean"]),
                "Calibration/ScoreStd": float(tau_stats["score_std"]),
                "Calibration/ScoreMin": float(tau_stats["score_min"]),
                "Calibration/ScoreMax": float(tau_stats["score_max"]),
                "Calibration/NStates": int(tau_stats["n_states"]),
                "Calibration/NPositive": int(tau_stats["n_positive"]),
                "Compute/EnvStepsTotal": env_steps_total,
                **score_logs,
            },
            step=env_steps_total,
        )

        print(
            f"[REPEDIT] Update {update}/{args.updates} | "
            f"Rew {mean_reward_step:.4f} | "
            f"Cost {mean_cost_rate:.4f} | "
            f"Alpha {float(agent.editor.alpha.detach().cpu().item()):.4f} | "
            f"Tau {float(agent.editor.tau.detach().cpu().item()):.4f}"
        )

        if update in save_points:
            path = model_checkpoint_path("repedit", args.env, update)
            save_repedit_checkpoint(
                path=path,
                agent=agent,
                editor=agent.editor,
                extra_meta={
                    "mode": "repedit",
                    "env": args.env,
                    "seed": args.seed,
                    "artifact_path": args.repedit_artifact,
                    "tau_source": (
                        "manual"
                        if args.repedit_tau is not None
                        else "artifact"
                        if args.repedit_use_artifact_tau and ("tau" in artifact_meta)
                        else "calibrated"
                    ),
                },
            )
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

    p.add_argument(
        "--repedit_artifact",
        type=str,
        default=probe_artifact_path("repedit_direction.pt"),
    )
    p.add_argument("--repedit_alpha", type=float, default=1.0)
    p.add_argument("--repedit_beta", type=float, default=10.0)
    p.add_argument("--repedit_tau", type=float, default=None)
    p.add_argument("--repedit_tau_quantile", type=float, default=0.80)
    p.add_argument("--repedit_calib_steps", type=int, default=5000)
    p.add_argument("--repedit_use_artifact_tau", action="store_true")
    p.add_argument("--repedit_deterministic", action="store_true")

    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    args = p.parse_args()
    train_repedit(args)