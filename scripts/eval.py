import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import numpy as np
import wandb
import imageio
import torch
import pandas as pd
import glob

from reifule.algorithm import PPOUnlearner
from reifule.experiments import (
    gaussian_policy_kl,
    collect_states,
    behaviour_localization
)

from scripts.train_unsafe import make_env


@torch.no_grad()
def run_episode(env, agent, max_steps=2000):

    obs, info = env.reset()
    done = False

    ep_ret = 0.0
    ep_cost_sum = 0.0
    ep_cforget_sum = 0.0
    ep_len = 0

    while not done and ep_len < max_steps:

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        obs, rew, cost, term, trunc, info = env.step(act)

        done = bool(term or trunc)

        ep_ret += float(rew)
        ep_cost_sum += float(cost)
        ep_len += 1

        if isinstance(info, dict) and "cost_hazards" in info:
            ep_cforget_sum += 1.0 if float(info["cost_hazards"]) > 0 else 0.0
        else:
            ep_cforget_sum += 1.0 if float(cost) > 0 else 0.0

    cforget_rate = ep_cforget_sum / max(1, ep_len)

    return ep_ret, ep_cost_sum, ep_cforget_sum, cforget_rate, ep_len


def record_video(env, agent, out_path, max_steps=1000, fps=30):

    frames = []
    obs, _ = env.reset()
    done = False
    t = 0

    while not done and t < max_steps:

        frames.append(env.render())

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        obs, _, _, term, trunc, _ = env.step(act)

        done = bool(term or trunc)
        t += 1

    imageio.mimsave(out_path, frames, fps=fps)


def load_all_models(env, args):
    """
    Auto-discover all model checkpoints
    """

    model_paths = {}

    # Unsafe
    unsafe_path = f"unsafe_expert_{args.env}.pt"
    if os.path.exists(unsafe_path):
        model_paths["Unsafe_Expert"] = unsafe_path

    # Concept unlearning
    for p in sorted(glob.glob(f"safe_concept_{args.env}_*.pt")):
        name = "Concept_" + p.split("_")[-1].replace(".pt", "")
        model_paths[name] = p

    # Trajectory unlearning
    for p in sorted(glob.glob(f"safe_trajectory_{args.env}_*.pt")):
        name = "Trajectory_" + p.split("_")[-1].replace(".pt", "")
        model_paths[name] = p

    # Oracle
    for p in sorted(glob.glob(f"oracle_{args.env}_*.pt")):
        name = "Oracle_" + p.split("_")[-1].replace(".pt", "")
        model_paths[name] = p

    return model_paths


def evaluate(args):

    wandb.init(
        project=args.project,
        name=f"Eval_{args.env}",
        config=vars(args),
    )

    env = make_env(args.env, n_envs=1)

    model_paths = load_all_models(env, args)

    agents = {}
    results = []

    # ========================
    # LOAD + EVALUATE
    # ========================

    for name, path in model_paths.items():

        print(f"\nEvaluating {name}")

        agent = PPOUnlearner(env)
        agent.load(path)
        agent.policy.eval()

        agents[name] = agent

        ep_returns, ep_costs, ep_cfr, ep_lens = [], [], [], []

        for _ in range(args.episodes):

            r, csum, _, cfr, L = run_episode(env, agent)

            ep_returns.append(r)
            ep_costs.append(csum)
            ep_cfr.append(cfr)
            ep_lens.append(L)

        row = {
            "Model": name,
            "Reward": float(np.mean(ep_returns)),
            "Cost_Sum": float(np.mean(ep_costs)),
            "C_forget_Rate": float(np.mean(ep_cfr)),
            "Episode_Length": float(np.mean(ep_lens)),
        }

        results.append(row)

        wandb.log({
            f"{name}/Reward": row["Reward"],
            f"{name}/Cost": row["Cost_Sum"],
            f"{name}/C_forget_Rate": row["C_forget_Rate"],
        })

        print(row)

    # ========================
    # POLICY ANALYSIS
    # ========================

    if "Unsafe_Expert" in agents:

        print("\nRunning policy analysis...")

        states = collect_states(env, agents["Unsafe_Expert"], n_steps=5000)

        for name, agent in agents.items():

            if name == "Unsafe_Expert":
                continue

            kl = gaussian_policy_kl(
                agents["Unsafe_Expert"],
                agent,
                states
            )

            kl_near, kl_far = behaviour_localization(
                agents["Unsafe_Expert"],
                agent,
                states
            )

            wandb.log({
                f"{name}/KL_vs_Expert": kl,
                f"{name}/KL_Near": kl_near,
                f"{name}/KL_Far": kl_far,
            })

            print(f"{name} KL:", kl)
            print(f"{name} KL near:", kl_near)
            print(f"{name} KL far:", kl_far)

    # ========================
    # FINAL TABLE
    # ========================

    df = pd.DataFrame(results)
    print("\nFINAL COMPARISON:\n", df.to_string(index=False))

    wandb.log({"Eval/Table": wandb.Table(dataframe=df)})

    env.close()
    wandb.finish()


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule-eval")

    p.add_argument("--episodes", type=int, default=50)

    args = p.parse_args()

    evaluate(args)
