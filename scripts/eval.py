import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import argparse
import numpy as np
import wandb
import gymnasium as gym
import safety_gymnasium
import imageio
import torch
from reifule.algorithm import PPOUnlearner


def make_single_env(env_id: str, render_mode=None):
    if "Safety" in env_id:
        return safety_gymnasium.make(env_id, render_mode=render_mode)
    return gym.make(env_id, render_mode=render_mode)


@torch.no_grad()
def run_episode(env, agent, use_cforget=True, max_steps=2000):
    obs, info = env.reset()
    done = False

    ep_ret = 0.0
    ep_cost_sum = 0.0
    ep_cforget_sum = 0.0
    ep_len = 0

    while not done and ep_len < max_steps:
        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        step = env.step(act)
        if len(step) == 6:
            obs, rew, cost, term, trunc, info = step
        else:
            obs, rew, term, trunc, info = step
            cost = 0.0

        done = bool(term or trunc)

        ep_ret += float(rew)
        ep_cost_sum += float(cost)
        ep_len += 1

        if use_cforget:
            if isinstance(info, dict) and "cost_hazards" in info:
                ep_cforget_sum += 1.0 if float(info["cost_hazards"]) > 0 else 0.0
            else:
                ep_cforget_sum += 1.0 if float(cost) > 0 else 0.0

    cforget_rate = ep_cforget_sum / max(1, ep_len)
    return ep_ret, ep_cost_sum, ep_cforget_sum, cforget_rate, ep_len


def record_video(env_id, agent, out_path, max_steps=1000, fps=30):
    env = make_single_env(env_id, render_mode="rgb_array")
    frames = []

    obs, info = env.reset()
    done = False
    t = 0

    while not done and t < max_steps:
        frames.append(env.render())

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        obs, rew, cost, term, trunc, info = env.step(act)
        done = bool(term or trunc)
        t += 1

    env.close()
    imageio.mimsave(out_path, frames, fps=fps)


def evaluate(args):
    wandb.init(project=args.project, name=f"Eval_{args.env}", config=vars(args))

    models = {
        "Unsafe_Expert": f"unsafe_expert_{args.env}.pt",
        "Unlearned_Agent": f"safe_agent_{args.env}.pt",
        "Oracle_Agent": f"oracle_agent_{args.env}.pt",
    }

    results = []

    for name, path in models.items():
        if not os.path.exists(path):
            print(f"[WARN] Missing: {path} (skipping {name})")
            continue

        env = make_single_env(args.env, render_mode=None)
        agent = PPOUnlearner(env)  # will infer obs/act dims from env
        agent.load(path)
        agent.policy.eval()

        ep_returns, ep_costs, ep_cforgets, ep_cforget_rates, ep_lens = [], [], [], [], []

        for _ in range(args.episodes):
            r, csum, cfsum, cfr, L = run_episode(env, agent, use_cforget=True)
            ep_returns.append(r)
            ep_costs.append(csum)
            ep_cforgets.append(cfsum)
            ep_cforget_rates.append(cfr)
            ep_lens.append(L)

        env.close()

        row = {
            "Model": name,
            "Reward": float(np.mean(ep_returns)),
            "Cost_Sum": float(np.mean(ep_costs)),
            "C_forget_Sum": float(np.mean(ep_cforgets)),
            "C_forget_Rate": float(np.mean(ep_cforget_rates)),
            "Episode_Length": float(np.mean(ep_lens)),
        }
        results.append(row)

        wandb.log({f"{name}/Mean_Reward": row["Reward"]})
        wandb.log({f"{name}/Mean_Cost_Sum": row["Cost_Sum"]})
        wandb.log({f"{name}/Mean_C_forget_Rate": row["C_forget_Rate"]})
        wandb.log({f"{name}/Mean_Episode_Length": row["Episode_Length"]})

        print(name, row)

        if args.video:
            vid_path = f"video_{name}_{args.env}.mp4"
            record_video(args.env, agent, vid_path, max_steps=args.video_steps, fps=args.fps)
            wandb.log({f"Video/{name}": wandb.Video(vid_path, fps=args.fps, format="mp4")})

    # log table
    import pandas as pd
    df = pd.DataFrame(results)
    print("\nFINAL COMPARISON:\n", df.to_string(index=False))
    wandb.log({"Eval/Table": wandb.Table(dataframe=df)})

    wandb.finish()


if __name__ == "__main__":
    import torch  # placed here to avoid importing torch if not running

    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reinforcement-Unlearning")
    p.add_argument("--episodes", type=int, default=50)

    p.add_argument("--video", action="store_true")
    p.add_argument("--video_steps", type=int, default=600)
    p.add_argument("--fps", type=int, default=30)

    args = p.parse_args()
    evaluate(args)
