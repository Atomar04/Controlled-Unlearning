'''
python -m scripts.eval_analysis.eval_policy_suite --env SafetyPointGoal1-v0 --episodes 50 --record_videos
'''


import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import imageio
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import safety_gymnasium

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.experiments import (
    gaussian_policy_kl,
    collect_states,
    behaviour_localization,
)
from reifule.utils import (
    make_env,
    extract_cforget,
    artifact_dir,
    model_checkpoint_path,
)


# =========================================================
# Environment helpers
# =========================================================

def make_render_env(env_id: str):
    """
    Separate render-capable env for local rollout videos.
    """
    if "Safety" in env_id:
        return safety_gymnasium.make(env_id, render_mode="rgb_array")
    return gym.make(env_id, render_mode="rgb_array")


# =========================================================
# Checkpoint discovery / loading
# =========================================================

def _parse_update_from_name(path: str, env_id: str):
    stem = Path(path).stem
    m = re.search(rf"{re.escape(env_id)}_(\d+)$", stem)
    if m:
        return int(m.group(1))
    return None


def _build_label(family: str, update):
    if update is None:
        return family
    return f"{family}_{update}"


def discover_model_specs(env_id: str):
    """
    Discover checkpoints from the standardized artifacts tree.

    Returns a list of dicts:
      {
        "family": "unsafe" | "oracle" | "concept" | "trajectory_decremental" | "repedit",
        "label":  str,
        "path":   str,
        "update": int | None,
      }
    """
    specs = []

    # Unsafe expert
    unsafe_path = model_checkpoint_path("unsafe", env_id)
    if Path(unsafe_path).exists():
        specs.append(
            {
                "family": "unsafe",
                "label": "unsafe",
                "path": unsafe_path,
                "update": None,
            }
        )

    families = [
        ("oracle", r"oracle_.*_(\d+)\.pt$"),
        ("concept", r"safe_concept_.*_(\d+)\.pt$"),
        ("trajectory_decremental", r"safe_trajectory_decremental_.*_(\d+)\.pt$"),
        ("repedit", r"safe_repedit_.*_(\d+)\.pt$"),
    ]

    for family, _ in families:
        folder = artifact_dir("models", family)
        pattern = str(folder / "*.pt")

        for path in sorted(Path(folder).glob("*.pt")):
            update = _parse_update_from_name(str(path), env_id)
            label = _build_label(family, update)
            specs.append(
                {
                    "family": family,
                    "label": label,
                    "path": str(path),
                    "update": update,
                }
            )

    # Keep only checkpoints relevant to this env.
    filtered = []
    for spec in specs:
        if env_id in Path(spec["path"]).name or spec["family"] == "unsafe":
            filtered.append(spec)

    # Sort for stable ordering.
    family_order = {
        "unsafe": 0,
        "oracle": 1,
        "concept": 2,
        "trajectory_decremental": 3,
        "repedit": 4,
    }

    filtered.sort(
        key=lambda x: (
            family_order.get(x["family"], 999),
            -1 if x["update"] is None else x["update"],
            x["label"],
        )
    )
    return filtered


def select_oracle_reference(specs, explicit_path=None, env_id=None):
    """
    Oracle is NOT treated as gold truth.
    It is a constrained-from-scratch safe reference.

    We use the latest oracle checkpoint as the default oracle reference
    unless the user explicitly provides a path.
    """
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Oracle reference not found: {explicit_path}")
        update = _parse_update_from_name(str(path), env_id) if env_id is not None else None
        return {
            "family": "oracle",
            "label": _build_label("oracle_ref", update),
            "path": str(path),
            "update": update,
        }

    oracle_specs = [s for s in specs if s["family"] == "oracle"]
    if not oracle_specs:
        return None

    oracle_specs = sorted(
        oracle_specs,
        key=lambda x: -1 if x["update"] is None else x["update"],
    )
    return oracle_specs[-1]


def load_agent_for_eval(env, ckpt_path: str):
    """
    Loads a checkpoint for evaluation.

    Handles:
      - standard checkpoints with only state_dict
      - repedit checkpoints with extra editor metadata
    """
    agent = PPOUnlearner(env)
    ckpt = torch.load(ckpt_path, map_location=agent.device)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")

    agent.policy.load_state_dict(ckpt["state_dict"])
    agent.policy.eval()

    if "editor" in ckpt and ckpt["editor"] is not None:
        ed = ckpt["editor"]
        editor = RepresentationEditor(
            direction=ed["direction"],
            alpha=float(ed["alpha"]),
            tau=float(ed["tau"]),
            beta=float(ed["beta"]),
        )
        agent.set_editor(editor)
        agent.editor.eval()

    return agent, ckpt


# =========================================================
# Rollout helpers
# =========================================================

@torch.no_grad()
def run_episode(env, agent, max_steps=2000):
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
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

        c_forget = extract_cforget(cost, info)
        ep_cforget_sum += float(np.asarray(c_forget).reshape(-1)[0])

    cforget_rate = ep_cforget_sum / max(1, ep_len)

    return {
        "reward": ep_ret,
        "cost_sum": ep_cost_sum,
        "cforget_sum": ep_cforget_sum,
        "cforget_rate": cforget_rate,
        "episode_length": ep_len,
    }


def record_video(env, agent, out_path, max_steps=1000, fps=30):
    frames = []

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    done = False
    t = 0

    while not done and t < max_steps:
        frame = env.render()
        frames.append(frame)

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        obs, _, _, term, trunc, _ = env.step(act)
        done = bool(term or trunc)
        t += 1

    imageio.mimsave(out_path, frames, fps=fps)


def evaluate_agent(env, agent, episodes: int, max_steps: int):
    episode_rows = []

    for ep_idx in range(episodes):
        metrics = run_episode(env, agent, max_steps=max_steps)
        metrics["episode_idx"] = ep_idx
        episode_rows.append(metrics)

    df = pd.DataFrame(episode_rows)

    summary = {
        "Reward": float(df["reward"].mean()),
        "RewardStd": float(df["reward"].std(ddof=0)),
        "Cost_Sum": float(df["cost_sum"].mean()),
        "CostStd": float(df["cost_sum"].std(ddof=0)),
        "C_forget_Sum": float(df["cforget_sum"].mean()),
        "C_forget_Rate": float(df["cforget_rate"].mean()),
        "Episode_Length": float(df["episode_length"].mean()),
    }
    return summary, df


# =========================================================
# Main evaluation
# =========================================================

def evaluate(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_id = args.eval_id or f"{args.env}__eval__{timestamp}"

    eval_root = artifact_dir("eval", "policy_suite_eval", eval_id)
    videos_dir = artifact_dir("eval", "policy_suite_eval", eval_id, "videos")

    specs = discover_model_specs(args.env)
    if len(specs) == 0:
        raise RuntimeError(
            f"No checkpoints discovered for env={args.env} under artifacts/models/"
        )

    oracle_ref_spec = select_oracle_reference(
        specs=specs,
        explicit_path=args.oracle_ref_path,
        env_id=args.env,
    )

    env = make_env(args.env, n_envs=1)

    agents = {}
    ckpt_meta = {}
    rows = []
    per_episode_rows = []

    # -----------------------------------------------------
    # Load all agents
    # -----------------------------------------------------
    for spec in specs:
        print(f"\nLoading {spec['label']} from {spec['path']}")
        agent, ckpt = load_agent_for_eval(env, spec["path"])
        agents[spec["label"]] = agent
        ckpt_meta[spec["label"]] = ckpt

    if "unsafe" not in agents:
        raise RuntimeError("Unsafe expert checkpoint is required for evaluation.")

    unsafe_agent = agents["unsafe"]
    oracle_ref_agent = None
    oracle_ref_label = None

    if oracle_ref_spec is not None:
        oracle_ref_label = oracle_ref_spec["label"]
        oracle_ref_agent = agents.get(oracle_ref_label)
        if oracle_ref_agent is None:
            oracle_ref_agent, _ = load_agent_for_eval(env, oracle_ref_spec["path"])

    # -----------------------------------------------------
    # Reference state distributions for KL
    # -----------------------------------------------------
    print("\nCollecting unsafe-reference states for KL/localization...")
    states_unsafe = collect_states(env, unsafe_agent, n_steps=args.kl_states)

    states_oracle = None
    if oracle_ref_agent is not None:
        print("Collecting oracle-reference states for secondary KL...")
        states_oracle = collect_states(env, oracle_ref_agent, n_steps=args.kl_states)

    # -----------------------------------------------------
    # Evaluate all discovered models
    # -----------------------------------------------------
    for spec in specs:
        label = spec["label"]
        family = spec["family"]
        update = spec["update"]
        path = spec["path"]

        print(f"\nEvaluating {label}")
        agent = agents[label]

        summary, df_ep = evaluate_agent(
            env=env,
            agent=agent,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )

        # Primary policy-change analysis:
        # compare every model against the original unsafe policy
        # on unsafe rollout states.
        if label == "unsafe":
            kl_vs_unsafe = 0.0
            kl_near_vs_unsafe = 0.0
            kl_far_vs_unsafe = 0.0
            kl_ratio_vs_unsafe = 1.0
        else:
            kl_vs_unsafe = gaussian_policy_kl(
                unsafe_agent,
                agent,
                states_unsafe,
            )
            kl_near_vs_unsafe, kl_far_vs_unsafe = behaviour_localization(
                unsafe_agent,
                agent,
                states_unsafe,
            )
            kl_ratio_vs_unsafe = float(
                kl_near_vs_unsafe / (kl_far_vs_unsafe + 1e-8)
            )

        # Secondary reference:
        # oracle is NOT gold, but useful as constrained-from-scratch reference.
        if oracle_ref_agent is not None and states_oracle is not None:
            kl_vs_oracle = gaussian_policy_kl(
                oracle_ref_agent,
                agent,
                states_oracle,
            )
        else:
            kl_vs_oracle = float("nan")

        row = {
            "Model": label,
            "Family": family,
            "Update": update,
            "Path": path,
            **summary,
            "KL_vs_Unsafe": float(kl_vs_unsafe),
            "KL_Near_vs_Unsafe": float(kl_near_vs_unsafe),
            "KL_Far_vs_Unsafe": float(kl_far_vs_unsafe),
            "KL_Locality_Ratio_vs_Unsafe": float(kl_ratio_vs_unsafe),
            "KL_vs_OracleRef": float(kl_vs_oracle),
            "OracleRef": oracle_ref_label,
        }
        rows.append(row)

        df_ep = df_ep.copy()
        df_ep["Model"] = label
        df_ep["Family"] = family
        df_ep["Update"] = update
        per_episode_rows.append(df_ep)

        print(pd.Series(row).to_string())

        if args.record_videos:
            render_env = make_render_env(args.env)
            video_path = videos_dir / f"{label}.gif"
            try:
                print(f"Recording video -> {video_path}")
                record_video(
                    render_env,
                    agent,
                    out_path=str(video_path),
                    max_steps=args.video_max_steps,
                    fps=args.video_fps,
                )
            finally:
                render_env.close()

    # -----------------------------------------------------
    # Save outputs locally
    # -----------------------------------------------------
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(
        by=["Family", "Update"],
        key=lambda s: s.fillna(-1) if s.name == "Update" else s,
    ).reset_index(drop=True)

    episodes_df = pd.concat(per_episode_rows, axis=0, ignore_index=True)

    summary_csv = eval_root / "summary.csv"
    episodes_csv = eval_root / "per_episode.csv"
    summary_json = eval_root / "summary.json"
    meta_json = eval_root / "metadata.json"

    summary_df.to_csv(summary_csv, index=False)
    episodes_df.to_csv(episodes_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_df.to_dict(orient="records"), f, indent=2)

    metadata = {
        "eval_id": eval_id,
        "env": args.env,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "kl_states": args.kl_states,
        "record_videos": bool(args.record_videos),
        "oracle_reference": None if oracle_ref_spec is None else oracle_ref_spec["path"],
        "notes": {
            "primary_reference": "unsafe",
            "primary_kl_meaning": "How much each policy changed from the original unsafe expert on unsafe-state support.",
            "localization_meaning": "Near-vs-far hazard policy change relative to the unsafe expert.",
            "oracle_reference_role": "Secondary constrained-from-scratch safe reference, not gold truth.",
        },
        "discovered_models": rows,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved evaluation to: {eval_root}")
    print("\nFINAL COMPARISON:\n")
    print(summary_df.to_string(index=False))

    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--eval_id", type=str, default=None)

    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--kl_states", type=int, default=5000)

    p.add_argument("--oracle_ref_path", type=str, default=None)

    p.add_argument("--record_videos", action="store_true")
    p.add_argument("--video_max_steps", type=int, default=1000)
    p.add_argument("--video_fps", type=int, default=30)

    args = p.parse_args()
    evaluate(args)