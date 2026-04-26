import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.utils import make_env, artifact_dir, model_checkpoint_path


def load_agent_checkpoint(env, ckpt_path: str):
    """
    Loads:
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


def evaluate_cost(agent, env, n_episodes=5):
    total_cost = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        done = False

        while not done:
            act, _, _, _ = agent.act(obs, deterministic=True)
            act = act[0] if act.ndim == 2 else act

            obs, rew, cost, term, trunc, info = env.step(act)

            total_cost += float(np.mean(cost))
            total_steps += 1

            done = bool(term or trunc)

    return total_cost / max(total_steps, 1)


def relearn(
    agent,
    train_env,
    eval_env,
    threshold,
    horizon=1024,
    max_updates=100,
    eval_every=5,
    eval_episodes=5,
):
    reset_out = train_env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    global_steps = 0
    start_time = time.time()
    history = []

    print("\n===== START RELEARNING =====")

    # Important:
    # relearning should optimize the base policy only.
    # If a repedit checkpoint was loaded, disable the editor for relearning.
    if getattr(agent, "editor", None) is not None:
        print("[INFO] Loaded model has editor attached; clearing editor before relearning.")
        agent.clear_editor()

    for update in range(1, max_updates + 1):
        update_start = time.time()

        buf = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "costs": [],
            "terminated": [],
        }

        for _ in range(horizon):
            act, logp, _, _ = agent.act(obs, deterministic=False)
            act = act[0] if act.ndim == 2 else act

            next_obs, rew, cost, term, trunc, info = train_env.step(act)

            rew_scalar = float(np.mean(rew))
            cost_scalar = float(np.mean(cost))
            term_bool = bool(np.any(term))
            trunc_bool = bool(np.any(trunc))

            buf["states"].append(np.asarray(obs, dtype=np.float32))
            buf["actions"].append(np.asarray(act, dtype=np.float32))
            buf["log_probs"].append(float(logp[0] if np.ndim(logp) > 0 else logp))
            buf["rewards"].append(rew_scalar)
            buf["costs"].append(cost_scalar)
            buf["terminated"].append(term_bool)

            global_steps += 1

            if term_bool or trunc_bool:
                reset_out = train_env.reset()
                obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            else:
                obs = next_obs

        batch = {
            "states": np.asarray(buf["states"], dtype=np.float32)[:, None, :],
            "actions": np.asarray(buf["actions"], dtype=np.float32)[:, None, :],
            "log_probs": np.asarray(buf["log_probs"], dtype=np.float32)[:, None],
            "rewards": np.asarray(buf["rewards"], dtype=np.float32)[:, None],
            "costs": np.asarray(buf["costs"], dtype=np.float32)[:, None],
            "terminated": np.asarray(buf["terminated"], dtype=np.bool_)[:, None],
            "last_state": np.asarray(obs, dtype=np.float32)[None, :],
            "last_terminated": np.asarray([False], dtype=np.bool_),
        }

        loss = agent.update(batch, lambda_val=0.0)

        update_time = time.time() - update_start
        total_time = time.time() - start_time

        row = {
            "update": update,
            "global_steps": global_steps,
            "loss": float(loss),
            "update_time_sec": float(update_time),
            "total_time_sec": float(total_time),
        }

        if update % eval_every == 0:
            cost_eval = evaluate_cost(agent, eval_env, n_episodes=eval_episodes)
            row["eval_cost"] = float(cost_eval)

            print(
                f"[Update {update}] "
                f"Cost: {cost_eval:.4f} | "
                f"Steps: {global_steps} | "
                f"Loss: {loss:.4f} | "
                f"UpdateTime: {update_time:.2f}s | "
                f"TotalTime: {total_time:.2f}s"
            )

            history.append(row)

            if cost_eval >= threshold:
                print("\n===== THRESHOLD REACHED =====")
                return {
                    "threshold_reached": True,
                    "steps_to_threshold": int(global_steps),
                    "relearn_time_sec": float(total_time),
                    "history": history,
                }
        else:
            history.append(row)

    print("\n===== MAX UPDATES REACHED =====")
    return {
        "threshold_reached": False,
        "steps_to_threshold": int(global_steps),
        "relearn_time_sec": float(time.time() - start_time),
        "history": history,
    }


def main(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_id = args.eval_id or f"{args.env}__relearn__{timestamp}"

    out_dir = artifact_dir("eval", "relearn_test_eval", eval_id)
    train_env = make_env(args.env, n_envs=1)
    eval_env = make_env(args.env, n_envs=1)

    print("\n===== INITIALIZING =====")

    unsafe_model_path = args.unsafe_model or model_checkpoint_path("unsafe", args.env)
    unsafe_agent, _ = load_agent_checkpoint(eval_env, unsafe_model_path)

    unsafe_cost = evaluate_cost(unsafe_agent, eval_env, n_episodes=args.threshold_episodes)
    print(f"Unsafe cost ({args.threshold_episodes} eps): {unsafe_cost:.4f}")

    recommended_threshold = 0.9 * unsafe_cost

    if args.compute_threshold_only:
        result = {
            "env": args.env,
            "unsafe_model": unsafe_model_path,
            "unsafe_cost": float(unsafe_cost),
            "recommended_threshold_0p9x": float(recommended_threshold),
            "threshold_episodes": int(args.threshold_episodes),
        }

        with open(out_dir / "threshold_only.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"Recommended threshold (0.9x): {recommended_threshold:.4f}")
        print(f"Saved threshold info to: {out_dir / 'threshold_only.json'}")

        train_env.close()
        eval_env.close()
        return

    if args.threshold is None:
        raise ValueError("You must pass --threshold, or use --compute_threshold_only first.")

    threshold = args.threshold
    print(f"Using FIXED threshold: {threshold:.4f}")

    if args.model_path == "random":
        print("\nUsing RANDOM INIT model")
        agent = PPOUnlearner(train_env)
        model_meta = {"model_path": "random", "has_editor": False}
    else:
        print(f"\nLoading model: {args.model_path}")
        agent, ckpt = load_agent_checkpoint(train_env, args.model_path)
        model_meta = {
            "model_path": args.model_path,
            "has_editor": bool("editor" in ckpt and ckpt["editor"] is not None),
        }

    total_start = time.time()

    result = relearn(
        agent=agent,
        train_env=train_env,
        eval_env=eval_env,
        threshold=threshold,
        horizon=args.horizon,
        max_updates=args.max_updates,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
    )

    total_wallclock = time.time() - total_start

    print("\n===== FINAL RESULT =====")
    print(f"Threshold reached: {result['threshold_reached']}")
    print(f"Steps to threshold: {result['steps_to_threshold']}")
    print(f"Relearn time (internal): {result['relearn_time_sec']:.2f} sec")
    print(f"Total wall-clock time: {total_wallclock:.2f} sec")

    history_df = pd.DataFrame(result["history"])
    history_csv = out_dir / "relearn_history.csv"
    history_df.to_csv(history_csv, index=False)

    final_result = {
        "eval_id": eval_id,
        "env": args.env,
        "threshold": float(threshold),
        "unsafe_model": unsafe_model_path,
        "unsafe_cost": float(unsafe_cost),
        "recommended_threshold_0p9x": float(recommended_threshold),
        "model": model_meta,
        "threshold_reached": bool(result["threshold_reached"]),
        "steps_to_threshold": int(result["steps_to_threshold"]),
        "relearn_time_sec": float(result["relearn_time_sec"]),
        "total_wallclock_sec": float(total_wallclock),
        "horizon": int(args.horizon),
        "max_updates": int(args.max_updates),
        "eval_every": int(args.eval_every),
        "eval_episodes": int(args.eval_episodes),
        "threshold_episodes": int(args.threshold_episodes),
        "history_csv": str(history_csv),
    }

    with open(out_dir / "relearn_result.json", "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2)

    print(f"Saved relearn outputs to: {out_dir}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--eval_id", type=str, default=None)

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint, or 'random'",
    )
    parser.add_argument(
        "--unsafe_model",
        type=str,
        default=None,
        help="Defaults to artifacts/models/unsafe/unsafe_expert_<env>.pt",
    )

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--max_updates", type=int, default=100)

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--threshold_episodes", type=int, default=50)

    parser.add_argument("--compute_threshold_only", action="store_true")

    args = parser.parse_args()
    main(args)