import time
import argparse
import numpy as np

from reifule.algorithm import PPOUnlearner
from scripts.train_unsafe import make_env


def evaluate_cost(agent, env, n_episodes=5):
    total_cost = 0.0
    total_steps = 0

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            act, _, _, _ = agent.act(obs, deterministic=True)
            act = act[0] if act.ndim == 2 else act

            obs, rew, cost, term, trunc, info = env.step(act)

            total_cost += float(np.mean(cost))
            total_steps += 1

            done = bool(term or trunc)

    return total_cost / total_steps

def relearn(agent, train_env, eval_env, threshold, horizon=1024, max_updates=100):
    obs, _ = train_env.reset()
    global_steps = 0
    start_time = time.time()

    print("\n===== START RELEARNING =====")

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
                obs, _ = train_env.reset()
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

        agent.update(batch, lambda_val=0.0)

        update_time = time.time() - update_start
        total_time = time.time() - start_time

        if update % 5 == 0:
            cost_eval = evaluate_cost(agent, eval_env, n_episodes=5)

            print(
                f"[Update {update}] "
                f"Cost: {cost_eval:.4f} | "
                f"Steps: {global_steps} | "
                f"UpdateTime: {update_time:.2f}s | "
                f"TotalTime: {total_time:.2f}s"
            )

            if cost_eval >= threshold:
                print("\n===== THRESHOLD REACHED =====")
                return global_steps, total_time

    print("\n===== MAX UPDATES REACHED =====")
    return global_steps, time.time() - start_time


def main(args):
    train_env = make_env(args.env, n_envs=1)
    eval_env = make_env(args.env, n_envs=1)

    print("\n===== INITIALIZING =====")

    unsafe_agent = PPOUnlearner(eval_env)
    unsafe_agent.load(args.unsafe_model)

    # unsafe_cost_rate = evaluate_cost(unsafe_agent, eval_env, n_episodes=10)
    # threshold = args.threshold if args.threshold is not None else 0.9 * unsafe_cost_rate

    # print(f"Unsafe cost: {unsafe_cost_rate:.4f}")
    # print(f"Threshold: {threshold:.4f}")
    unsafe_cost = evaluate_cost(unsafe_agent, eval_env, n_episodes=50)

    print(f"Unsafe cost (50 eps): {unsafe_cost:.4f}")

    # --- Threshold-only mode ---
    if args.compute_threshold_only:
        print(f"Recommended threshold (0.9x): {0.9 * unsafe_cost:.4f}")
        return

    # --- Use fixed threshold ---
    if args.threshold is None:
        raise ValueError("You MUST pass --threshold after computing it once.")

    threshold = args.threshold

    print(f"Using FIXED threshold: {threshold:.4f}")

    if args.model_path == "random":
        print("\nUsing RANDOM INIT model")
        agent = PPOUnlearner(train_env)
    else:
        print(f"\nLoading model: {args.model_path}")
        agent = PPOUnlearner(train_env)
        agent.load(args.model_path)

    total_start = time.time()

    steps, time_taken = relearn(
        agent,
        train_env,
        eval_env,
        threshold,
        horizon=args.horizon,
        max_updates=args.max_updates,
    )

    total_wallclock = time.time() - total_start

    print("\n===== FINAL RESULT =====")
    print(f"Steps to threshold: {steps}")
    print(f"Relearn time (internal): {time_taken:.2f} sec")
    print(f"Total wall-clock time: {total_wallclock:.2f} sec")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model OR 'random'")
    parser.add_argument("--unsafe_model", type=str, default="unsafe_expert_SafetyPointGoal1-v0.pt")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--horizon", type=int, default=1024)
    parser.add_argument("--max_updates", type=int, default=100)
    parser.add_argument("--compute_threshold_only", action="store_true")
    args = parser.parse_args()
    main(args)
