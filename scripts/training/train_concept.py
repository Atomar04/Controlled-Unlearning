import argparse
import numpy as np
import wandb

from reifule.algorithm import PPOUnlearner
from reifule.computation_amnesiac import PIDLagrangian
from reifule.utils import (
    make_env,
    set_seed,
    extract_cforget,
    model_checkpoint_path,
)


def train_concept(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_concept_{args.env}_Seed{args.seed}",
        group=f"Unlearning_concept_{args.env}",
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
            path = model_checkpoint_path("concept", args.env, update)
            agent.save(path)
            wandb.save(path)

    env.close()
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--kp", type=float, default=2.0)
    p.add_argument("--ki", type=float, default=0.01)
    p.add_argument("--kd", type=float, default=0.1)
    p.add_argument("--lambda_init", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=200.0)
    p.add_argument("--integral_max", type=float, default=5.0)
    p.add_argument("--target_cost", type=float, default=0.03)

    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--unsafe_updates", type=int, default=300)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)

    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    args = p.parse_args()
    train_concept(args)