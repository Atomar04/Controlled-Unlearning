import argparse
import numpy as np
import wandb

from reifule.algorithm import PPOUnlearner
from reifule.utils import make_env, model_checkpoint_path


def train_unsafe(args):
    wandb.init(
        project=args.project,
        name=f"Unsafe_{args.env}_Seed{args.seed}",
        group=f"Unsafe_{args.env}",
        config=vars(args),
    )

    env = make_env(args.env, n_envs=args.n_envs)
    np.random.seed(args.seed)

    agent = PPOUnlearner(
        env,
        lr=args.lr,
        ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
    )

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    global_step = 0

    for update in range(args.updates):
        buf = {
            "states": [],
            "actions": [],
            "logp": [],
            "rewards": [],
            "costs": [],
            "terminated": [],
            "truncated": [],
        }

        batch_rew_sum = 0.0
        batch_cost_sum = 0.0

        for _ in range(args.horizon):
            act, logp, _, _ = agent.act(obs, deterministic=False)

            step = env.step(act)
            next_obs, rew, cost, term, trunc, info = step

            buf["states"].append(obs)
            buf["actions"].append(act)
            buf["logp"].append(logp)
            buf["rewards"].append(rew)
            buf["costs"].append(cost)
            buf["terminated"].append(term)
            buf["truncated"].append(trunc)

            rew = np.asarray(rew, dtype=np.float32)
            cost = np.asarray(cost, dtype=np.float32)
            batch_rew_sum += float(rew.mean())
            batch_cost_sum += float(cost.mean())

            obs = next_obs
            global_step += args.n_envs

        mean_reward_step = batch_rew_sum / args.horizon
        mean_cost_rate = batch_cost_sum / args.horizon

        batch = {
            "states": np.asarray(buf["states"]),
            "actions": np.asarray(buf["actions"]),
            "log_probs": np.asarray(buf["logp"]),
            "rewards": np.asarray(buf["rewards"]),
            "costs": np.asarray(buf["costs"]),
            "terminated": np.asarray(buf["terminated"]),
            "last_state": obs,
            "last_terminated": np.asarray(buf["terminated"][-1]),
        }

        loss = agent.update(batch, lambda_val=0.0)

        wandb.log(
            {
                "Train/Update": update,
                "Train/Loss": loss,
                "Train/Batch_MeanRewardStep": mean_reward_step,
                "Train/Batch_Cost_Rate": mean_cost_rate,
            },
            step=global_step,
        )

        print(
            f"Update {update+1}/{args.updates} | "
            f"Loss {loss:.4f} | "
            f"StepRew {mean_reward_step:.4f} | "
            f"CostRate {mean_cost_rate:.4f}"
        )

    save_path = model_checkpoint_path("unsafe", args.env)
    agent.save(save_path)
    wandb.save(save_path)
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1024)
    p.add_argument("--updates", type=int, default=300)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent_coef", type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=256)

    args = p.parse_args()
    train_unsafe(args)