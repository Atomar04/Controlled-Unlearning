import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from reifule.algorithm import PPOUnlearner
from scripts.train_unsafe import make_env


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_agent(env, model_path: str) -> PPOUnlearner:
    agent = PPOUnlearner(env)
    agent.load(model_path)
    agent.policy.eval()
    return agent


@torch.no_grad()
def extract_features_and_targets(agent: PPOUnlearner, states: np.ndarray, batch_size: int = 4096):
    feats, mus = [], []
    n = len(states)
    for i in range(0, n, batch_size):
        s = torch.as_tensor(states[i:i + batch_size], dtype=torch.float32, device=agent.device)
        feat = agent.policy.shared(s)
        mean, _, _, _ = agent.policy(s)
        feats.append(feat.cpu().numpy())
        mus.append(mean.cpu().numpy())
    return np.concatenate(feats, axis=0), np.concatenate(mus, axis=0)


def eval_regressor(reg, X, Y):
    pred = reg.predict(X)
    mse = float(mean_squared_error(Y, pred))
    r2 = float(r2_score(Y, pred, multioutput="uniform_average"))
    return {
        "n": int(len(X)),
        "mse": mse,
        "r2": r2,
    }


def print_metrics(name, m):
    print(f"\n[{name}]")
    print(f"n   : {m['n']}")
    print(f"mse : {m['mse']:.6f}")
    print(f"r2  : {m['r2']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--probe_path", type=str, required=True,
                        help="Path to hazard_probe_*.pkl so we reuse the exact same splits.")
    parser.add_argument("--metrics_out", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.probe_path, "rb") as f:
        payload = pickle.load(f)

    data = np.load(args.dataset, allow_pickle=True)
    states = data["states"]
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in payload["splits"].items()}

    env = make_env(args.env, n_envs=1)
    agent = load_agent(env, args.model_path)

    X, Y = extract_features_and_targets(agent, states, batch_size=args.batch_size)

    reg = MultiOutputRegressor(Ridge(alpha=args.alpha))
    reg.fit(X[splits["train"]], Y[splits["train"]])

    train_m = eval_regressor(reg, X[splits["train"]], Y[splits["train"]])
    val_m = eval_regressor(reg, X[splits["val"]], Y[splits["val"]])
    test_m = eval_regressor(reg, X[splits["test"]], Y[splits["test"]])

    print(f"\nmodel: {args.model_path}")
    print_metrics("train", train_m)
    print_metrics("val", val_m)
    print_metrics("test", test_m)

    out = {
        "model_path": args.model_path,
        "train": train_m,
        "val": val_m,
        "test": test_m,
    }

    ensure_parent(args.metrics_out)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
