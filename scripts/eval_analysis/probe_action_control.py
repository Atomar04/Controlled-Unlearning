"""
Action-control probe.

This script measures how well the policy's hidden representation linearly predicts
the policy's action mean.

Purpose:
    This is an optional diagnostic, not part of the core training loop.

Interpretation:
    High R^2  -> hidden features preserve action-control information.
    Low R^2   -> hidden features are less linearly predictive of policy actions.

Example:
    python -m scripts.eval_analysis.probe_action_control \
        --env SafetyPointGoal1-v0 \
        --dataset artifacts/probes/probe_states_fixed_SafetyPointGoal1-v0.npz \
        --model_path artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_150.pt \
        --probe_path artifacts/probes/hazard_probe.pkl \
        --metrics_out artifacts/eval/action_control/concept_150.json
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from reifule.algorithm import PPOUnlearner
from reifule.utils import make_env


# =========================================================
# File helpers
# =========================================================

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


# =========================================================
# Agent loading
# =========================================================

def load_agent(env, model_path: str) -> PPOUnlearner:
    """
    Load a policy checkpoint for action-control probing.

    Supports standard checkpoints saved as:
        {"state_dict": ...}

    Also supports older/simple checkpoints if PPOUnlearner.load() handles them.
    """
    agent = PPOUnlearner(env)

    ckpt = torch.load(model_path, map_location=agent.device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        agent.policy.load_state_dict(ckpt["state_dict"])
    else:
        # Fallback for older checkpoints if agent.load supports the format.
        agent.load(model_path)

    agent.policy.eval()
    return agent


# =========================================================
# Feature extraction
# =========================================================

@torch.no_grad()
def extract_features_and_targets(
    agent: PPOUnlearner,
    states: np.ndarray,
    batch_size: int = 4096,
):
    """
    Extract:
        X = hidden features from policy.shared(s)
        Y = policy action mean from policy(s)

    Then the probe learns:
        hidden features -> action mean
    """
    feats = []
    mus = []

    n = len(states)

    for i in range(0, n, batch_size):
        s_np = states[i:i + batch_size]
        s = torch.as_tensor(
            s_np,
            dtype=torch.float32,
            device=agent.device,
        )

        feat = agent.policy.shared(s)
        mean, _, _, _ = agent.policy(s)

        feats.append(feat.detach().cpu().numpy())
        mus.append(mean.detach().cpu().numpy())

    X = np.concatenate(feats, axis=0)
    Y = np.concatenate(mus, axis=0)

    return X, Y


# =========================================================
# Metrics
# =========================================================

def eval_regressor(reg, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
    pred = reg.predict(X)

    mse = float(mean_squared_error(Y, pred))
    r2 = float(r2_score(Y, pred, multioutput="uniform_average"))

    return {
        "n": int(len(X)),
        "mse": mse,
        "r2": r2,
    }


def print_metrics(name: str, metrics: Dict[str, Any]) -> None:
    print(f"\n[{name}]")
    print(f"n   : {metrics['n']}")
    print(f"mse : {metrics['mse']:.6f}")
    print(f"r2  : {metrics['r2']:.6f}")


# =========================================================
# Main
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env",
        type=str,
        default="SafetyPointGoal1-v0",
        help="Environment id.",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=(
            "Path to .npz dataset containing states. "
            "Example: artifacts/probes/probe_states_fixed_SafetyPointGoal1-v0.npz"
        ),
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint to probe.",
    )

    parser.add_argument(
        "--probe_path",
        type=str,
        required=True,
        help=(
            "Path to hazard probe .pkl file. "
            "Used only to reuse the exact same train/val/test splits."
        ),
    )

    parser.add_argument(
        "--metrics_out",
        type=str,
        required=True,
        help="Where to save action-control probe metrics as JSON.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size for feature extraction.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularization strength.",
    )

    args = parser.parse_args()

    # -----------------------------------------------------
    # Load split metadata from existing hazard probe
    # -----------------------------------------------------
    with open(args.probe_path, "rb") as f:
        payload = pickle.load(f)

    if "splits" not in payload:
        raise KeyError(
            f"Expected key 'splits' in probe file: {args.probe_path}"
        )

    splits = {
        k: np.asarray(v, dtype=np.int64)
        for k, v in payload["splits"].items()
    }

    required_splits = {"train", "val", "test"}
    missing = required_splits - set(splits.keys())
    if missing:
        raise KeyError(
            f"Missing split(s) in probe file {args.probe_path}: {sorted(missing)}"
        )

    # -----------------------------------------------------
    # Load states
    # -----------------------------------------------------
    data = np.load(args.dataset, allow_pickle=True)

    if "states" not in data:
        raise KeyError(
            f"Expected key 'states' in dataset file: {args.dataset}"
        )

    states = np.asarray(data["states"], dtype=np.float32)

    n_states = len(states)
    max_split_idx = max(
        int(splits["train"].max()),
        int(splits["val"].max()),
        int(splits["test"].max()),
    )

    if max_split_idx >= n_states:
        raise ValueError(
            f"Split index {max_split_idx} is out of bounds for dataset "
            f"with {n_states} states."
        )

    # -----------------------------------------------------
    # Load agent and extract representation/action targets
    # -----------------------------------------------------
    env = make_env(args.env, n_envs=1)

    try:
        agent = load_agent(env, args.model_path)

        X, Y = extract_features_and_targets(
            agent,
            states,
            batch_size=args.batch_size,
        )

    finally:
        env.close()

    # -----------------------------------------------------
    # Train linear action-control probe
    # -----------------------------------------------------
    reg = MultiOutputRegressor(Ridge(alpha=args.alpha))

    reg.fit(
        X[splits["train"]],
        Y[splits["train"]],
    )

    train_metrics = eval_regressor(
        reg,
        X[splits["train"]],
        Y[splits["train"]],
    )

    val_metrics = eval_regressor(
        reg,
        X[splits["val"]],
        Y[splits["val"]],
    )

    test_metrics = eval_regressor(
        reg,
        X[splits["test"]],
        Y[splits["test"]],
    )

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------
    print(f"\nmodel      : {args.model_path}")
    print(f"dataset    : {args.dataset}")
    print(f"probe file : {args.probe_path}")
    print(f"features   : {X.shape}")
    print(f"targets    : {Y.shape}")
    print(f"ridge alpha: {args.alpha}")

    print_metrics("train", train_metrics)
    print_metrics("val", val_metrics)
    print_metrics("test", test_metrics)

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------
    out = {
        "env": args.env,
        "model_path": args.model_path,
        "dataset": args.dataset,
        "probe_path": args.probe_path,
        "metrics_out": args.metrics_out,
        "ridge_alpha": float(args.alpha),
        "num_states": int(n_states),
        "feature_dim": int(X.shape[1]),
        "action_dim": int(Y.shape[1]),
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "interpretation": {
            "task": "Predict policy action mean from hidden representation using a linear Ridge probe.",
            "high_r2": "Hidden representation linearly preserves action-control information.",
            "low_r2": "Action-control information is less linearly accessible from hidden representation.",
        },
    }

    ensure_parent(args.metrics_out)

    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved metrics to: {args.metrics_out}")


if __name__ == "__main__":
    main()