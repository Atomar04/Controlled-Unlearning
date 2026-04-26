import os
import random
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym
import safety_gymnasium


# ---------------------------------------------------------
# Environment setup
# ---------------------------------------------------------

# Keep the same headless behavior you already use in training scripts.
os.environ["MUJOCO_GL"] = "egl"
os.environ["SDL_VIDEODRIVER"] = "dummy"


def make_env(env_id: str, n_envs: int = 1):
    """
    Same logic as your current scripts/train_unsafe.py.
    """
    if "Safety" in env_id:
        if n_envs > 1:
            return safety_gymnasium.vector.make(env_id, num_envs=n_envs)
        return safety_gymnasium.make(env_id)

    if n_envs > 1:
        return gym.vector.SyncVectorEnv([lambda: gym.make(env_id) for _ in range(n_envs)])
    return gym.make(env_id)


# ---------------------------------------------------------
# Common helpers
# ---------------------------------------------------------

def set_seed(seed: int):
    """
    Same logic as your current train_unlearn.py helper.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def extract_cforget(cost, info):
    """
    Binary forget concept: hazard contact.
    Uses cost_hazards if present, otherwise falls back to cost > 0.
    Supports single-env and vector-env outputs.

    Same logic as your current train_unlearn.py helper.
    """
    if isinstance(info, dict) and "cost_hazards" in info:
        ch = np.asarray(info["cost_hazards"], dtype=np.float32)
        return (ch > 0).astype(np.float32)
    return (np.asarray(cost, dtype=np.float32) > 0).astype(np.float32)


# ---------------------------------------------------------
# Artifact / checkpoint paths
# ---------------------------------------------------------

ARTIFACTS_ROOT = Path("artifacts")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_dir(*parts):
    """
    Returns a directory path inside artifacts/, ensuring it exists.
    Example:
        artifact_dir("models", "unsafe")
    """
    return ensure_dir(ARTIFACTS_ROOT.joinpath(*parts))


def artifact_file(*parts):
    """
    Returns a file path inside artifacts/, ensuring parent dir exists.
    Example:
        artifact_file("probes", "hazard_direction.pt")
    """
    path = ARTIFACTS_ROOT.joinpath(*parts)
    ensure_dir(path.parent)
    return str(path)


def model_checkpoint_path(method: str, env: str, update: int | None = None):
    """
    Standardized checkpoint path builder.

    method ∈ {
        "unsafe",
        "oracle",
        "concept",
        "trajectory_decremental",
        "repedit",
    }
    """
    base = artifact_dir("models", method)

    if method == "unsafe":
        filename = f"unsafe_expert_{env}.pt"

    elif method == "oracle":
        if update is None:
            raise ValueError("oracle checkpoint requires update")
        filename = f"oracle_{env}_{update}.pt"

    elif method == "concept":
        if update is None:
            raise ValueError("concept checkpoint requires update")
        filename = f"safe_concept_{env}_{update}.pt"

    elif method == "trajectory_decremental":
        if update is None:
            raise ValueError("trajectory_decremental checkpoint requires update")
        filename = f"safe_trajectory_decremental_{env}_{update}.pt"

    elif method == "repedit":
        if update is None:
            raise ValueError("repedit checkpoint requires update")
        filename = f"safe_repedit_{env}_{update}.pt"

    else:
        raise ValueError(f"Unknown method: {method}")

    return str(base / filename)


def probe_artifact_path(filename: str):
    return artifact_file("probes", filename)


def eval_artifact_path(filename: str):
    return artifact_file("eval", filename)


def log_artifact_path(filename: str):
    return artifact_file("logs", filename)


# ---------------------------------------------------------
# Rep-edit checkpoint helper
# ---------------------------------------------------------

def save_repedit_checkpoint(path: str, agent, editor, extra_meta: dict | None = None):
    """
    Same logic as your current train_unlearn.py helper:
    saves normal policy weights plus editor config.
    """
    payload = {
        "state_dict": agent.policy.state_dict(),
        "editor": {
            "direction": editor.direction.detach().cpu(),
            "alpha": float(editor.alpha.detach().cpu().item()),
            "tau": float(editor.tau.detach().cpu().item()),
            "beta": float(editor.beta),
        },
    }
    if extra_meta is not None:
        payload["meta"] = extra_meta
    torch.save(payload, path)