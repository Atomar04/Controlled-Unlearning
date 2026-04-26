import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.utils import make_env, artifact_dir, probe_artifact_path


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def load_agent(env, model_path: str) -> PPOUnlearner:
    """
    Loads:
      - standard checkpoints with only state_dict
      - repedit checkpoints with editor metadata
    """
    agent = PPOUnlearner(env)
    ckpt = torch.load(model_path, map_location=agent.device)

    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError(f"Invalid checkpoint format: {model_path}")

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

    return agent


@torch.no_grad()
def extract_latent_and_mu(
    agent: PPOUnlearner,
    states: np.ndarray,
    batch_size: int = 4096,
    use_editor_features: bool = True,
):
    """
    Extract latent representation and action mean.

    Default behavior:
      - if agent has an attached editor and use_editor_features=True,
        analyze the operative edited latent + edited action mean
      - otherwise analyze the raw backbone latent + raw action mean

    This preserves the old behavior for normal checkpoints while making
    repedit checkpoints analysis meaningful.
    """
    zs, mus = [], []

    has_editor = getattr(agent, "editor", None) is not None
    use_editor = bool(use_editor_features and has_editor)

    for i in range(0, len(states), batch_size):
        s = torch.as_tensor(
            states[i:i + batch_size],
            dtype=torch.float32,
            device=agent.device,
        )

        if use_editor:
            mu, _, _, _, _, z = agent.policy(
                s,
                editor=agent.editor,
                return_feat=True,
            )
        else:
            z = agent.policy.encode(s)
            mu, _, _, _ = agent.policy(s, editor=None)

        zs.append(z.cpu().numpy())
        mus.append(mu.cpu().numpy())

    return np.concatenate(zs, axis=0), np.concatenate(mus, axis=0)


def cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8):
    a_n = np.linalg.norm(a, axis=1, keepdims=True)
    b_n = np.linalg.norm(b, axis=1, keepdims=True)
    denom = np.maximum(a_n * b_n, eps)
    return np.sum(a * b, axis=1, keepdims=False) / denom.squeeze(1)


def summarize_group(
    name: str,
    idx: np.ndarray,
    z_ref: np.ndarray,
    z_cmp: np.ndarray,
    mu_ref: np.ndarray,
    mu_cmp: np.ndarray,
):
    zr = z_ref[idx]
    zc = z_cmp[idx]
    mr = mu_ref[idx]
    mc = mu_cmp[idx]

    dz = zc - zr
    dmu = mc - mr

    z_shift_l2 = np.linalg.norm(dz, axis=1)
    z_cos = cosine_similarity_rows(zr, zc)
    mu_shift_l2 = np.linalg.norm(dmu, axis=1)

    out = {
        "n": int(len(idx)),
        "latent_shift_l2_mean": float(np.mean(z_shift_l2)),
        "latent_shift_l2_std": float(np.std(z_shift_l2)),
        "latent_shift_l2_median": float(np.median(z_shift_l2)),
        "latent_cosine_mean": float(np.mean(z_cos)),
        "latent_cosine_std": float(np.std(z_cos)),
        "action_mean_shift_l2_mean": float(np.mean(mu_shift_l2)),
        "action_mean_shift_l2_std": float(np.std(mu_shift_l2)),
        "action_mean_shift_l2_median": float(np.median(mu_shift_l2)),
        "latent_norm_ref_mean": float(np.mean(np.linalg.norm(zr, axis=1))),
        "latent_norm_cmp_mean": float(np.mean(np.linalg.norm(zc, axis=1))),
    }

    print(f"\n[{name}]")
    for k, v in out.items():
        print(f"{k}: {v}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument(
        "--dataset",
        type=str,
        default=probe_artifact_path("probe_states_fixed_v2.npz"),
    )
    parser.add_argument("--ref_model_path", type=str, required=True)
    parser.add_argument("--cmp_model_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=4096)

    parser.add_argument(
        "--raw_features",
        action="store_true",
        help="Force raw backbone latent comparison even if a repedit editor is attached.",
    )

    parser.add_argument("--eval_id", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_id = args.eval_id or f"{args.env}__latent_shift__{timestamp}"

    if args.out is None:
        out_dir = artifact_dir("eval", "latent_shift_eval", eval_id)
        out_path = out_dir / "latent_shift_metrics.json"
    else:
        out_path = Path(args.out)
        ensure_parent(out_path)

    use_editor_features = not args.raw_features

    data = np.load(args.dataset, allow_pickle=True)
    states = data["states"]
    labels = data["labels"]

    env = make_env(args.env, n_envs=1)
    ref_agent = load_agent(env, args.ref_model_path)
    cmp_agent = load_agent(env, args.cmp_model_path)

    print("\n===== EXTRACTING REFERENCE LATENTS =====")
    z_ref, mu_ref = extract_latent_and_mu(
        ref_agent,
        states,
        batch_size=args.batch_size,
        use_editor_features=use_editor_features,
    )

    print("\n===== EXTRACTING COMPARISON LATENTS =====")
    z_cmp, mu_cmp = extract_latent_and_mu(
        cmp_agent,
        states,
        batch_size=args.batch_size,
        use_editor_features=use_editor_features,
    )

    idx_all = np.arange(len(labels))
    idx_safe = np.where(labels == 0)[0]
    idx_haz = np.where(labels == 1)[0]

    metrics = {
        "eval_id": eval_id,
        "env": args.env,
        "dataset": args.dataset,
        "ref_model_path": args.ref_model_path,
        "cmp_model_path": args.cmp_model_path,
        "use_editor_features": bool(use_editor_features),
        "overall": summarize_group("overall", idx_all, z_ref, z_cmp, mu_ref, mu_cmp),
        "safe": summarize_group("safe", idx_safe, z_ref, z_cmp, mu_ref, mu_cmp),
        "hazard": summarize_group("hazard", idx_haz, z_ref, z_cmp, mu_ref, mu_cmp),
    }

    metrics["contrasts"] = {
        "hazard_minus_safe_latent_shift_l2_mean":
            metrics["hazard"]["latent_shift_l2_mean"] - metrics["safe"]["latent_shift_l2_mean"],
        "hazard_minus_safe_action_shift_l2_mean":
            metrics["hazard"]["action_mean_shift_l2_mean"] - metrics["safe"]["action_mean_shift_l2_mean"],
        "safe_overall_latent_shift_ratio":
            metrics["safe"]["latent_shift_l2_mean"] / max(metrics["overall"]["latent_shift_l2_mean"], 1e-8),
        "hazard_overall_latent_shift_ratio":
            metrics["hazard"]["latent_shift_l2_mean"] / max(metrics["overall"]["latent_shift_l2_mean"], 1e-8),
    }

    print("\n[contrasts]")
    for k, v in metrics["contrasts"].items():
        print(f"{k}: {v}")

    ensure_parent(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved latent-shift metrics to: {out_path}")

    env.close()


if __name__ == "__main__":
    main()