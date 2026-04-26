import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.utils import make_env, probe_artifact_path


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def to_obs_np(obs):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.ndim == 2 and obs.shape[0] == 1:
        return obs[0]
    return obs


def done_flag(x) -> bool:
    arr = np.asarray(x)
    return bool(arr.any())


def infer_hazard_label(cost, info) -> int:
    if isinstance(info, dict) and "cost_hazards" in info:
        try:
            return int(float(np.asarray(info["cost_hazards"]).mean()) > 0.0)
        except Exception:
            pass
    return int(float(np.asarray(cost, dtype=np.float32).mean()) > 0.0)


def load_agent(env, model_path: str) -> PPOUnlearner:
    """
    Loads:
      - normal checkpoints with only state_dict
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
def extract_features(
    agent: PPOUnlearner,
    states: np.ndarray,
    batch_size: int = 4096,
    use_editor_features: bool = True,
) -> np.ndarray:
    """
    Extract latent features.

    Default behavior:
      - if agent has an attached editor and use_editor_features=True,
        use the edited latent representation
      - otherwise use the raw shared features

    This preserves the old behavior for normal checkpoints, while making
    repedit checkpoints actually probe the edited representation.
    """
    feats = []

    has_editor = getattr(agent, "editor", None) is not None
    use_editor = bool(use_editor_features and has_editor)

    for i in range(0, len(states), batch_size):
        s = torch.as_tensor(
            states[i:i + batch_size],
            dtype=torch.float32,
            device=agent.device,
        )

        if use_editor:
            _, _, _, _, _, z = agent.policy(s, editor=agent.editor, return_feat=True)
        else:
            z = agent.policy.encode(s)

        feats.append(z.cpu().numpy())

    return np.concatenate(feats, axis=0)


def collect_dataset(
    env,
    agent: PPOUnlearner,
    n_per_class: int = 10000,
    max_env_steps: int = 400000,
    deterministic: bool = True,
    seed: int = 42,
):
    states, labels, episode_ids = [], [], []

    counts = {0: 0, 1: 0}
    env_steps = 0
    episode_id = 0

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    obs = to_obs_np(obs)

    while env_steps < max_env_steps and (counts[0] < n_per_class or counts[1] < n_per_class):
        act, _, _, _ = agent.act(obs, deterministic=deterministic)
        act = act[0] if np.asarray(act).ndim == 2 else act

        next_obs, rew, cost, term, trunc, info = env.step(act)
        y = infer_hazard_label(cost, info)

        if counts[y] < n_per_class:
            states.append(obs.copy())
            labels.append(y)
            episode_ids.append(episode_id)
            counts[y] += 1

        env_steps += 1

        if done_flag(term) or done_flag(trunc):
            episode_id += 1
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            obs = to_obs_np(obs)
        else:
            obs = to_obs_np(next_obs)

    states = np.asarray(states, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    episode_ids = np.asarray(episode_ids, dtype=np.int64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(labels))
    return {
        "states": states[perm],
        "labels": labels[perm],
        "episode_ids": episode_ids[perm],
    }


def save_dataset(dataset, out_path: str, meta: dict):
    ensure_parent(out_path)
    np.savez_compressed(
        out_path,
        states=dataset["states"],
        labels=dataset["labels"],
        episode_ids=dataset["episode_ids"],
        meta_json=np.asarray(json.dumps(meta)),
    )


def load_dataset(path: str):
    data = np.load(path, allow_pickle=True)
    dataset = {
        "states": data["states"],
        "labels": data["labels"],
        "episode_ids": data["episode_ids"],
    }
    meta = json.loads(str(data["meta_json"]))
    return dataset, meta


def split_score(labels: np.ndarray, idx: np.ndarray, global_ratio: float):
    if len(idx) == 0:
        return np.inf

    y = labels[idx]
    vals, cnts = np.unique(y, return_counts=True)

    if len(vals) < 2:
        return np.inf
    if cnts.min() < 25:
        return np.inf

    return abs(float(y.mean()) - global_ratio)


def make_splits(
    labels: np.ndarray,
    episode_ids: np.ndarray,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
    tries: int = 256,
):
    idx_all = np.arange(len(labels))
    global_ratio = float(labels.mean())

    best = None
    best_score = np.inf

    unique_eps = np.unique(episode_ids)
    use_group_split = len(unique_eps) >= 10

    if use_group_split:
        for t in range(tries):
            gss1 = GroupShuffleSplit(
                n_splits=1,
                test_size=(val_frac + test_frac),
                random_state=seed + t,
            )
            train_idx, temp_idx = next(gss1.split(idx_all, labels, groups=episode_ids))

            temp_local = np.arange(len(temp_idx))
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=test_frac / (val_frac + test_frac),
                random_state=seed + 1000 + t,
            )
            val_local, test_local = next(
                gss2.split(temp_local, labels[temp_idx], groups=episode_ids[temp_idx])
            )

            val_idx = temp_idx[val_local]
            test_idx = temp_idx[test_local]

            candidate = {"train": train_idx, "val": val_idx, "test": test_idx}
            score = (
                split_score(labels, train_idx, global_ratio)
                + split_score(labels, val_idx, global_ratio)
                + split_score(labels, test_idx, global_ratio)
            )

            if np.isfinite(score) and score < best_score:
                best = candidate
                best_score = score

    if best is None:
        train_idx, temp_idx = train_test_split(
            idx_all,
            test_size=(val_frac + test_frac),
            random_state=seed,
            stratify=labels,
        )
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=test_frac / (val_frac + test_frac),
            random_state=seed + 1,
            stratify=labels[temp_idx],
        )
        best = {"train": train_idx, "val": val_idx, "test": test_idx}

    for split_name, idx in best.items():
        y = labels[idx]
        vals, cnts = np.unique(y, return_counts=True)
        if len(vals) < 2:
            raise RuntimeError(f"{split_name} split has only one class.")
        if cnts.min() < 25:
            raise RuntimeError(
                f"{split_name} split has too few examples of one class: {dict(zip(vals, cnts))}"
            )

    return best


def build_probe(seed: int = 42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_probe(clf, features: np.ndarray, labels: np.ndarray):
    pred = clf.predict(features)

    out = {
        "balanced_accuracy": float(balanced_accuracy_score(labels, pred)),
        "confusion_matrix": confusion_matrix(labels, pred).tolist(),
        "n": int(len(labels)),
        "hazard_ratio": float(labels.mean()),
        "safe_count": int((labels == 0).sum()),
        "hazard_count": int((labels == 1).sum()),
    }

    if len(np.unique(labels)) == 2:
        prob = clf.predict_proba(features)[:, 1]
        out["auroc"] = float(roc_auc_score(labels, prob))
    else:
        out["auroc"] = None

    return out


def print_metrics(name: str, m: dict):
    print(f"\n[{name}]")
    print(f"n                : {m['n']}")
    print(f"safe_count       : {m['safe_count']}")
    print(f"hazard_count     : {m['hazard_count']}")
    print(f"hazard_ratio     : {m['hazard_ratio']:.4f}")
    print(f"balanced_acc     : {m['balanced_accuracy']:.4f}")
    print(f"auroc            : {m['auroc']}")
    print("confusion_matrix :")
    print(np.array(m["confusion_matrix"]))


def cmd_collect(args):
    if args.out is None:
        args.out = probe_artifact_path(f"probe_states_fixed_{args.env}.npz")

    env = make_env(args.env, n_envs=1)
    agent = load_agent(env, args.model_path)

    dataset = collect_dataset(
        env=env,
        agent=agent,
        n_per_class=args.n_per_class,
        max_env_steps=args.max_env_steps,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    meta = {
        "env": args.env,
        "collector_model_path": args.model_path,
        "n_per_class": args.n_per_class,
        "max_env_steps": args.max_env_steps,
        "deterministic": not args.stochastic,
        "seed": args.seed,
    }
    save_dataset(dataset, args.out, meta)

    print(f"saved: {args.out}")
    print(f"size : {len(dataset['labels'])}")
    print(f"safe : {(dataset['labels'] == 0).sum()}")
    print(f"haz  : {(dataset['labels'] == 1).sum()}")
    print(f"eps  : {len(np.unique(dataset['episode_ids']))}")

    env.close()

def cmd_fit(args):
    if args.dataset is None:
        args.dataset = probe_artifact_path(f"probe_states_fixed_{args.env}.npz")

    dataset, meta = load_dataset(args.dataset)
    states = dataset["states"]
    labels = dataset["labels"]
    episode_ids = dataset["episode_ids"]

    splits = make_splits(
        labels=labels,
        episode_ids=episode_ids,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        tries=args.split_tries,
    )

    env = make_env(args.env, n_envs=1)
    agent = load_agent(env, args.model_path)
    feats = extract_features(
        agent,
        states,
        batch_size=args.batch_size,
        use_editor_features=args.use_editor_features,
    )

    clf = build_probe(seed=args.seed)
    clf.fit(feats[splits["train"]], labels[splits["train"]])

    train_m = evaluate_probe(clf, feats[splits["train"]], labels[splits["train"]])
    val_m = evaluate_probe(clf, feats[splits["val"]], labels[splits["val"]])
    test_m = evaluate_probe(clf, feats[splits["test"]], labels[splits["test"]])

    print_metrics("train", train_m)
    print_metrics("val", val_m)
    print_metrics("test", test_m)

    payload = {
        "probe": clf,
        "dataset_path": str(Path(args.dataset).resolve()),
        "dataset_meta": meta,
        "reference_model_path": args.model_path,
        "env": args.env,
        "seed": args.seed,
        "use_editor_features": bool(args.use_editor_features),
        "splits": {k: v.tolist() for k, v in splits.items()},
        "metrics": {"train": train_m, "val": val_m, "test": test_m},
    }

    ensure_parent(args.out)
    with open(args.out, "wb") as f:
        pickle.dump(payload, f)

    if args.metrics_out:
        ensure_parent(args.metrics_out)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(payload["metrics"], f, indent=2)

    env.close()

def cmd_eval(args):
    with open(args.probe_path, "rb") as f:
        payload = pickle.load(f)

    dataset_path = args.dataset or payload["dataset_path"]
    env_name = args.env or payload["env"]

    dataset, _ = load_dataset(dataset_path)
    states = dataset["states"]
    labels = dataset["labels"]
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in payload["splits"].items()}

    env = make_env(env_name, n_envs=1)
    agent = load_agent(env, args.model_path)

    use_editor_features = (
        args.use_editor_features
        if args.use_editor_features is not None
        else bool(payload.get("use_editor_features", True))
    )

    feats = extract_features(
        agent,
        states,
        batch_size=args.batch_size,
        use_editor_features=use_editor_features,
    )

    clf = payload["probe"]

    train_m = evaluate_probe(clf, feats[splits["train"]], labels[splits["train"]])
    val_m = evaluate_probe(clf, feats[splits["val"]], labels[splits["val"]])
    test_m = evaluate_probe(clf, feats[splits["test"]], labels[splits["test"]])

    print(f"\nreference model: {payload['reference_model_path']}")
    print(f"eval model     : {args.model_path}")
    print(f"use_editor_features: {use_editor_features}")
    print_metrics("train", train_m)
    print_metrics("val", val_m)
    print_metrics("test", test_m)

    metrics = {
        "reference_model_path": payload["reference_model_path"],
        "eval_model_path": args.model_path,
        "use_editor_features": bool(use_editor_features),
        "train": train_m,
        "val": val_m,
        "test": test_m,
    }

    if args.metrics_out:
        ensure_parent(args.metrics_out)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    env.close()


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    c.add_argument("--model_path", type=str, required=True)
    c.add_argument("--out", type=str, default=None)
    c.add_argument("--n_per_class", type=int, default=10000)
    c.add_argument("--max_env_steps", type=int, default=400000)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--stochastic", action="store_true")
    c.set_defaults(func=cmd_collect)

    f = sub.add_parser("fit")
    f.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    f.add_argument("--dataset", type=str, default=None)
    f.add_argument("--model_path", type=str, required=True)
    f.add_argument("--out", type=str, default=probe_artifact_path("hazard_probe.pkl"))
    f.add_argument(
        "--metrics_out",
        type=str,
        default=probe_artifact_path("hazard_probe_metrics.json"),
    )
    f.add_argument("--batch_size", type=int, default=4096)
    f.add_argument("--val_frac", type=float, default=0.15)
    f.add_argument("--test_frac", type=float, default=0.15)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--split_tries", type=int, default=256)
    f.add_argument("--use_editor_features", action="store_true")
    f.set_defaults(func=cmd_fit)

    e = sub.add_parser("eval")
    e.add_argument("--probe_path", type=str, required=True)
    e.add_argument("--model_path", type=str, required=True)
    e.add_argument("--dataset", type=str, default=None)
    e.add_argument("--env", type=str, default=None)
    e.add_argument("--metrics_out", type=str, default=None)
    e.add_argument("--batch_size", type=int, default=4096)

    # omitted -> inherit from probe payload
    # passed   -> force True
    e.add_argument("--use_editor_features", action="store_true", default=None)

    e.set_defaults(func=cmd_eval)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)