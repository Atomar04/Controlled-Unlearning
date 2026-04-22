import numpy as np
import torch
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from reifule.algorithm import PPOUnlearner
from scripts.train_unsafe import make_env


# =========================
# Extract features
# =========================
def extract_features(agent, states):
    states_t = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        feat = agent.policy.shared(states_t)
    return feat.cpu().numpy()


# =========================
# Collect dataset
# =========================
def collect_dataset(env, agent, n_steps=50000, target_ratio=0.5):
    states_hazard = []
    states_safe = []

    obs, _ = env.reset()

    while len(states_hazard) < n_steps // 2 or len(states_safe) < n_steps // 2:

        act, _, _, _ = agent.act(obs, deterministic=True)
        act = act[0] if act.ndim == 2 else act

        next_obs, rew, cost, term, trunc, info = env.step(act)

        is_hazard = float(np.mean(cost)) > 0

        if is_hazard and len(states_hazard) < n_steps // 2:
            states_hazard.append(obs)
        elif not is_hazard and len(states_safe) < n_steps // 2:
            states_safe.append(obs)

        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    states = np.array(states_hazard + states_safe)
    labels = np.array([1]*len(states_hazard) + [0]*len(states_safe))

    return states, labels


# =========================
# Train probe
# =========================
def train_probe(features, labels):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features, labels)
    preds = clf.predict(features)
    acc = balanced_accuracy_score(labels, preds)
    print(confusion_matrix(labels, preds))
    return clf, acc


# =========================
# Main
# =========================
def main(args):
    env = make_env(args.env, n_envs=1)

    print("\n===== LOADING MODEL =====")
    agent = PPOUnlearner(env)
    agent.load(args.model_path)

    print("\n===== COLLECTING DATA =====")
    states, labels = collect_dataset(env, agent, n_steps=args.steps)

    print(f"Dataset size: {len(states)}")
    print(f"Hazard ratio: {labels.mean():.4f}")

    print("\n===== EXTRACTING FEATURES =====")
    features = extract_features(agent, states)

    print("\n===== TRAINING PROBE =====")
    _, acc = train_probe(features, labels)

    print("\n===== RESULT =====")
    print(f"Probe accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="SafetyPointGoal1-v0")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=20000)

    args = parser.parse_args()
    main(args)
