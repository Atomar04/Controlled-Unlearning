"""
train_repedit.py — 3-phase offline representation editing for policy unlearning.

Phase 1  Collect a fixed replay buffer from the FROZEN unsafe expert.
         Label every transition as near-hazard or far-safe using a lookahead
         window so the retain loss has real signal.

Phase 2  Train the RepresentationEditor (alpha, tau) on the FIXED buffer.
         The policy backbone is frozen throughout — no PPO, no env interaction.
         Three sub-steps:
           2a. Tau sweep  — find the quantile that maximises locality ratio
           2b. Alpha solve — analytical closed-form initialisation
           2c. Gradient refinement on fixed buffer (forget + retain + locality)

Phase 3  Evaluate the frozen edited policy online.
         No backward pass ever. Pure rollout + logging + checkpoint.
"""

import argparse
from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np
import torch
import wandb

from reifule.algorithm import PPOUnlearner, RepresentationEditor
from reifule.utils import (
    extract_cforget,
    make_env,
    model_checkpoint_path,
    probe_artifact_path,
    save_repedit_checkpoint,
    set_seed,
)


# =============================================================================
# Data structures
# =============================================================================

class OfflineBuffer(NamedTuple):
    """Fixed buffer collected from the unsafe expert (Phase 1 output)."""
    states:      np.ndarray   # [N, obs_dim]  float32
    c_forget:    np.ndarray   # [N]           float32  (binary hazard label)
    near_hazard: np.ndarray   # [N]           bool     (within lookahead of hazard)
    far_safe:    np.ndarray   # [N]           bool     (no hazard in ±window)
    n_steps:     int
    n_hazard:    int
    n_near:      int
    n_far:       int


# =============================================================================
# Artifact I/O
# =============================================================================

def load_repedit_artifact(path: str) -> Tuple[np.ndarray, dict]:
    """
    Load a hazard concept direction from .pt/.pth or .npz.
    Accepted keys: direction, u, vector, probe_weight, w
    Optional metadata: tau, alpha, beta
    """
    DIRECTION_KEYS = ["direction", "u", "vector", "probe_weight", "w"]

    if path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        direction = None
        for k in DIRECTION_KEYS:
            if k in data:
                direction = np.asarray(data[k], dtype=np.float32)
                break
        if direction is None:
            raise KeyError(f"No direction key in {path}. Expected one of {DIRECTION_KEYS}.")
        meta = {k: float(np.asarray(data[k]).reshape(-1)[0])
                for k in ("tau", "alpha", "beta") if k in data}
        return direction, meta

    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported repedit artifact format in {path}")

    direction = None
    for k in DIRECTION_KEYS:
        if k in obj:
            val = obj[k]
            direction = (
                val.detach().cpu().float().numpy()
                if isinstance(val, torch.Tensor)
                else np.asarray(val, dtype=np.float32)
            )
            break
    if direction is None:
        raise KeyError(f"No direction key in {path}. Expected one of {DIRECTION_KEYS}.")

    meta = {
        k: float(obj[k].detach().cpu().item() if isinstance(obj[k], torch.Tensor) else obj[k])
        for k in ("tau", "alpha", "beta") if k in obj
    }
    return direction, meta


# =============================================================================
# Phase 1 — Collect fixed offline buffer
# =============================================================================

def phase1_collect_buffer(
    unsafe_agent:   PPOUnlearner,
    env_id:         str,
    seed:           int,
    n_steps:        int,
    lookahead:      int = 20,
) -> OfflineBuffer:
    """
    Roll out the FROZEN unsafe expert for n_steps in a single env.
    Labels:
      c_forget[t]    — 1 if hazard contact at step t
      near_hazard[t] — 1 if any c_forget in [t, t+lookahead] is positive
      far_safe[t]    — 1 if no c_forget in [t-lookahead, t+lookahead]

    Single env, sequential → stable, reproducible state distribution.
    """
    assert unsafe_agent.editor is None, \
        "Phase 1: unsafe_agent must have no editor attached."

    env = make_env(env_id, n_envs=1)
    try:
        env.reset(seed=seed + 1000)
    except TypeError:
        pass

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    states_list   = []
    c_forget_list = []

    print(f"\n[Phase 1] Collecting {n_steps:,} steps from unsafe expert...")
    step = 0
    while step < n_steps:
        act, _, _, _ = unsafe_agent.act(obs, deterministic=False)
        act_in = act[0] if act.ndim == 2 else act

        next_obs, rew, cost, term, trunc, info = env.step(act_in)
        c_forget = extract_cforget(cost, info)
        c_scalar = float(np.asarray(c_forget).reshape(-1)[0])

        states_list.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        c_forget_list.append(c_scalar)

        obs  = next_obs
        step += 1

        done = bool(np.asarray(term).any()) or bool(np.asarray(trunc).any())
        if done:
            reset_out = env.reset()
            obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    env.close()

    states_np   = np.asarray(states_list,   dtype=np.float32)
    c_forget_np = np.asarray(c_forget_list, dtype=np.float32)
    N = len(states_np)

    # near_hazard[t]: any hazard in [t, t + lookahead]
    near_hazard = np.zeros(N, dtype=bool)
    for t in range(N):
        near_hazard[t] = c_forget_np[t : min(t + lookahead + 1, N)].any()

    # far_safe[t]: no hazard in [t - lookahead, t + lookahead]
    far_safe = np.zeros(N, dtype=bool)
    for t in range(N):
        lo = max(0, t - lookahead)
        hi = min(N, t + lookahead + 1)
        far_safe[t] = not c_forget_np[lo:hi].any()

    n_hazard = int(c_forget_np.sum())
    n_near   = int(near_hazard.sum())
    n_far    = int(far_safe.sum())

    print(
        f"[Phase 1] Done. N={N:,} | hazard={n_hazard:,} ({100*n_hazard/N:.1f}%) | "
        f"near={n_near:,} ({100*n_near/N:.1f}%) | far={n_far:,} ({100*n_far/N:.1f}%)"
    )
    return OfflineBuffer(
        states=states_np, c_forget=c_forget_np,
        near_hazard=near_hazard, far_safe=far_safe,
        n_steps=N, n_hazard=n_hazard, n_near=n_near, n_far=n_far,
    )


# =============================================================================
# Phase 2 helpers
# =============================================================================

@torch.no_grad()
def _compute_raw_scores(
    agent:       PPOUnlearner,
    direction_t: torch.Tensor,
    states_np:   np.ndarray,
    batch_size:  int = 4096,
) -> np.ndarray:
    """Project all buffer states onto the unit concept direction → [N] scores."""
    device = agent.device
    scores = []
    for start in range(0, len(states_np), batch_size):
        s    = torch.as_tensor(states_np[start:start+batch_size], dtype=torch.float32, device=device)
        feat = agent.policy.encode(s)
        scores.append((feat @ direction_t).cpu().numpy())
    return np.concatenate(scores)


def _gate_relu(scores: np.ndarray, tau: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    gate      = 1.0 / (1.0 + np.exp(-beta * (scores - tau)))
    relu_part = np.maximum(scores - tau, 0.0)
    return gate, relu_part


def _locality_ratio(
    scores:      np.ndarray,
    near_hazard: np.ndarray,
    far_safe:    np.ndarray,
    tau:         float,
    beta:        float,
    alpha:       float,
) -> float:
    """
    mean_suppression_near / mean_suppression_far.
    > 1 means the edit is concentrated near hazards. ✓
    """
    gate, relu_part = _gate_relu(scores, tau, beta)
    suppression     = alpha * gate * relu_part
    near_s = suppression[near_hazard].mean() if near_hazard.any() else 0.0
    far_s  = suppression[far_safe].mean()    if far_safe.any()    else 1e-9
    return float(near_s / (far_s + 1e-9))


def phase2a_sweep_tau(
    agent:        PPOUnlearner,
    buffer:       OfflineBuffer,
    direction_t:  torch.Tensor,
    beta:         float,
    quantile_min: float = 0.30,
    quantile_max: float = 0.90,
    n_candidates: int   = 13,
    alpha_probe:  float = 5.0,
) -> Tuple[float, dict]:
    """
    Sweep tau across quantiles of the hazard-state score distribution.
    Pick the quantile that maximises the locality ratio.
    alpha_probe is a fixed probe value used only during the sweep.
    """
    print(f"\n[Phase 2a] Sweeping tau over {n_candidates} candidates...")
    scores = _compute_raw_scores(agent, direction_t, buffer.states)

    if buffer.n_hazard >= 32:
        ref_scores = scores[buffer.c_forget > 0.5]
    else:
        ref_scores = scores
        print("[Phase 2a] Warning: <32 hazard-positive states, calibrating from all states.")

    quantiles  = np.linspace(quantile_min, quantile_max, n_candidates)
    best_tau   = float(np.quantile(ref_scores, 0.50))
    best_ratio = -1.0
    sweep_log  = []

    for q in quantiles:
        tau_cand = float(np.quantile(ref_scores, q))
        ratio    = _locality_ratio(scores, buffer.near_hazard, buffer.far_safe,
                                   tau_cand, beta, alpha_probe)
        sweep_log.append({"quantile": float(q), "tau": tau_cand, "locality_ratio": ratio})
        print(f"  q={q:.2f}  tau={tau_cand:.4f}  locality_ratio={ratio:.4f}")
        if ratio > best_ratio:
            best_ratio = ratio
            best_tau   = tau_cand

    print(f"[Phase 2a] Best tau={best_tau:.6f}  locality_ratio={best_ratio:.4f}")
    return best_tau, {
        "score_mean":          float(scores.mean()),
        "score_std":           float(scores.std()),
        "score_mean_hazard":   float(ref_scores.mean()),
        "score_std_hazard":    float(ref_scores.std()),
        "best_tau":            best_tau,
        "best_locality_ratio": best_ratio,
        "sweep":               sweep_log,
    }


def phase2b_analytical_alpha(
    scores:             np.ndarray,
    near_hazard:        np.ndarray,
    far_safe:           np.ndarray,
    tau:                float,
    beta:               float,
    target_percentile:  float = 10.0,
) -> float:
    """
    Closed-form alpha* given a fixed tau.

    Target: push mean post-edit hazard score down to the
    `target_percentile`-th percentile of the far-safe score distribution.

    post_edit = score - alpha * gate * relu(score - tau)
    → alpha* = (mean_score_near - target) / mean(gate * relu)[near]
    """
    gate, relu_part = _gate_relu(scores, tau, beta)

    ref_near  = scores[near_hazard] if near_hazard.any() else scores
    mean_near = float(ref_near.mean())
    mean_eff  = float((gate * relu_part)[near_hazard].mean()
                      if near_hazard.any() else (gate * relu_part).mean())

    if far_safe.any():
        target = float(np.percentile(scores[far_safe], target_percentile))
    else:
        target = float(scores.mean() - scores.std())

    if mean_eff < 1e-8:
        print("[Phase 2b] Warning: gate*relu denominator near zero → alpha=1.0")
        return 1.0

    alpha_star = (mean_near - target) / mean_eff
    clamped    = float(np.clip(alpha_star, 0.0, 50.0))
    print(f"[Phase 2b] Analytical alpha* = {alpha_star:.4f}  (clamped → {clamped:.4f})")
    return clamped


def phase2_train_editor(
    agent:     PPOUnlearner,
    buffer:    OfflineBuffer,
    direction: np.ndarray,
    args,
) -> Tuple[RepresentationEditor, dict]:
    """
    Phase 2 entry point:
      2a  tau sweep
      2b  analytical alpha
      2c  gradient refinement on the fixed offline buffer
    Returns a calibrated, trained editor ready for Phase 3.
    """
    device      = agent.device
    direction_t = torch.as_tensor(direction, dtype=torch.float32, device=device)
    direction_t = direction_t / (direction_t.norm() + 1e-8)

    # ---- 2a: tau sweep (or override) ----
    if args.repedit_tau is not None:
        tau        = float(args.repedit_tau)
        sweep_stats = {"best_tau": tau, "source": "manual"}
        print(f"[Phase 2a] Using manual tau = {tau:.6f}")
    else:
        tau, sweep_stats = phase2a_sweep_tau(
            agent        = agent,
            buffer       = buffer,
            direction_t  = direction_t,
            beta         = args.repedit_beta,
            quantile_min = args.tau_quantile_min,
            quantile_max = args.tau_quantile_max,
            n_candidates = args.tau_sweep_candidates,
        )
        sweep_stats["source"] = "sweep"

    # ---- 2b: analytical alpha ----
    scores      = _compute_raw_scores(agent, direction_t, buffer.states)
    alpha_init  = phase2b_analytical_alpha(
        scores            = scores,
        near_hazard       = buffer.near_hazard,
        far_safe          = buffer.far_safe,
        tau               = tau,
        beta              = args.repedit_beta,
        target_percentile = args.alpha_target_percentile,
    )

    # ---- 2c: gradient refinement ----
    editor = RepresentationEditor(
        direction = direction,
        alpha     = alpha_init,
        tau       = tau,
        beta      = args.repedit_beta,
    ).to(device)
    editor.train()

    opt = torch.optim.Adam([
        {"params": [editor.alpha], "lr": args.editor_lr},
        {"params": [editor.tau],   "lr": args.editor_tau_lr},
    ])

    N       = buffer.n_steps
    idx     = np.arange(N)
    BS      = args.editor_batch_size
    history = []

    print(f"\n[Phase 2c] Gradient refinement | {args.editor_steps} steps | BS={BS}")

    for step in range(1, args.editor_steps + 1):
        np.random.shuffle(idx)
        mb = idx[:BS]

        s_t    = torch.as_tensor(buffer.states[mb],      dtype=torch.float32, device=device)
        near_t = torch.as_tensor(buffer.near_hazard[mb], dtype=torch.bool,    device=device)
        far_t  = torch.as_tensor(buffer.far_safe[mb],    dtype=torch.bool,    device=device)

        # Backbone frozen — no grad through policy weights.
        with torch.no_grad():
            raw_feat = agent.policy.encode(s_t)                         # [B, H]

        # Replicate editor formula explicitly so grad flows to alpha and tau.
        score     = raw_feat @ direction_t                              # [B]
        gate      = torch.sigmoid(editor.beta * (score - editor.tau))  # [B]
        relu_part = torch.relu(score - editor.tau)                     # [B]
        post_edit = score - editor.alpha * gate * relu_part            # [B]
        supp      = editor.alpha * gate * relu_part                    # [B]

        # Forget loss: drive post-edit concept score down near hazards.
        forget_loss = post_edit[near_t].mean() if near_t.any() else post_edit.mean()

        # Retain loss: suppress gate activation on safe states (locality).
        retain_loss = gate[far_t].mean() if far_t.any() else torch.tensor(0.0, device=device)

        # Locality loss: explicitly maximise near-suppression > far-suppression.
        if near_t.any() and far_t.any():
            locality_loss = supp[far_t].mean() - supp[near_t].mean()
        else:
            locality_loss = torch.tensor(0.0, device=device)

        loss = (
            args.forget_coef   * forget_loss
            + args.retain_coef  * retain_loss
            + args.locality_coef * locality_loss
        )

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([editor.alpha, editor.tau], max_norm=2.0)
        opt.step()

        with torch.no_grad():
            editor.alpha.clamp_(min=0.0, max=args.alpha_max)

        log_every = max(1, args.editor_steps // 10)
        if step % log_every == 0 or step == 1:
            a_v = float(editor.alpha.item())
            t_v = float(editor.tau.item())
            loc = _locality_ratio(scores, buffer.near_hazard, buffer.far_safe,
                                  t_v, editor.beta, a_v)
            history.append({
                "step": step, "loss": float(loss.item()),
                "forget_loss": float(forget_loss.item()),
                "retain_loss": float(retain_loss.item()),
                "locality_loss": float(locality_loss.item()),
                "alpha": a_v, "tau": t_v, "locality_ratio": loc,
            })
            print(
                f"  step {step:4d}/{args.editor_steps} | loss={loss.item():.5f} | "
                f"alpha={a_v:.4f} | tau={t_v:.4f} | locality_ratio={loc:.4f}"
            )

    final_alpha = float(editor.alpha.detach().cpu().item())
    final_tau   = float(editor.tau.detach().cpu().item())
    final_ratio = _locality_ratio(scores, buffer.near_hazard, buffer.far_safe,
                                  final_tau, editor.beta, final_alpha)

    print(
        f"\n[Phase 2] Final | alpha={final_alpha:.4f} | tau={final_tau:.4f} | "
        f"locality_ratio={final_ratio:.4f}"
    )

    return editor, {
        "tau_sweep":          sweep_stats,
        "alpha_analytical":   alpha_init,
        "alpha_final":        final_alpha,
        "tau_final":          final_tau,
        "locality_ratio":     final_ratio,
        "refinement_history": history,
    }


# =============================================================================
# Phase 2 diagnostics (run once before Phase 3)
# =============================================================================

@torch.no_grad()
def editor_diagnostics(
    agent:       PPOUnlearner,
    editor:      RepresentationEditor,
    direction_t: torch.Tensor,
    buffer:      OfflineBuffer,
    sample_n:    int = 4096,
) -> dict:
    """Full concept-suppression diagnostics over the offline buffer."""
    device = agent.device
    idx    = np.random.choice(buffer.n_steps, min(sample_n, buffer.n_steps), replace=False)
    s_t    = torch.as_tensor(buffer.states[idx], dtype=torch.float32, device=device)
    near_t = torch.as_tensor(buffer.near_hazard[idx], dtype=torch.bool, device=device)
    far_t  = torch.as_tensor(buffer.far_safe[idx],    dtype=torch.bool, device=device)

    raw_feat  = agent.policy.encode(s_t)
    score     = raw_feat @ direction_t
    gate      = torch.sigmoid(editor.beta * (score - editor.tau))
    relu_part = torch.relu(score - editor.tau)
    post_edit = score - editor.alpha * gate * relu_part
    supp      = editor.alpha * gate * relu_part

    def _m(x, mask=None):
        x = x[mask] if (mask is not None and mask.any()) else x
        return float(x.mean().item()) if len(x) > 0 else float("nan")

    near_s = _m(supp, near_t)
    far_s  = _m(supp, far_t)

    return {
        "Diag/RawScore_Mean":       _m(score),
        "Diag/RawScore_Near":       _m(score, near_t),
        "Diag/RawScore_Far":        _m(score, far_t),
        "Diag/PostEditScore_Mean":  _m(post_edit),
        "Diag/PostEditScore_Near":  _m(post_edit, near_t),
        "Diag/PostEditScore_Far":   _m(post_edit, far_t),
        "Diag/Suppression_Mean":    _m(supp),
        "Diag/Suppression_Near":    near_s,
        "Diag/Suppression_Far":     far_s,
        "Diag/Locality_Ratio":      near_s / (far_s + 1e-9),
        "Diag/GateMean":            _m(gate),
        "Diag/GateMean_Near":       _m(gate, near_t),
        "Diag/GateMean_Far":        _m(gate, far_t),
    }


# =============================================================================
# Phase 3 — Online evaluation (no training)
# =============================================================================

def phase3_eval_round(
    agent:           PPOUnlearner,
    env_id:          str,
    n_envs:          int,
    horizon:         int,
    update:          int,
    env_steps_total: int,
) -> Tuple[dict, int]:
    """
    One evaluation rollout block with the frozen edited policy.
    No backward pass. Returns metrics and updated step counter.
    """
    env = make_env(env_id, n_envs=n_envs)
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    batch_rew    = 0.0
    batch_cforget = 0.0

    for _ in range(horizon):
        with torch.no_grad():
            act, _, _, _ = agent.act(obs, deterministic=False)
        next_obs, rew, cost, term, trunc, info = env.step(act)
        c_forget = extract_cforget(cost, info)

        batch_rew     += float(np.asarray(rew,      dtype=np.float32).mean())
        batch_cforget += float(np.asarray(c_forget, dtype=np.float32).mean())
        obs             = next_obs
        env_steps_total += n_envs

    env.close()

    return {
        "Eval/Update":            update,
        "Eval/MeanRewardStep":    batch_rew     / horizon,
        "Eval/CostRate":          batch_cforget / horizon,
        "RepEdit/Alpha":          float(agent.editor.alpha.detach().cpu().item()),
        "RepEdit/Tau":            float(agent.editor.tau.detach().cpu().item()),
        "Compute/EnvStepsTotal":  env_steps_total,
    }, env_steps_total


# =============================================================================
# Main
# =============================================================================

def train_repedit(args):
    wandb.init(
        project=args.project,
        name=f"Unlearn_repedit_{args.env}_Seed{args.seed}",
        group=f"Unlearning_repedit_{args.env}",
        config=vars(args),
    )
    set_seed(args.seed)

    # ------------------------------------------------------------------
    # Bootstrap: frozen unsafe expert, no editor.
    # ------------------------------------------------------------------
    bootstrap_env = make_env(args.env, n_envs=1)
    unsafe_agent  = PPOUnlearner(
        bootstrap_env,
        lr=args.lr, ent_coef=args.ent_coef,
        ppo_epochs=args.ppo_epochs, batch_size=args.batch_size,
    )
    unsafe_agent.load(model_checkpoint_path("unsafe", args.env))
    unsafe_agent.freeze_policy()
    bootstrap_env.close()

    # ------------------------------------------------------------------
    # Load concept direction artifact.
    # ------------------------------------------------------------------
    direction, artifact_meta = load_repedit_artifact(args.repedit_artifact)
    print(f"[Init] Direction loaded from {args.repedit_artifact} | shape={direction.shape}")

    # Allow artifact tau to propagate as manual override.
    if args.repedit_use_artifact_tau and "tau" in artifact_meta and args.repedit_tau is None:
        args.repedit_tau = float(artifact_meta["tau"])
        print(f"[Init] Using artifact tau = {args.repedit_tau:.6f}")

    # ------------------------------------------------------------------
    # Phase 1: fixed offline buffer.
    # ------------------------------------------------------------------
    buffer = phase1_collect_buffer(
        unsafe_agent = unsafe_agent,
        env_id       = args.env,
        seed         = args.seed,
        n_steps      = args.buffer_steps,
        lookahead    = args.near_hazard_lookahead,
    )
    wandb.log({
        "Phase1/BufferSize": buffer.n_steps,
        "Phase1/NHazard":    buffer.n_hazard,
        "Phase1/HazardRate": buffer.n_hazard / buffer.n_steps,
        "Phase1/NNear":      buffer.n_near,
        "Phase1/NFar":       buffer.n_far,
    })

    # ------------------------------------------------------------------
    # Phase 2: offline editor training.
    # ------------------------------------------------------------------
    editor, phase2_stats = phase2_train_editor(
        agent=unsafe_agent, buffer=buffer, direction=direction, args=args,
    )

    # Log Phase 2 summary.
    wandb.log({
        "Phase2/AlphaAnalytical": phase2_stats["alpha_analytical"],
        "Phase2/AlphaFinal":      phase2_stats["alpha_final"],
        "Phase2/TauFinal":        phase2_stats["tau_final"],
        "Phase2/LocalityRatio":   phase2_stats["locality_ratio"],
        "Phase2/TauSource":       phase2_stats["tau_sweep"].get("source", "unknown"),
    })
    for h in phase2_stats["refinement_history"]:
        wandb.log(
            {f"Phase2/{k}": v for k, v in h.items() if k != "step"},
            step=h["step"],
        )

    # Post-training diagnostics on the fixed buffer.
    direction_t = torch.as_tensor(direction, dtype=torch.float32, device=unsafe_agent.device)
    direction_t = direction_t / (direction_t.norm() + 1e-8)
    diag = editor_diagnostics(unsafe_agent, editor, direction_t, buffer)
    wandb.log(diag)
    print("\n[Phase 2 Diagnostics]")
    for k, v in diag.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Phase 3: attach frozen editor → online eval only.
    # ------------------------------------------------------------------
    unsafe_agent.set_editor(editor)
    editor.eval()   # fully frozen from here on

    env_steps_total = args.unsafe_updates * args.horizon * args.n_envs
    save_points     = set(args.save_points)

    print(f"\n[Phase 3] Online evaluation | {args.eval_updates} rounds")

    for update in range(1, args.eval_updates + 1):
        metrics, env_steps_total = phase3_eval_round(
            agent           = unsafe_agent,
            env_id          = args.env,
            n_envs          = args.n_envs,
            horizon         = args.horizon,
            update          = update,
            env_steps_total = env_steps_total,
        )
        wandb.log(metrics, step=env_steps_total)

        print(
            f"[Phase 3] Update {update}/{args.eval_updates} | "
            f"Rew {metrics['Eval/MeanRewardStep']:.4f} | "
            f"Cost {metrics['Eval/CostRate']:.4f} | "
            f"Alpha {metrics['RepEdit/Alpha']:.4f} | "
            f"Tau {metrics['RepEdit/Tau']:.4f}"
        )

        if update in save_points:
            path = model_checkpoint_path("repedit", args.env, update)
            save_repedit_checkpoint(
                path       = path,
                agent      = unsafe_agent,
                editor     = editor,
                extra_meta = {
                    "mode":             "repedit",
                    "env":              args.env,
                    "seed":             args.seed,
                    "artifact_path":    args.repedit_artifact,
                    "alpha_analytical": phase2_stats["alpha_analytical"],
                    "alpha_final":      phase2_stats["alpha_final"],
                    "tau_final":        phase2_stats["tau_final"],
                    "locality_ratio":   phase2_stats["locality_ratio"],
                    "tau_source":       phase2_stats["tau_sweep"].get("source", "unknown"),
                },
            )
            wandb.save(path)
            print(f"[Phase 3] Checkpoint saved → {path}")

    wandb.finish()


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="3-phase offline RepEdit for policy unlearning."
    )

    # Environment
    p.add_argument("--env",     type=str, default="SafetyPointGoal1-v0")
    p.add_argument("--project", type=str, default="Reifule")
    p.add_argument("--seed",    type=int, default=0)

    # Rollout (Phase 3 + step-counter alignment)
    p.add_argument("--n_envs",         type=int, default=8)
    p.add_argument("--horizon",        type=int, default=1024)
    p.add_argument("--eval_updates",   type=int, default=150)
    p.add_argument("--unsafe_updates", type=int, default=300,
                   help="Used only to align env_steps_total with other runs.")
    p.add_argument("--save_points", type=int, nargs="+", default=[50, 100, 150])

    # PPOUnlearner constructor args (needed to build the agent, not for training)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--ent_coef",   type=float, default=0.03)
    p.add_argument("--ppo_epochs", type=int,   default=6)
    p.add_argument("--batch_size", type=int,   default=256)

    # Phase 1: buffer collection
    p.add_argument("--buffer_steps",          type=int, default=50_000,
                   help="Single-env steps to collect from the unsafe expert.")
    p.add_argument("--near_hazard_lookahead", type=int, default=20,
                   help="Lookahead window for near-hazard labelling.")

    # Concept direction artifact
    p.add_argument(
        "--repedit_artifact", type=str,
        default=probe_artifact_path("hazard_direction_SafetyPointGoal1-v0.pt"),
    )
    p.add_argument("--repedit_use_artifact_tau", action="store_true",
                   help="Use the tau stored in the artifact (skips sweep).")

    # Phase 2a: tau sweep
    p.add_argument("--repedit_tau",          type=float, default=None,
                   help="Hard-override for tau. Skips sweep if set.")
    p.add_argument("--tau_quantile_min",     type=float, default=0.30)
    p.add_argument("--tau_quantile_max",     type=float, default=0.90)
    p.add_argument("--tau_sweep_candidates", type=int,   default=13)

    # Phase 2b: analytical alpha
    p.add_argument("--alpha_target_percentile", type=float, default=1.0,
                   help="Target Nth-percentile of far-safe scores as the "
                        "desired post-edit hazard score.")
    p.add_argument("--alpha_max", type=float, default=50.0)

    # Phase 2c: gradient refinement
    p.add_argument("--repedit_beta",      type=float, default=10.0,
                   help="Gate sharpness. Fixed, not learned.")
    p.add_argument("--editor_steps",      type=int,   default=500)
    p.add_argument("--editor_batch_size", type=int,   default=512)
    p.add_argument("--editor_lr",         type=float, default=5e-3,
                   help="Adam lr for alpha.")
    p.add_argument("--editor_tau_lr",     type=float, default=1e-3,
                   help="Adam lr for tau. Keep small — stay near sweep init.")

    # Loss weights
    p.add_argument("--forget_coef",   type=float, default=1.0)
    p.add_argument("--retain_coef",   type=float, default=1.0)
    p.add_argument("--locality_coef", type=float, default=0.5)

    args = p.parse_args()
    train_repedit(args)