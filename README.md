# Controlled Unlearning

This repository contains the official implementation of our framework for **Targeted Reinforcement Unlearning** in continuous control environments. 

Our method performs post-hoc safety repair on pre-trained, unconstrained Reinforcement Learning (RL) agents. By utilizing a Dual-Head Proximal Policy Optimization (PPO) architecture and a PID-Lagrangian controller, we surgically suppress specific unsafe behaviors (represented as a binary concept signal $C_{\text{forget}}$) without requiring computationally expensive retraining from scratch.



## Repository Structure

```text
Controlled-Unlearning/
├── README.md
├── requirements.txt
│
├── reifule/
│   ├── __init__.py
│   ├── algorithm.py
│   ├── computation_amnesiac.py
│   ├── experiments.py
│   └── utils.py
│
├── scripts/
│   ├── __init__.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_unsafe.py
│   │   ├── train_oracle.py
│   │   ├── train_concept.py
│   │   ├── train_trajectory_decremental.py
│   │   └── train_repedit.py
│   │
│   └── eval_analysis/
│       ├── __init__.py
│       ├── eval_policy_suite.py
│       ├── relearn_test.py
│       ├── probe_representation.py
│       ├── analyze_latent_shift.py
│       └── export_hazard_direction.py
│
└── artifacts/
    ├── models/
    │   ├── unsafe/
    │   │   └── unsafe_expert_<env>.pt
    │   │
    │   ├── oracle/
    │   │   ├── oracle_<env>_300.pt
    │   │   ├── oracle_<env>_350.pt
    │   │   ├── oracle_<env>_400.pt
    │   │   └── oracle_<env>_450.pt
    │   │
    │   ├── concept/
    │   │   ├── safe_concept_<env>_50.pt
    │   │   ├── safe_concept_<env>_100.pt
    │   │   └── safe_concept_<env>_150.pt
    │   │
    │   ├── trajectory_decremental/
    │   │   ├── safe_trajectory_decremental_<env>_50.pt
    │   │   ├── safe_trajectory_decremental_<env>_100.pt
    │   │   └── safe_trajectory_decremental_<env>_150.pt
    │   │
    │   └── repedit/
    │       ├── safe_repedit_<env>_50.pt
    │       ├── safe_repedit_<env>_100.pt
    │       └── safe_repedit_<env>_150.pt
    │
    ├── probes/
    │   ├── probe_states_fixed_v2.npz
    │   ├── hazard_probe.pkl
    │   ├── hazard_probe_metrics.json
    │   ├── hazard_probe_v2.pkl                
    │   ├── hazard_probe_concept.pkl          # optional reverse/concept variant
    │   ├── hazard_direction.pt
    │   ├── concept_probe_eval.json           # optional eval output
    │   └── repedit_probe_eval.json           # optional eval output
    │
    ├── eval/
    │   ├── policy_suite_eval/
    │   │   └── <eval_id>/
    │   │       ├── summary.csv
    │   │       ├── per_episode.csv
    │   │       ├── summary.json
    │   │       ├── metadata.json
    │   │       └── videos/
    │   │           ├── unsafe.gif
    │   │           ├── oracle_<step>.gif
    │   │           ├── concept_<step>.gif
    │   │           ├── trajectory_decremental_<step>.gif
    │   │           └── repedit_<step>.gif
    │   │
    │   ├── relearn_test_eval/
    │   │   └── <eval_id>/
    │   │       ├── threshold_only.json       # when using --compute_threshold_only
    │   │       ├── relearn_history.csv
    │   │       └── relearn_result.json
    │   │
    │   └── latent_shift_eval/
    │       └── <eval_id>/
    │           └── latent_shift_metrics.json
    │
    └── logs/
        ├── train/
        ├── eval/
        └── debug/
```

# Running the Project

> Run all commands from the **repository root** (`Controlled-Unlearning/`).

## 1. Train the unsafe expert

```bash
python -m scripts.training.train_unsafe \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --n_envs 8 \
  --horizon 1024 \
  --updates 300 \
  --lr 3e-4 \
  --ent_coef 0.03 \
  --ppo_epochs 6 \
  --batch_size 256
```
Output will be saved as ```artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt```

## 2. Train the oracle (constrained-from-scratch safe baseline)

```bash
python -m scripts.training.train_oracle \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --n_envs 8 \
  --horizon 1024 \
  --updates 450 \
  --lr 3e-4 \
  --ent_coef 0.03 \
  --ppo_epochs 6 \
  --batch_size 256 \
  --kp 0.5 \
  --ki 0.003 \
  --kd 0.02 \
  --lambda_init 0.0 \
  --lambda_max 200.0 \
  --integral_max 5.0 \
  --target_cost 0.03
```
Saved checkpoints
- ```artifacts/models/oracle/oracle_SafetyPointGoal1-v0_300.pt```
- ```artifacts/models/oracle/oracle_SafetyPointGoal1-v0_350.pt```
- ```artifacts/models/oracle/oracle_SafetyPointGoal1-v0_400.pt```
- ```artifacts/models/oracle/oracle_SafetyPointGoal1-v0_450.pt```


## 3. Train concept-level unlearning

```bash
python -m scripts.training.train_concept \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --n_envs 8 \
  --horizon 1024 \
  --updates 150 \
  --unsafe_updates 300 \
  --lr 3e-4 \
  --ent_coef 0.03 \
  --ppo_epochs 6 \
  --batch_size 256 \
  --kp 2.0 \
  --ki 0.01 \
  --kd 0.1 \
  --lambda_init 0.0 \
  --lambda_max 200.0 \
  --integral_max 5.0 \
  --target_cost 0.03 \
  --save_points 50 100 150
```

Saved checkpoints
- ```artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_50.pt```
- ```artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_100.pt```
- ```artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_150.pt```

## 4. Train trajectory-decremental unlearning

```bash
python -m scripts.training.train_trajectory_decremental \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --updates 150 \
  --unsafe_updates 300 \
  --lr 3e-4 \
  --ent_coef 0.03 \
  --ppo_epochs 6 \
  --batch_size 256 \
  --dataset_episodes 200 \
  --decremental_epochs_per_update 10 \
  --decremental_bc_coef 1.0 \
  --decremental_value_coef 0.5 \
  --decremental_ent_coef 0.01 \
  --recollect_every 10 \
  --save_points 50 100 150
```

Saved checkpoints
- ```artifacts/models/trajectory_decremental/safe_trajectory_decremental_SafetyPointGoal1-v0_50.pt```
- ```artifacts/models/trajectory_decremental/safe_trajectory_decremental_SafetyPointGoal1-v0_100.pt```
- ```artifacts/models/trajectory_decremental/safe_trajectory_decremental_SafetyPointGoal1-v0_150.pt```


## 5. Probe representation dataset collection
Collect a balanced safe/hazard dataset from the unsafe expert:
```bash
python -m scripts.eval_analysis.probe_representation collect \
  --env SafetyPointGoal1-v0 \
  --model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --out artifacts/probes/probe_states_fixed_v2.npz \
  --n_per_class 10000 \
  --max_env_steps 400000 \
  --seed 42
```

Output: ```artifacts/probes/probe_states_fixed_v2.npz```

## 6. Fit the hazard probe

```bash
python -m scripts.eval_analysis.probe_representation fit \
  --env SafetyPointGoal1-v0 \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --out artifacts/probes/hazard_probe.pkl \
  --metrics_out artifacts/probes/hazard_probe_metrics.json \
  --batch_size 4096 \
  --val_frac 0.15 \
  --test_frac 0.15 \
  --seed 42 \
  --split_tries 256
```

Outputs:
- ```artifacts/probes/hazard_probe.pkl```
- ```artifacts/probes/hazard_probe_metrics.json```

## 7. Export hazard direction for representation editing

```bash
python -m scripts.eval_analysis.export_hazard_direction \
  --probe_path artifacts/probes/hazard_probe.pkl \
  --out artifacts/probes/hazard_direction.pt
```
Output: ```artifacts/probes/hazard_direction.pt```


## 8. Train representation editing (repedit)

```bash
python -m scripts.training.train_repedit \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --updates 150 \
  --unsafe_updates 300 \
  --n_envs 8 \
  --horizon 1024 \
  --lr 3e-4 \
  --ent_coef 0.03 \
  --ppo_epochs 6 \
  --batch_size 256 \
  --repedit_artifact artifacts/probes/hazard_direction.pt \
  --repedit_alpha 1.5 \
  --repedit_tau_quantile 0.8 \
  --repedit_beta 10.0 \
  --save_points 50 100 150
```
Saved checkpoints:
- ```artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_50.pt```
- ```artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_100.pt```
- ```artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_150.pt```

Variant:\
Mannual ```tau```:

```bash
python -m scripts.training.train_repedit \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --updates 150 \
  --unsafe_updates 300 \
  --n_envs 8 \
  --horizon 1024 \
  --repedit_artifact artifacts/probes/hazard_direction.pt \
  --repedit_alpha 1.5 \
  --repedit_tau 0.5 \
  --repedit_beta 10.0
```
Use artifact tau if exported:
```bash
python -m scripts.training.train_repedit \
  --env SafetyPointGoal1-v0 \
  --seed 0 \
  --updates 150 \
  --unsafe_updates 300 \
  --n_envs 8 \
  --horizon 1024 \
  --repedit_artifact artifacts/probes/hazard_direction.pt \
  --repedit_alpha 1.5 \
  --repedit_use_artifact_tau \
  --repedit_beta 10.0
```

## 9. Run full policy evaluation suite
This evaluates:
- unsafe experrt
- oracle
- concept checkpoints
- trajectory-decremental checkpoints
- rep-edit checkpoints

and saves all outputs locally.
```bash
python -m scripts.eval_analysis.eval_policy_suite \
  --env SafetyPointGoal1-v0 \
  --episodes 50 \
  --max_steps 2000 \
  --kl_states 5000 \
  --record_videos
```
Outputs:
- ```artifacts/eval/policy_suite_eval/<eval_id>/summary.csv```
- ```artifacts/eval/policy_suite_eval/<eval_id>/per_episode.csv```
- ```artifacts/eval/policy_suite_eval/<eval_id>/summary.json```
- ```artifacts/eval/policy_suite_eval/<eval_id>/metadata.json```
- ```artifacts/eval/policy_suite_eval/<eval_id>/videos/```

To use specific oracle as reference use flag ```--oracle_ref_path```

## 10. Relearn Test
First compute the unsafe expert cost threshold:
```bash
python -m scripts.eval_analysis.relearn_test \
  --env SafetyPointGoal1-v0 \
  --model_path random \
  --unsafe_model artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --compute_threshold_only
```

Output: ```artifacts/eval/relearn_test_eval/<eval_id>/threshold_only.json```
\
Run relearning test:
\
Example for concept checkpoint:
```bash
python -m scripts.eval_analysis.relearn_test \
  --env SafetyPointGoal1-v0 \
  --model_path artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_150.pt \
  --unsafe_model artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --threshold 0.09 \
  --horizon 1024 \
  --max_updates 100 \
  --eval_every 5 \
  --eval_episodes 5
```

Example for trajectory-decremental checkpoint:
```bash
python -m scripts.eval_analysis.relearn_test \
  --env SafetyPointGoal1-v0 \
  --model_path artifacts/models/trajectory_decremental/safe_trajectory_decremental_SafetyPointGoal1-v0_150.pt \
  --unsafe_model artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --threshold 0.09 \
  --horizon 1024 \
  --max_updates 100 \
  --eval_every 5 \
  --eval_episodes 5
```

Example for repedit checkpoint:
```bash
python -m scripts.eval_analysis.relearn_test \
  --env SafetyPointGoal1-v0 \
  --model_path artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_150.pt \
  --unsafe_model artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --threshold 0.09 \
  --horizon 1024 \
  --max_updates 100 \
  --eval_every 5 \
  --eval_episodes 5
```
 Outputs:
 - ```artifacts/eval/relearn_test_eval/<eval_id>/relearn_history.csv```
 - ```artifacts/eval/relearn_test_eval/<eval_id>/relearn_result.json```

## 11. Evaluate a fitted probe on another model
Example: evaluate the unsafe-trained probe on a concept-unlearned policy.
```bash
python -m scripts.eval_analysis.probe_representation eval \
  --probe_path artifacts/probes/hazard_probe.pkl \
  --model_path artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_150.pt \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --metrics_out artifacts/probes/concept_probe_eval.json
```

Example: evaluate on repedit using edited latent features:
```bash
python -m scripts.eval_analysis.probe_representation eval \
  --probe_path artifacts/probes/hazard_probe.pkl \
  --model_path artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_150.pt \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --metrics_out artifacts/probes/repedit_probe_eval.json \
  --use_editor_features
```

## 12. Analyze latent shift between two policies
Unsafe vs concept:
```bash
python -m scripts.eval_analysis.analyze_latent_shift \
  --env SafetyPointGoal1-v0 \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --ref_model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --cmp_model_path artifacts/models/concept/safe_concept_SafetyPointGoal1-v0_150.pt
```

Unsafe vs trajectory-decremental:
```bash
python -m scripts.eval_analysis.analyze_latent_shift \
  --env SafetyPointGoal1-v0 \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --ref_model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --cmp_model_path artifacts/models/trajectory_decremental/safe_trajectory_decremental_SafetyPointGoal1-v0_150.pt
```

Unsafe vs repedit (operative edited latent):
```bash
python -m scripts.eval_analysis.analyze_latent_shift \
  --env SafetyPointGoal1-v0 \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --ref_model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --cmp_model_path artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_150.pt
```
Unsafe vs repedit (raw backbone latent only):
```bash
python -m scripts.eval_analysis.analyze_latent_shift \
  --env SafetyPointGoal1-v0 \
  --dataset artifacts/probes/probe_states_fixed_v2.npz \
  --ref_model_path artifacts/models/unsafe/unsafe_expert_SafetyPointGoal1-v0.pt \
  --cmp_model_path artifacts/models/repedit/safe_repedit_SafetyPointGoal1-v0_150.pt \
  --raw_features
```

Output:
```artifacts/eval/latent_shift_eval/<eval_id>/latent_shift_metrics.json```