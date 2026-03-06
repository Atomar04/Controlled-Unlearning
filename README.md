# Targeted Reinforcement Unlearning via PID Lagrangian Control

This repository contains the official implementation of our framework for **Targeted Reinforcement Unlearning** in continuous control environments. 

Our method performs post-hoc safety repair on pre-trained, unconstrained Reinforcement Learning (RL) agents. By utilizing a Dual-Head Proximal Policy Optimization (PPO) architecture and a PID-Lagrangian controller, we surgically suppress specific unsafe behaviors (represented as a binary concept signal $C_{\text{forget}}$) without requiring computationally expensive retraining from scratch.



## Repository Structure

```text
ReUle/
├── reifule/
│   ├── algorithm.py           # Dual-Head PPO Unlearner architecture
│   └── computation_amnesiac.py# Vectorized GAE and PID Lagrangian Controller
├── scripts/
│   ├── agent.py               # Trains the Unsafe Expert
│   ├── train.py               # Runs the Reinforcement Unlearning process
│   ├── oracle.py              # Trains the Oracle Safe Agent from scratch
│   └── eval.py                # Evaluates policies and generates metric tables/videos
└── requirements.txt           # Project dependencies

```
