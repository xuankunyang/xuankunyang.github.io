---
title: RAG-TA
date: 2026-01-18
external_link: https://github.com/xuankunyang/RL-Project
tags:
  - Reinforcement Learning
  - Deep Q-Networks
  - Proximal Policy Optimization
---

# RL-Project: Comprehensive Reinforcement Learning Framework for Atari and MuJoCo

This project implements a comprehensive Reinforcement Learning framework capable of solving both discrete control tasks (Atari games using DQN) and continuous control tasks (MuJoCo robotics using PPO). It is designed for modularity, scalability, and ease of experimentation, featuring automated parallel training, configuration-driven evaluation, and robust headless visualization support.

**Key Features:**
*   **DQN (Deep Q-Network):** Supports Vanilla, Double, Dueling, and Rainbow variants.
*   **PPO (Proximal Policy Optimization):** Optimized for continuous control with observation normalization and reward clipping.
*   **Parallel Training:** Efficient data collection using vectorized environments.
*   **Automated Evaluation:** `run.py` for rendering, video recording, and performance metrics.
*   **Configuration Registry:** Centralized management of best model checkpoints via `configs/best_models.py`.

<!--more-->