# 🤖 BipedalWalker-v3 — Deep RL with Proximal Policy Optimization

<div align="center">

**Teaching a simulated robot to walk using Deep Reinforcement Learning**

![Bipedal Walker](videos/demo.gif)

*A PPO agent learning to master continuous locomotion control over 1 M+ timesteps.*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-2.3+-FF6F00?logo=python)](https://stable-baselines3.readthedocs.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-0081A5)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 Overview

This project trains an autonomous bipedal robot to walk across rough terrain using **Proximal Policy Optimization (PPO)** — a state-of-the-art policy-gradient algorithm for continuous control. The agent receives **24-dimensional observations** (hull angle, velocity, joint angles, leg contact, lidar) and outputs **4 continuous actions** (torques for hip and knee joints on each leg).

| Component            | Details                              |
|----------------------|--------------------------------------|
| **Environment**      | `BipedalWalker-v3` (Gymnasium)       |
| **Algorithm**        | PPO (clip objective, GAE)            |
| **Library**          | Stable Baselines3                    |
| **Observation space**| `Box(24,)` — continuous              |
| **Action space**     | `Box(4,)` — continuous torques       |
| **Solved threshold** | Mean reward ≥ 300 over 100 episodes  |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Saumojyoti-Shiv/BipedalWalker-v3.git
cd BipedalWalker-v3
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On Linux you may also need `sudo apt-get install swig` for the Box2D bindings.

---

## 🏋️ Train the Agent

```bash
python train.py
```

#### Optional flags

| Flag            | Default     | Description                           |
|-----------------|-------------|---------------------------------------|
| `--timesteps`   | `1000000`   | Total training environment steps      |
| `--n-envs`      | `4`         | Number of parallel environments       |

```bash
# Example: train for 2 million steps with 8 workers
python train.py --timesteps 2e6 --n-envs 8
```

Training progress is logged to `logs/` — launch TensorBoard to monitor live:

```bash
tensorboard --logdir logs
```

The best model is saved automatically to `models/best_model.zip`.

---

## 👀 Watch the Trained Agent

```bash
python watch_agent.py
```

This opens a **live pygame window** showing the robot walking and simultaneously records a video to `videos/`.

#### Optional flags

| Flag            | Default                  | Description                     |
|-----------------|--------------------------|---------------------------------|
| `--model`       | `models/best_model.zip`  | Path to the model checkpoint    |
| `--episodes`    | `3`                      | Number of episodes to run       |
| `--no-render`   | *off*                    | Headless mode (record only)     |
| `--no-record`   | *off*                    | Disable video recording         |

```bash
# Record 5 episodes without opening a window
python watch_agent.py --episodes 5 --no-render
```

---

## 🧠 How PPO Learns to Walk

**Proximal Policy Optimization** is an actor-critic method that simultaneously trains:

1. **Policy (Actor)** — a neural network that maps observations → action distributions.
2. **Value function (Critic)** — a neural network that estimates future cumulative reward.

### The training loop

```
Observe state ─→ Sample action from policy ─→ Execute in environment
       ↑                                              │
       └─── Update policy using clipped objective ←───┘
                     (maximize advantage while
                      staying close to old policy)
```

Key mechanisms that make PPO effective for locomotion:

| Mechanism                       | Why it matters                                                      |
|---------------------------------|---------------------------------------------------------------------|
| **Clipped surrogate objective** | Prevents destructively large policy updates that cause the walker to fall. |
| **Generalized Advantage Estimation (GAE)** | Balances bias vs. variance in advantage estimates for smoother learning. |
| **Observation normalisation**   | Keeps inputs zero-mean / unit-variance so the network trains stably. |
| **Reward normalisation**        | Stabilises value-function learning when reward magnitudes vary.     |
| **Vectorized environments**     | Collects diverse experience in parallel, dramatically speeding up training. |

Over ~1 M steps the agent progresses from **falling immediately** → **stumbling forward** → **smooth, efficient walking** with a reward above 300.

---

## 📁 Project Structure

```
bipedal-walker-ppo/
├── train.py              # Training pipeline (PPO + SubprocVecEnv)
├── watch_agent.py         # Visualization & video recording
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── models/                # Saved model checkpoints (auto-created)
│   ├── best_model.zip
│   ├── final_model.zip
│   └── vec_normalize.pkl
├── videos/                # Recorded episode videos (auto-created)
│   └── bipedal-walker-ppo-episode-0.mp4
└── logs/                  # TensorBoard training logs (auto-created)
```

---

## 👥 Team Members

| Name                     | Roll Number   |
|--------------------------|---------------|
| **Saumojyoti Chakraborty** | 23BAI11065    |
| **Antisha Ray**            | 23BAI10516    |

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Built with ❤️ using Stable Baselines3, Gymnasium &amp; PPO</sub>
</div>
