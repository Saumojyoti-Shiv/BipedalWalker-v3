"""
==========================================================================
 BipedalWalker-v3 — PPO Training Script
==========================================================================
 Trains a Proximal Policy Optimization (PPO) agent to solve the
 BipedalWalker-v3 continuous-control task from Gymnasium.

 Key features
 ------------
 * Vectorized environments via SubprocVecEnv for parallel rollouts.
 * VecNormalize wrapper to normalize observations & rewards on-the-fly.
 * EvalCallback that evaluates every 10 000 steps, saving the best model.
 * TensorBoard logging for real-time training curves.

 Usage
 -----
     python train.py                    # train with default settings
     python train.py --timesteps 2e6   # override total timesteps
     python train.py --n-envs 8        # use 8 parallel environments

 License: MIT
==========================================================================
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Callable

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize


# ──────────────────────────────────────────────────────────────
# Configuration defaults
# ──────────────────────────────────────────────────────────────
DEFAULT_TIMESTEPS: int = 1_000_000       # total training steps
DEFAULT_N_ENVS: int = 4                  # parallel environments
EVAL_FREQ: int = 10_000                  # evaluate every N steps
N_EVAL_EPISODES: int = 5                 # episodes per evaluation
MODEL_DIR: str = "models"                # directory for saved models
LOG_DIR: str = "logs"                    # TensorBoard log directory
BEST_MODEL_NAME: str = "best_model"      # filename for the best checkpoint


# ──────────────────────────────────────────────────────────────
# Helper: environment factory
# ──────────────────────────────────────────────────────────────
def make_env(rank: int, seed: int = 0) -> Callable[[], gym.Env]:
    """
    Return a *thunk* (zero-argument callable) that creates a single
    BipedalWalker-v3 environment. Each environment gets its own seed
    so that the parallel workers explore different trajectories.

    Parameters
    ----------
    rank : int
        Worker index (used to offset the seed).
    seed : int
        Base random seed.

    Returns
    -------
    Callable that instantiates the environment.
    """
    def _init() -> gym.Env:
        env = gym.make("BipedalWalker-v3")
        env.reset(seed=seed + rank)
        return env
    return _init


# ──────────────────────────────────────────────────────────────
# PPO hyper-parameters (tuned for continuous locomotion tasks)
# ──────────────────────────────────────────────────────────────
PPO_HYPERPARAMS: dict = {
    "learning_rate":    3e-4,        # Adam learning rate
    "n_steps":          2048,        # rollout buffer length per env
    "batch_size":       64,          # minibatch size for SGD updates
    "n_epochs":         10,          # SGD passes per rollout
    "gamma":            0.99,        # discount factor
    "gae_lambda":       0.95,        # GAE λ for advantage estimation
    "clip_range":       0.2,         # PPO surrogate clipping ε
    "ent_coef":         0.0,         # entropy bonus (0 = off)
    "vf_coef":          0.5,         # value-function loss coefficient
    "max_grad_norm":    0.5,         # gradient clipping
    "policy_kwargs":    dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),  # two hidden layers
    ),
}


# ──────────────────────────────────────────────────────────────
# Main training routine
# ──────────────────────────────────────────────────────────────
def train(total_timesteps: int, n_envs: int) -> None:
    """
    End-to-end training pipeline:
      1. Build vectorized + normalised environments.
      2. Instantiate PPO with tuned hyper-parameters.
      3. Attach an EvalCallback for periodic evaluation.
      4. Train and save the final model + normalisation stats.

    Parameters
    ----------
    total_timesteps : int
        Total number of environment steps to collect.
    n_envs : int
        Number of parallel SubprocVecEnv workers.
    """

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ── 1. Vectorized training environments ──────────────────
    print(f"\n{'='*60}")
    print(f"  [+] Creating {n_envs} parallel BipedalWalker-v3 environments")
    print(f"{'='*60}\n")

    train_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Normalise observations (zero-mean, unit-variance) and rewards.
    # This is critical for stable PPO training on continuous tasks.
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True,
                             clip_obs=10.0)

    # ── 2. Separate evaluation environment ───────────────────
    eval_env = SubprocVecEnv([make_env(i, seed=42) for i in range(1)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False,
                            clip_obs=10.0, training=False)

    # ── 3. Instantiate PPO ───────────────────────────────────
    print("  [*] Initializing PPO agent ...")
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        seed=0,
        device="auto",       # uses CUDA if available, else CPU
        **PPO_HYPERPARAMS,
    )
    print(f"       Policy architecture: {model.policy}")
    print(f"       Device            : {model.device}\n")

    # ── 4. Evaluation callback ───────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(EVAL_FREQ // n_envs, 1),   # per-env frequency
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False,
    )

    # ── 5. Train! ────────────────────────────────────────────
    print(f"  [>] Starting training for {total_timesteps:,} timesteps ...")
    print(f"      Evaluating every {EVAL_FREQ:,} steps "
          f"({N_EVAL_EPISODES} episodes each)\n")

    start = time.time()
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,     # tqdm progress bar in the terminal
    )
    elapsed = time.time() - start

    # ── 6. Save final model & normalisation stats ────────────
    final_path = os.path.join(MODEL_DIR, "final_model")
    model.save(final_path)
    train_env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    print(f"\n{'='*60}")
    print(f"  [OK] Training complete!")
    print(f"       Wall-clock time : {elapsed / 60:.1f} minutes")
    print(f"       Best model saved: {MODEL_DIR}/{BEST_MODEL_NAME}.zip")
    print(f"       Final model     : {final_path}.zip")
    print(f"       VecNormalize    : {MODEL_DIR}/vec_normalize.pkl")
    print(f"{'='*60}\n")

    # Cleanup
    train_env.close()
    eval_env.close()


# ──────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on BipedalWalker-v3"
    )
    parser.add_argument(
        "--timesteps", type=float, default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--n-envs", type=int, default=DEFAULT_N_ENVS,
        help=f"Number of parallel environments (default: {DEFAULT_N_ENVS})"
    )
    args = parser.parse_args()

    train(total_timesteps=int(args.timesteps), n_envs=args.n_envs)
