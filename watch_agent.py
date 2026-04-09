"""
==========================================================================
 BipedalWalker-v3 — Watch & Record the Trained Agent
==========================================================================
 Loads the best PPO checkpoint and:
   1. Renders the agent walking in a live pygame window.
   2. Records an .mp4 video of the run and saves it to videos/.

 Usage
 -----
     python watch_agent.py                       # watch + record
     python watch_agent.py --episodes 5          # run 5 episodes
     python watch_agent.py --no-render           # record only (headless)
     python watch_agent.py --model models/final_model.zip

 License: MIT
==========================================================================
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv


# ──────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH: str = os.path.join("models", "best_model.zip")
VEC_NORMALIZE_PATH: str = os.path.join("models", "vec_normalize.pkl")
VIDEO_DIR: str = "videos"
DEFAULT_EPISODES: int = 3


# ──────────────────────────────────────────────────────────────
# Core: run the agent and (optionally) record video
# ──────────────────────────────────────────────────────────────
def watch_agent(
    model_path: str,
    num_episodes: int = DEFAULT_EPISODES,
    render: bool = True,
    record: bool = True,
) -> None:
    """
    Load a trained PPO model and run it in BipedalWalker-v3.

    Parameters
    ----------
    model_path : str
        Path to the saved .zip model checkpoint.
    num_episodes : int
        Number of episodes to play.
    render : bool
        Whether to open a live pygame rendering window.
    record : bool
        Whether to save an .mp4 video to VIDEO_DIR.
    """

    # ── 1. Load the trained model ────────────────────────────
    print(f"\n{'='*60}")
    print(f"  [+] Loading model from: {model_path}")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)

    # ── 2. Load VecNormalize statistics (if available) ───────
    # During training we normalised observations. We need the same
    # running-mean / running-variance statistics at inference time.
    vec_norm_available = os.path.exists(VEC_NORMALIZE_PATH)
    if vec_norm_available:
        print(f"  [*] Loading VecNormalize stats from: {VEC_NORMALIZE_PATH}")

    # ── 3. Build the environment ─────────────────────────────
    # RecordVideo requires render_mode="rgb_array".
    # Use "human" only when we want a live window WITHOUT recording.
    if record:
        render_mode = "rgb_array"
    else:
        render_mode = "human" if render else "rgb_array"

    # Base environment
    env = gym.make("BipedalWalker-v3", render_mode=render_mode)

    # Wrap with video recorder if requested
    if record:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        # DummyVecEnv auto-resets after each done, so RecordVideo sees
        # 2 "episodes" per real episode: the actual run (even) and a
        # 1-frame stub from the auto-reset (odd).  Record only the real ones.
        env = RecordVideo(
            env,
            video_folder=VIDEO_DIR,
            episode_trigger=lambda ep_id: ep_id % 2 == 0,
            name_prefix="bipedal-walker-ppo",
        )
        print(f"  [REC] Recording video to: {VIDEO_DIR}/\n")

    # If we have VecNormalize stats we need a VecEnv wrapper to
    # normalise observations at inference time.
    # Keep a reference to the base env so we can close it explicitly
    # (ensures RecordVideo finalises the last episode's video).
    base_env = env

    if vec_norm_available:
        # Wrap in DummyVecEnv so VecNormalize can be applied
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(VEC_NORMALIZE_PATH, vec_env)
        vec_env.training = False       # don't update running stats
        vec_env.norm_reward = False    # we don't need reward normalisation
        use_vec = True
    else:
        print("  [!] VecNormalize stats not found -- running without "
              "observation normalisation.\n")
        use_vec = False

    # ── 4. Run episodes ──────────────────────────────────────
    all_rewards: list[float] = []

    for ep in range(1, num_episodes + 1):
        if use_vec:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            # Deterministic action for a clean demo
            action, _ = model.predict(obs, deterministic=True)

            if use_vec:
                obs, reward, dones, infos = vec_env.step(action)
                done = dones[0]
                total_reward += reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

            steps += 1

        all_rewards.append(total_reward)
        print(f"  [>] Episode {ep}/{num_episodes}  |  "
              f"Reward: {total_reward:>8.2f}  |  Steps: {steps}")

    # ── 5. Summary ───────────────────────────────────────────
    mean_r = np.mean(all_rewards)
    std_r = np.std(all_rewards)

    print(f"\n{'='*60}")
    print(f"  [OK] Mean reward over {num_episodes} episodes: "
          f"{mean_r:.2f} +/- {std_r:.2f}")
    if record:
        print(f"  [SAVED] Videos saved to: {VIDEO_DIR}/")
    print(f"{'='*60}\n")

    # Cleanup — close the base env explicitly so RecordVideo
    # flushes buffers and finalises the last video file.
    if use_vec:
        vec_env.close()
    base_env.close()


# ──────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Watch and record the trained BipedalWalker PPO agent"
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to model .zip file (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--episodes", type=int, default=DEFAULT_EPISODES,
        help=f"Number of episodes to run (default: {DEFAULT_EPISODES})"
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable the live rendering window (headless recording)"
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="Disable video recording"
    )
    args = parser.parse_args()

    watch_agent(
        model_path=args.model,
        num_episodes=args.episodes,
        render=not args.no_render,
        record=not args.no_record,
    )
