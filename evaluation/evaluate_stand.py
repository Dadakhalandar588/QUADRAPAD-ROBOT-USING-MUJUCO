from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.go1_env import Go1StandEnv


MODEL_PATH = (
    ROOT
    / "checkpoints"
    / "stage1_standing"
    / "go1_stand_final.zip"
)

EPISODES = 5


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    # No graphical viewer: avoids viewer shutdown issues.
    env = Go1StandEnv(render_mode=None)

    model = PPO.load(
        str(MODEL_PATH),
        env=env,
        device="cpu",
    )

    print("Loaded model:", MODEL_PATH)
    print("Evaluation device: CPU")
    print("Starting headless evaluation...\n")

    episode_rewards = []
    episode_lengths = []
    fall_count = 0

    try:
        for episode in range(1, EPISODES + 1):
            obs, info = env.reset(seed=episode)

            total_reward = 0.0
            steps = 0

            while True:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                if terminated or truncated:
                    break

            fallen = bool(info.get("fallen", False))

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            fall_count += int(fallen)

            print(
                f"Episode {episode}: "
                f"steps={steps}, "
                f"reward={total_reward:.2f}, "
                f"fallen={fallen}, "
                f"height={info['height']:.3f} m, "
                f"roll={info['roll']:.3f}, "
                f"pitch={info['pitch']:.3f}"
            )

        print("\n===== EVALUATION SUMMARY =====")
        print(
            "Mean reward:",
            f"{np.mean(episode_rewards):.2f}",
        )
        print(
            "Mean episode length:",
            f"{np.mean(episode_lengths):.1f}",
        )
        print(
            f"Falls: {fall_count}/{EPISODES}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    main()
