"""
Watch the Stage 4 (walk forward) policy.

    python scripts/view_stage4.py                 # final model, 0.3 m/s
    python scripts/view_stage4.py --level 10      # the 0.1 m/s checkpoint
    python scripts/view_stage4.py --level 20      # the 0.2 m/s checkpoint
    python scripts/view_stage4.py --vx 0.15       # ask for a different speed
    python scripts/view_stage4.py --pace 0.5      # slow motion

The camera follows the robot. Close the viewer window to stop.
"""
from pathlib import Path
import argparse
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from envs.go1_env import Go1GaitEnv
except ModuleNotFoundError:
    from envs.go1_env import Go1GaitEnv

STAGE4_DIR = ROOT / "checkpoints" / "curriculum" / "stage4_forward"

# level (cm/s) -> (model file, target speed in m/s)
LEVELS = {
    10: ("go1_stage4_v10.zip", 0.1),
    20: ("go1_stage4_v20.zip", 0.2),
    30: ("go1_stage4_v30.zip", 0.3),
}
FINAL = ("go1_stage4_forward.zip", 0.3)

CLEARANCE = 0.045


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, choices=(10, 20, 30), default=None)
    parser.add_argument("--model", default=None, help="optional .zip path")
    parser.add_argument("--vx", type=float, default=None,
                        help="target forward speed in m/s")
    parser.add_argument("--clearance", type=float, default=CLEARANCE)
    parser.add_argument("--noise", type=int, default=0)
    parser.add_argument("--pace", type=float, default=1.0,
                        help="1.0 = real time, 0 = as fast as possible")
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        target = 0.3
    elif args.level:
        name, target = LEVELS[args.level]
        model_path = STAGE4_DIR / name
    else:
        name, target = FINAL
        model_path = STAGE4_DIR / name

    if args.vx is not None:
        target = args.vx

    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    env = Go1GaitEnv(
        stage=4,
        foot_clearance=args.clearance,
        init_noise_level=args.noise,
    )
    env.set_target_velocity(vx=target)

    model = PPO.load(str(model_path), device="cpu")

    print(f"\nModel:        {model_path}")
    print(f"Target speed: {target:.2f} m/s")
    print(f"Clearance:    {args.clearance * 1000:.0f} mm")
    print("Close the viewer window to stop.\n")

    obs, _ = env.reset()
    start_xy = env.data.qpos[0:2].copy()

    episode = 1
    steps = 0
    ep_reward = 0.0
    vx_list, vy_list, joint_vel = [], [], []
    peak_lift = 0.0

    summary = {"falls": 0, "speed": [], "dist": [], "cycles": []}

    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = env.base_body_id
            viewer.cam.distance = 1.8
            viewer.cam.elevation = -20

            while viewer.is_running():
                t0 = time.time()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                ep_reward += float(reward)
                steps += 1
                vx_list.append(info["body_vx"])
                vy_list.append(info["body_vy"])
                joint_vel.append(info["mean_abs_joint_vel"])
                peak_lift = max(peak_lift, max(info["foot_heights"]))
                viewer.sync()

                if terminated or truncated:
                    dist = float(
                        np.linalg.norm(env.data.qpos[0:2] - start_xy)
                    )
                    mean_vx = float(np.mean(vx_list))

                    summary["falls"] += int(terminated)
                    summary["speed"].append(mean_vx)
                    summary["dist"].append(dist)
                    summary["cycles"].append(info["cycles"])

                    print(
                        f"Episode {episode}: steps={steps}, "
                        f"speed={mean_vx:.2f} (target {target:.2f}) m/s, "
                        f"sideways={np.mean(np.abs(vy_list)):.2f} m/s, "
                        f"distance={dist:.2f} m, "
                        f"cycles={info['cycles']}, "
                        f"peak lift={peak_lift * 100:.1f} cm, "
                        f"gap={info['last_land_gap_steps']} steps, "
                        f"{'FELL' if terminated else 'stood'}"
                    )

                    episode += 1
                    steps = 0
                    ep_reward = 0.0
                    vx_list, vy_list, joint_vel = [], [], []
                    peak_lift = 0.0
                    obs, _ = env.reset()
                    start_xy = env.data.qpos[0:2].copy()

                if args.pace > 0:
                    wait = env.control_dt / args.pace - (time.time() - t0)
                    if wait > 0:
                        time.sleep(wait)

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

        if summary["speed"]:
            print(
                f"\nSummary over {len(summary['speed'])} episodes: "
                f"{summary['falls']} falls, "
                f"mean speed {np.mean(summary['speed']):.2f} m/s "
                f"(target {target:.2f}), "
                f"mean distance {np.mean(summary['dist']):.2f} m, "
                f"mean cycles {np.mean(summary['cycles']):.1f}"
            )
        print("Viewer closed.")


if __name__ == "__main__":
    main()