"""
Go1 curriculum training with ONE policy.

Stage 1: stand still                      (trained from scratch)
Stage 2: diagonal lift and land in place  (continues the same network)

The network, observation (59) and action (12) never change between
stages. Only the reward and task change inside the environment.

Run:  python scripts/train_curriculum.py
"""
from pathlib import Path
import sys

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.go1_env import Go1GaitEnv


# ============================================================
# CONFIGURATION
# ============================================================

CKPT = ROOT / "checkpoints" / "curriculum"
S1_DIR = CKPT / "stage1_stand"
S2_DIR = CKPT / "stage2_diagonal_lift"
S3_DIR = CKPT / "stage3_high_lift"
S4_DIR = CKPT / "stage4_forward"
LOG_DIR = ROOT / "logs" / "curriculum"

for d in (S1_DIR, S2_DIR, S3_DIR, S4_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

S1_MODEL = S1_DIR / "go1_stage1_stand.zip"
S1_STATS = S1_DIR / "go1_stage1_stand_vecnormalize.pkl"
S2_MODEL = S2_DIR / "go1_stage2_diagonal_lift.zip"
S2_STATS = S2_DIR / "go1_stage2_diagonal_lift_vecnormalize.pkl"

N_ENVS = 4

STAGE1_TIMESTEPS = 1_000_000
STAGE2_TIMESTEPS = 1_500_000

# PPO
N_STEPS = 512                # n_steps * N_ENVS = 2048 samples per update
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0

STAGE1_LR = 3e-4
STAGE2_LR = 2e-4

# Starting exploration noise for a fresh network: std = exp(-1.5) ~ 0.22
LOG_STD_INIT = -1.5

# Lifting a foot is new behaviour, so exploration noise is raised again
# at the start of Stage 2 (std = exp(-1.0) ~ 0.37). Same network.
STAGE2_MIN_LOG_STD = -1.0

# Stage settings
STAGE1_NOISE_LEVEL = 0      # nominal start pose
STAGE2_NOISE_LEVEL = 1      # small random start variations
STAGE2_CLEARANCE = 0.02     # metres of foot lift

# Stage 3: same task, higher foot lift, raised in steps.
# Each step continues from the previous one (same network).
STAGE3_CLEARANCES = (0.03, 0.045, 0.06)   # metres
STAGE3_TIMESTEPS_PER_LEVEL = 600_000
STAGE3_LR = 1e-4
STAGE3_NOISE_LEVEL = 1

S3_MODEL = S3_DIR / "go1_stage3_high_lift.zip"
S3_STATS = S3_DIR / "go1_stage3_high_lift_vecnormalize.pkl"

# Stage 4: walk forward. Target speed rises level by level (m/s).
# Each level continues from the previous one (same network).
STAGE4_TARGETS = (0.1, 0.2, 0.3)
STAGE4_TIMESTEPS_PER_LEVEL = 800_000
STAGE4_CLEARANCE = 0.045
STAGE4_LR = 2e-4
STAGE4_NOISE_LEVEL = 1

# Forward walking is a new behaviour, so exploration noise is raised
# again at the start of Stage 4 (same network).
STAGE4_MIN_LOG_STD = -1.2


# ============================================================
# HELPERS
# ============================================================

def make_vec(stage, clearance, noise_level, stats_path=None, target_vx=0.0):
    def _make():
        env = Go1GaitEnv(
            stage=stage,
            foot_clearance=clearance,
            init_noise_level=noise_level,
            render_mode=None,
        )
        env.set_target_velocity(vx=target_vx)
        return Monitor(env)

    base = DummyVecEnv([_make for _ in range(N_ENVS)])

    if stats_path is None:
        # Observations are already scaled inside the env, so only the
        # reward is normalised. This keeps the policy inputs identical
        # across stages.
        return VecNormalize(
            base,
            norm_obs=False,
            norm_reward=True,
            gamma=GAMMA,
        )

    if not Path(stats_path).is_file():
        raise FileNotFoundError(f"Stats not found:\n{stats_path}")

    venv = VecNormalize.load(str(stats_path), base)
    venv.training = True
    venv.norm_reward = True
    return venv


def save_all(model, venv, model_path, stats_path):
    model.save(str(model_path.with_suffix("")))
    venv.save(str(stats_path))
    print(f"Saved: {model_path}")
    print(f"Saved: {stats_path}")


def checkpointer(save_dir, prefix):
    return CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=str(save_dir),
        name_prefix=prefix,
        save_vecnormalize=True,
    )


# ============================================================
# STAGE 1 - STAND STILL (from scratch)
# ============================================================

def train_stage1():
    print("\n" + "=" * 65)
    print("STAGE 1: STAND STILL (new policy)")
    print("=" * 65)

    venv = make_vec(1, 0.0, STAGE1_NOISE_LEVEL)

    try:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=STAGE1_LR,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=LOG_STD_INIT,
            ),
            tensorboard_log=str(LOG_DIR),
            verbose=1,
            seed=0,
        )

        model.learn(
            total_timesteps=STAGE1_TIMESTEPS,
            callback=checkpointer(S1_DIR, "stage1_checkpoint"),
            progress_bar=False,
        )

        save_all(model, venv, S1_MODEL, S1_STATS)
    finally:
        venv.close()

    print("\nStage 1 finished.")


# ============================================================
# STAGE 2 - DIAGONAL LIFT AND LAND (same network continues)
# ============================================================

def train_stage2():
    print("\n" + "=" * 65)
    print("STAGE 2: DIAGONAL LIFT AND LAND IN PLACE")
    print("=" * 65)

    if not S1_MODEL.is_file():
        raise FileNotFoundError(
            f"Stage 1 model is missing:\n{S1_MODEL}\nRun Stage 1 first."
        )

    venv = make_vec(2, STAGE2_CLEARANCE, STAGE2_NOISE_LEVEL, stats_path=S1_STATS)

    try:
        model = PPO.load(
            str(S1_MODEL),
            env=venv,
            device="auto",
            custom_objects={
                "learning_rate": STAGE2_LR,
                "lr_schedule": lambda _: STAGE2_LR,
            },
        )

        # Raise exploration noise if Stage 1 shrank it below the floor.
        with torch.no_grad():
            model.policy.log_std.clamp_(min=STAGE2_MIN_LOG_STD)

        model.learn(
            total_timesteps=STAGE2_TIMESTEPS,
            callback=checkpointer(S2_DIR, "stage2_checkpoint"),
            reset_num_timesteps=False,
            progress_bar=False,
        )

        save_all(model, venv, S2_MODEL, S2_STATS)
    finally:
        venv.close()

    print("\nStage 2 finished.")


# ============================================================
# STAGE 3 - HIGHER FOOT CLEARANCE (same network continues)
# ============================================================

def train_stage3():
    import shutil

    print("\n" + "=" * 65)
    print("STAGE 3: HIGHER FOOT CLEARANCE, BETTER COORDINATION")
    print("=" * 65)

    prev_model = S2_MODEL
    prev_stats = S2_STATS

    if not prev_model.is_file():
        raise FileNotFoundError(
            f"Stage 2 model is missing:\n{prev_model}\nRun Stage 2 first."
        )

    for clearance in STAGE3_CLEARANCES:
        mm = int(round(clearance * 1000))
        print(f"\nFoot clearance target: {mm} mm")

        venv = make_vec(
            2, clearance, STAGE3_NOISE_LEVEL, stats_path=prev_stats
        )

        try:
            model = PPO.load(
                str(prev_model),
                env=venv,
                device="auto",
                custom_objects={
                    "learning_rate": STAGE3_LR,
                    "lr_schedule": lambda _: STAGE3_LR,
                },
            )

            model.learn(
                total_timesteps=STAGE3_TIMESTEPS_PER_LEVEL,
                callback=checkpointer(S3_DIR, f"stage3_{mm}mm_checkpoint"),
                reset_num_timesteps=False,
                progress_bar=False,
            )

            model_path = S3_DIR / f"go1_stage3_{mm}mm.zip"
            stats_path = S3_DIR / f"go1_stage3_{mm}mm_vecnormalize.pkl"
            save_all(model, venv, model_path, stats_path)
        finally:
            venv.close()

        prev_model = model_path
        prev_stats = stats_path

    # The last clearance becomes the final Stage 3 model.
    shutil.copyfile(prev_model, S3_DIR / "go1_stage3_high_lift.zip")
    shutil.copyfile(
        prev_stats, S3_DIR / "go1_stage3_high_lift_vecnormalize.pkl"
    )

    print("\nStage 3 finished.")
    print(f"Final model: {S3_DIR / 'go1_stage3_high_lift.zip'}")


# ============================================================
# STAGE 4 - WALK FORWARD (same network continues)
# ============================================================

def train_stage4():
    import shutil

    print("\n" + "=" * 65)
    print("STAGE 4: MOVE FORWARD AT A TARGET SPEED")
    print("=" * 65)

    prev_model = S3_MODEL
    prev_stats = S3_STATS

    if not prev_model.is_file():
        raise FileNotFoundError(
            f"Stage 3 model is missing:\n{prev_model}\nRun Stage 3 first."
        )

    first_level = True

    for vx in STAGE4_TARGETS:
        cms = int(round(vx * 100))
        print(f"\nTarget forward speed: {vx:.2f} m/s")

        venv = make_vec(
            4,
            STAGE4_CLEARANCE,
            STAGE4_NOISE_LEVEL,
            stats_path=prev_stats,
            target_vx=vx,
        )

        try:
            model = PPO.load(
                str(prev_model),
                env=venv,
                device="auto",
                custom_objects={
                    "learning_rate": STAGE4_LR,
                    "lr_schedule": lambda _: STAGE4_LR,
                },
            )

            if first_level:
                with torch.no_grad():
                    model.policy.log_std.clamp_(min=STAGE4_MIN_LOG_STD)
                first_level = False

            model.learn(
                total_timesteps=STAGE4_TIMESTEPS_PER_LEVEL,
                callback=checkpointer(S4_DIR, f"stage4_v{cms}_checkpoint"),
                reset_num_timesteps=False,
                progress_bar=False,
            )

            model_path = S4_DIR / f"go1_stage4_v{cms}.zip"
            stats_path = S4_DIR / f"go1_stage4_v{cms}_vecnormalize.pkl"
            save_all(model, venv, model_path, stats_path)
        finally:
            venv.close()

        prev_model = model_path
        prev_stats = stats_path

    shutil.copyfile(prev_model, S4_DIR / "go1_stage4_forward.zip")
    shutil.copyfile(
        prev_stats, S4_DIR / "go1_stage4_forward_vecnormalize.pkl"
    )

    print("\nStage 4 finished.")
    print(f"Final model: {S4_DIR / 'go1_stage4_forward.zip'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("GO1 PPO CURRICULUM")
    print("=" * 65)
    print("\n1. Stage 1 - stand still (from scratch)")
    print("2. Stage 2 - diagonal lift and land (continues Stage 1)")
    print("3. Stage 3 - higher foot clearance (continues Stage 2)")
    print("4. Stage 4 - walk forward (continues Stage 3)")
    print("5. Stage 1, then 2, 3 and 4")

    choice = input("\nEnter 1, 2, 3, 4, or 5: ").strip()

    if choice == "1":
        train_stage1()
    elif choice == "2":
        train_stage2()
    elif choice == "3":
        train_stage3()
    elif choice == "4":
        train_stage4()
    elif choice == "5":
        train_stage1()
        train_stage2()
        train_stage3()
        train_stage4()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()