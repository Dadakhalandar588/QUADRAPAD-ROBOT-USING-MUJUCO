"""
Go1 PPO training: Stage 1 (standing) + Stage 2A (random starts)
+ Stage 2B (push recovery), all in one file.

Run:  python scripts/train_all.py
Menu: 1 = Stage 1, 2 = Stage 2A, 3 = Stage 2B, 4 = everything in order.
"""
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.go1_env import Go1StandEnv


# ============================================================
# CONFIGURATION
# ============================================================

STAGE1_DIR = ROOT / "checkpoints" / "stage1_standing_v2"
STAGE2_DIR = ROOT / "checkpoints" / "stage2_robust_standing"
LOG_DIR = ROOT / "logs" / "go1_training"

for d in (STAGE1_DIR, STAGE2_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

STAGE1_MODEL = STAGE1_DIR / "go1_stand_final.zip"
STAGE1_STATS = STAGE1_DIR / "go1_stand_final_vecnormalize.pkl"

N_ENVS = 4

# ---- Timesteps ----
STAGE1_TIMESTEPS = 1_000_000
STAGE2A_TIMESTEPS_PER_LEVEL = 150_000
STAGE2B_TIMESTEPS_PER_STRENGTH = 150_000

# ---- PPO (Stage 1 trains from scratch) ----
STAGE1_LEARNING_RATE = 3e-4
N_STEPS = 512                 # n_steps * N_ENVS = 2048 samples per update
BATCH_SIZE = 256
N_EPOCHS = 10
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.0
LOG_STD_INIT = -1.5           # starting noise std ~ 0.22 instead of 1.0

# ---- Stage 2 fine-tuning learning rate ----
STAGE2_LEARNING_RATE = 1e-4

# ---- Stage 2B pushes ----
# Go1 weighs about 125 N, so these are roughly 8%, 16%, 24% of body weight.
DISTURBANCE_STRENGTHS = (10.0, 20.0, 30.0)   # Newtons
DISTURBANCE_PROBABILITY = 0.5                # chance per eligible step
DISTURBANCE_INTERVAL = 25                    # control steps between tries
PUSH_DURATION_S = 0.1                        # seconds each push is held


# ============================================================
# SHARED HELPERS
# ============================================================

def make_env_fn(stage, level, strength):
    """Return a function that builds one monitored env."""

    def _make():
        env = Go1StandEnv(
            stage=stage,
            render_mode=None,
            disturbance_probability=(
                DISTURBANCE_PROBABILITY if strength > 0 else 0.0
            ),
            disturbance_strength=strength,
            disturbance_interval=DISTURBANCE_INTERVAL,
            push_duration_s=PUSH_DURATION_S,
        )
        env.set_curriculum_level(level)
        return Monitor(env)

    return _make


def make_new_vec(stage, level, strength):
    """Fresh vector env with new normalisation stats (Stage 1)."""
    base = DummyVecEnv(
        [make_env_fn(stage, level, strength) for _ in range(N_ENVS)]
    )
    return VecNormalize(
        base,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        gamma=GAMMA,
    )


def make_loaded_vec(stage, level, strength, stats_path):
    """Vector env that continues from saved normalisation stats."""
    if not Path(stats_path).is_file():
        raise FileNotFoundError(
            f"Normalisation stats not found:\n{stats_path}"
        )

    base = DummyVecEnv(
        [make_env_fn(stage, level, strength) for _ in range(N_ENVS)]
    )
    venv = VecNormalize.load(str(stats_path), base)
    venv.training = True
    venv.norm_reward = True
    return venv


def load_model(model_path, venv):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    # In SB3, setting model.learning_rate after loading does NOT change
    # the optimiser, so the schedule is overridden at load time.
    return PPO.load(
        str(model_path),
        env=venv,
        device="auto",
        custom_objects={
            "learning_rate": STAGE2_LEARNING_RATE,
            "lr_schedule": lambda _: STAGE2_LEARNING_RATE,
        },
    )


def run_phase(
    model,
    venv,
    timesteps,
    save_dir,
    prefix,
    save_model,
    save_stats,
    reset_timesteps=False,
):
    checkpoint = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path=str(save_dir),
        name_prefix=prefix,
        save_vecnormalize=True,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint,
        reset_num_timesteps=reset_timesteps,
        progress_bar=False,
    )

    model.save(str(save_model))
    venv.save(str(save_stats))
    print(f"Saved: {save_model}.zip")
    print(f"Saved: {save_stats}")


# ============================================================
# STAGE 1 - NOMINAL STANDING (from scratch)
# ============================================================

def train_stage1():
    print("\n" + "=" * 65)
    print("STAGE 1: STANDING FROM SCRATCH")
    print("=" * 65)

    venv = make_new_vec(stage=1, level=0, strength=0.0)

    try:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=STAGE1_LEARNING_RATE,
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

        run_phase(
            model,
            venv,
            STAGE1_TIMESTEPS,
            STAGE1_DIR,
            "stage1_checkpoint",
            STAGE1_DIR / "go1_stand_final",
            STAGE1_STATS,
            reset_timesteps=True,
        )
    finally:
        venv.close()

    print("\nStage 1 finished.")


# ============================================================
# STAGE 2A - RANDOMISED INITIAL CONDITIONS
# ============================================================

def train_stage2a():
    print("\n" + "=" * 65)
    print("STAGE 2A: RANDOMISED INITIAL CONDITIONS")
    print("=" * 65)

    prev_model = STAGE1_MODEL
    prev_stats = STAGE1_STATS

    if not prev_model.is_file():
        raise FileNotFoundError(
            f"Stage 1 model is missing:\n{prev_model}\n"
            "Run Stage 1 first."
        )

    for level in (1, 2, 3):
        print(f"\nCurriculum level {level}")

        venv = make_loaded_vec(2, level, 0.0, prev_stats)

        try:
            model = load_model(prev_model, venv)

            save_model = STAGE2_DIR / f"go1_stage2a_level{level}"
            save_stats = (
                STAGE2_DIR / f"go1_stage2a_level{level}_vecnormalize.pkl"
            )

            run_phase(
                model,
                venv,
                STAGE2A_TIMESTEPS_PER_LEVEL,
                STAGE2_DIR,
                f"stage2a_level{level}_checkpoint",
                save_model,
                save_stats,
            )
        finally:
            venv.close()

        prev_model = Path(str(save_model) + ".zip")
        prev_stats = save_stats

    print("\nStage 2A finished.")


# ============================================================
# STAGE 2B - PUSH RECOVERY (strength ramps up)
# ============================================================

def train_stage2b():
    print("\n" + "=" * 65)
    print("STAGE 2B: PUSH RECOVERY")
    print("=" * 65)

    prev_model = STAGE2_DIR / "go1_stage2a_level3.zip"
    prev_stats = STAGE2_DIR / "go1_stage2a_level3_vecnormalize.pkl"

    if not prev_model.is_file():
        raise FileNotFoundError(
            f"Stage 2A model is missing:\n{prev_model}\n"
            "Run Stage 2A first."
        )

    for strength in DISTURBANCE_STRENGTHS:
        print(f"\nPush strength up to {strength:.0f} N")

        venv = make_loaded_vec(2, 3, strength, prev_stats)

        try:
            model = load_model(prev_model, venv)

            tag = f"go1_stage2b_{int(strength)}N"
            save_model = STAGE2_DIR / tag
            save_stats = STAGE2_DIR / f"{tag}_vecnormalize.pkl"

            run_phase(
                model,
                venv,
                STAGE2B_TIMESTEPS_PER_STRENGTH,
                STAGE2_DIR,
                f"stage2b_{int(strength)}N_checkpoint",
                save_model,
                save_stats,
            )
        finally:
            venv.close()

        prev_model = Path(str(save_model) + ".zip")
        prev_stats = save_stats

    # Last strength level becomes the final robust model.
    final_model = STAGE2_DIR / "go1_stage2b_robust_standing"
    final_stats = (
        STAGE2_DIR / "go1_stage2b_robust_standing_vecnormalize.pkl"
    )

    venv = make_loaded_vec(
        2, 3, DISTURBANCE_STRENGTHS[-1], prev_stats
    )
    try:
        model = load_model(prev_model, venv)
        model.save(str(final_model))
        venv.save(str(final_stats))
    finally:
        venv.close()

    print("\nStage 2B finished.")
    print(f"Final model: {final_model}.zip")
    print(f"Final stats: {final_stats}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("GO1 PPO - STANDING AND ROBUST STANDING")
    print("=" * 65)

    print("\nChoose a training phase:")
    print("1. Stage 1  - Standing from scratch")
    print("2. Stage 2A - Randomised initial conditions")
    print("3. Stage 2B - Push recovery")
    print("4. Run everything: Stage 1 -> 2A -> 2B")

    choice = input("\nEnter 1, 2, 3, or 4: ").strip()

    if choice == "1":
        train_stage1()
    elif choice == "2":
        train_stage2a()
    elif choice == "3":
        train_stage2b()
    elif choice == "4":
        train_stage1()
        train_stage2a()
        train_stage2b()
    else:
        print("Invalid choice. Please run the script again.")


if __name__ == "__main__":
    main()