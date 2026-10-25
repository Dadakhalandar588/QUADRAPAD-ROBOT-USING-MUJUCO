from pathlib import Path
import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


ROOT = Path(__file__).resolve().parents[1]

# Leg order used everywhere (feet arrays, contacts, heights).
LEGS = ("FR", "FL", "RR", "RL")

# Diagonal pairs, as indices into LEGS.
PAIR_A = (1, 2)   # FL + RR
PAIR_B = (0, 3)   # FR + RL
PAIRS = (PAIR_A, PAIR_B)

# Phases (one-hot flags in the observation):
#   0 = lift pair A   1 = land pair A
#   2 = lift pair B   3 = land pair B

# Default action scale (joint-target offset per unit action).
# This is a starting value. Check it with scripts/test_action_scale.py.
ACTION_SCALE_DEFAULT = 0.3

# Phase switching (event based, no timer).
LIFT_DONE_FRAC = 0.8      # lift is done when both swing feet reach 80% of clearance
SETTLE_LIN = 0.15         # m/s, body must be this calm after landing
SETTLE_ANG = 0.6          # rad/s
MAX_PHASE_S = 1.5         # seconds before a "too slow" penalty starts

# Fixed observation scaling (so inputs stay the same across all stages).
S_TILT = 0.5
S_ANG = 2.0
S_LIN = 1.0
S_JPOS = 0.5
S_JVEL = 5.0
S_FOOT = 0.05
S_TARGET = 1.0


def find_model():
    candidates = [
        ROOT / "models" / "unitree_go1" / "scene.xml",
        ROOT / "models" / "go1" / "scene.xml",
        ROOT / "scene.xml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Could not find the Go1 MuJoCo XML. "
        "Pass its location using xml_path=..."
    )


class Go1GaitEnv(gym.Env):
    """
    Curriculum environment for Go1.

    stage=1 : stand still.
    stage=2 : lift and land diagonal pairs in place (small clearance).
              Stage 3 is the same environment with a larger
              foot_clearance (use set_clearance).

    Observation (59), identical for every stage:
        roll, pitch                         2
        body angular velocity               3
        body linear velocity (body frame)   3
        joint offsets from home            12
        joint velocities                   12
        previous action                    12
        foot heights above standing level   4
        foot contacts                       4
        gait phase flags                    4  (all zero in stage 1)
        target velocity (vx, vy, yaw rate)  3  (zero for now)

    Action (12): joint-target offsets around the home pose.
    """

    metadata = {"render_modes": ["human"]}
    OBS_DIM = 59

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        episode_seconds=10.0,
        frame_skip=10,
        stage=1,
        foot_clearance=0.02,
        init_noise_level=0,
        random_start_pair=True,
        action_scale=ACTION_SCALE_DEFAULT,
        disturbance_probability=0.0,
        disturbance_strength=0.0,
        disturbance_interval=25,
        push_duration_s=0.1,
    ):
        super().__init__()

        model_path = (
            Path(xml_path).expanduser().resolve()
            if xml_path is not None
            else find_model()
        )
        if not model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)

        if self.model.nu != 12:
            raise ValueError(f"Expected 12 actuators, found {self.model.nu}")
        if self.model.nkey < 1:
            raise ValueError("The XML must contain a home keyframe.")

        self.xml_path = str(model_path)
        self.render_mode = render_mode
        self.frame_skip = int(frame_skip)
        self.control_dt = float(self.model.opt.timestep) * self.frame_skip
        self.max_episode_steps = max(1, int(episode_seconds / self.control_dt))
        self.max_phase_steps = max(1, round(MAX_PHASE_S / self.control_dt))

        # 1 = stand, 2 = diagonal lift in place (also used for Stage 3
        # with a larger clearance), 4 = move forward at a target speed.
        self.stage = int(stage)
        if self.stage not in (1, 2, 4):
            raise ValueError("stage must be 1, 2 or 4")

        self.foot_clearance = float(foot_clearance)
        self.init_noise_level = int(np.clip(init_noise_level, 0, 3))
        self.random_start_pair = bool(random_start_pair)
        self.action_scale = float(action_scale)

        self.target_velocity = np.zeros(3, dtype=np.float64)

        # ---- pushes (off by default; used in Stage 6) ----
        self.disturbance_probability = float(np.clip(disturbance_probability, 0, 1))
        self.disturbance_strength = max(0.0, float(disturbance_strength))
        self.disturbance_interval = max(1, int(disturbance_interval))
        self.push_hold_steps = max(1, round(float(push_duration_s) / self.control_dt))
        self.push_force = np.zeros(3)
        self.push_steps_left = 0
        self.next_disturbance_step = self.disturbance_interval
        self.disturbance_count = 0

        self.step_count = 0
        self.viewer = None

        self.home_qpos = self.model.key_qpos[0].copy()
        self.home_ctrl = self.model.key_ctrl[0].copy()
        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()
        self.previous_action = np.zeros(12, dtype=np.float32)

        # ---- joints ----
        free_jid = None
        joint_ids, qadr, dadr = [], [], []
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                if free_jid is None:
                    free_jid = jid
                continue
            joint_ids.append(jid)
            qadr.append(int(self.model.jnt_qposadr[jid]))
            dadr.append(int(self.model.jnt_dofadr[jid]))

        if free_jid is None:
            raise ValueError("No free joint (floating base) found.")
        if len(joint_ids) != 12:
            raise ValueError(f"Expected 12 leg joints, found {len(joint_ids)}.")

        self.base_body_id = int(self.model.jnt_bodyid[free_jid])
        self.joint_ids = joint_ids
        self.joint_qpos_addresses = np.array(qadr, dtype=int)
        self.joint_dof_addresses = np.array(dadr, dtype=int)

        # ---- foot geoms (required now) ----
        names = {
            "FR": ("FR", "FR_foot", "front_right_foot"),
            "FL": ("FL", "FL_foot", "front_left_foot"),
            "RR": ("RR", "RR_foot", "rear_right_foot"),
            "RL": ("RL", "RL_foot", "rear_left_foot"),
        }
        self.foot_ids = []
        for leg in LEGS:
            gid = -1
            for name in names[leg]:
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, name
                )
                if gid >= 0:
                    break
            if gid < 0:
                raise ValueError(
                    f"Foot geom for leg {leg} not found. "
                    f"Tried {names[leg]}. Check the geom names in your XML."
                )
            self.foot_ids.append(int(gid))

        # ---- measure the real standing reference ----
        self._settle_and_measure()

        # ---- phase state ----
        self.phase = 0
        self.phase_steps = 0
        self.cycle_count = 0
        self.touch_steps = [-1, -1]
        self.land_sync_paid = False
        self.last_land_gap = -1
        self.prev_heights = np.zeros(4)
        self.prev_foot_xy = np.zeros((4, 2))

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -10.0, 10.0, shape=(self.OBS_DIM,), dtype=np.float32
        )

    # ---------------------------------------------------------
    # SETUP HELPERS
    # ---------------------------------------------------------

    def _settle_and_measure(self):
        """Stand at the home pose for 1 s and record the reference."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.home_ctrl
        for _ in range(int(1.0 / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.foot_ref_z = self._foot_world()[:, 2].copy()
        self.foot_ref_body_xy = self._foot_body()[:, :2].copy()
        self.target_height = float(self.data.qpos[2])

        print(
            f"[Go1GaitEnv] control step {self.control_dt * 1000:.0f} ms, "
            f"settled body height {self.target_height:.3f} m, "
            f"foot ref heights {np.round(self.foot_ref_z, 3)}"
        )
        if not 0.20 < self.target_height < 0.35:
            print(
                "WARNING: settled height looks unusual. "
                "Check the home keyframe in your XML."
            )

    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------

    def set_clearance(self, metres):
        self.foot_clearance = float(metres)

    def set_init_noise_level(self, level):
        self.init_noise_level = int(np.clip(level, 0, 3))

    def set_target_velocity(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        self.target_velocity = np.array([vx, vy, yaw_rate], dtype=np.float64)

    def configure_disturbances(
        self, probability=0.0, strength=0.0, interval=25, duration_s=0.1
    ):
        self.disturbance_probability = float(np.clip(probability, 0, 1))
        self.disturbance_strength = max(0.0, float(strength))
        self.disturbance_interval = max(1, int(interval))
        self.push_hold_steps = max(1, round(float(duration_s) / self.control_dt))

    def _limits(self):
        return {
            0: (0.000, 0.000, 0.000),
            1: (0.005, 0.035, 0.025),
            2: (0.010, 0.070, 0.050),
            3: (0.015, 0.105, 0.075),
        }[self.init_noise_level]

    # ---------------------------------------------------------
    # GEOMETRY
    # ---------------------------------------------------------

    @staticmethod
    def _quat(roll, pitch):
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        return np.array([cr * cp, sr * cp, cr * sp, -sr * sp], dtype=np.float64)

    def _roll_pitch(self):
        w, x, y, z = self.data.qpos[3:7]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        s = 2 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2, s) if abs(s) >= 1 else math.asin(s)
        return roll, pitch

    def _rot(self):
        return self.data.xmat[self.base_body_id].reshape(3, 3)

    def _foot_world(self):
        return np.array([self.data.geom_xpos[g] for g in self.foot_ids])

    def _foot_body(self):
        """Foot positions in the robot's own frame (x = nose direction)."""
        rel = self._foot_world() - self.data.xpos[self.base_body_id]
        return rel @ self._rot()

    def _foot_heights(self):
        return self._foot_world()[:, 2] - self.foot_ref_z

    def _contacts(self):
        contacts = np.zeros(4, dtype=np.float32)
        foot_set = set(self.foot_ids)

        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for j, fid in enumerate(self.foot_ids):
                if g1 == fid or g2 == fid:
                    other = g2 if g1 == fid else g1
                    if other not in foot_set:
                        contacts[j] = 1.0
        return contacts

    def _phase_flags(self):
        flags = np.zeros(4, dtype=np.float32)
        if self.stage in (2, 4):
            flags[self.phase] = 1.0
        return flags

    # ---------------------------------------------------------
    # OBSERVATION
    # ---------------------------------------------------------

    def _obs(self):
        roll, pitch = self._roll_pitch()
        lin_body = self._rot().T @ self.data.qvel[0:3]
        ang = self.data.qvel[3:6]

        joint_offsets = (
            self.data.qpos[self.joint_qpos_addresses]
            - self.home_qpos[self.joint_qpos_addresses]
        )
        joint_vel = self.data.qvel[self.joint_dof_addresses]

        obs = np.concatenate(
            [
                np.array([roll, pitch]) / S_TILT,
                ang / S_ANG,
                lin_body / S_LIN,
                joint_offsets / S_JPOS,
                joint_vel / S_JVEL,
                self.previous_action,
                self._foot_heights() / S_FOOT,
                self._contacts(),
                self._phase_flags(),
                self.target_velocity / S_TARGET,
            ]
        ).astype(np.float32)

        if obs.shape != (self.OBS_DIM,):
            raise RuntimeError(f"Expected {self.OBS_DIM} observations, got {obs.shape}")

        return np.clip(obs, -10.0, 10.0)

    def _fallen(self):
        roll, pitch = self._roll_pitch()
        height = float(self.data.qpos[2])
        return (
            height < 0.16
            or height > 0.48
            or abs(roll) > 0.85
            or abs(pitch) > 0.85
        )

    # ---------------------------------------------------------
    # PUSHES
    # ---------------------------------------------------------

    def _maybe_start_push(self):
        if (
            self.push_steps_left > 0
            or self.disturbance_strength <= 0.0
            or self.disturbance_probability <= 0.0
            or self.step_count < self.next_disturbance_step
        ):
            return False

        self.next_disturbance_step = self.step_count + self.disturbance_interval

        if self.np_random.random() >= self.disturbance_probability:
            return False

        direction = self.np_random.uniform(0.0, 2.0 * math.pi)
        magnitude = self.np_random.uniform(0.5, 1.0) * self.disturbance_strength
        self.push_force = magnitude * np.array(
            [math.cos(direction), math.sin(direction), 0.0]
        )
        self.push_steps_left = self.push_hold_steps
        self.disturbance_count += 1
        return True

    # ---------------------------------------------------------
    # PHASE LOGIC
    # ---------------------------------------------------------

    def _advance_phase(self):
        self.phase = (self.phase + 1) % 4
        self.phase_steps = 0
        self.touch_steps = [-1, -1]
        self.land_sync_paid = False

    def _gait_terms(self, heights, contacts, foot_body, lin_body, ang):
        """Reward terms for Stage 2, then phase switching by events."""
        pair = self.phase // 2
        swing = list(PAIRS[pair])
        stance = list(PAIRS[1 - pair])
        lift_phase = (self.phase % 2 == 0)
        c = self.foot_clearance

        sw_h, st_h = heights[swing], heights[stance]
        sw_c, st_c = contacts[swing], contacts[stance]

        terms = {}

        # Stance pair stays down and does not slide.
        terms["stance_contact"] = 1.0 * float(np.mean(st_c))
        terms["stance_lift"] = -10.0 * float(
            np.mean(np.clip(st_h, 0.0, 0.05))
        )
        if self.stage == 4:
            # Walking: feet must be free to move in the body frame, and
            # planted feet move backwards under a moving body. So only
            # penalise a planted foot that slides over the GROUND.
            foot_xy = self._foot_world()[:, :2]
            slip = (
                np.linalg.norm(foot_xy - self.prev_foot_xy, axis=1)
                / self.control_dt
            )
            terms["stance_slip"] = -2.0 * float(np.mean(slip[stance] * st_c))
        else:
            drift = np.linalg.norm(
                foot_body[:, :2] - self.foot_ref_body_xy, axis=1
            )
            terms["swing_drift"] = -8.0 * float(np.mean(drift[swing]))
            terms["stance_drift"] = -3.0 * float(np.mean(drift[stance]))

        if lift_phase:
            # Using the LOWER swing foot forces both to rise together.
            terms["lift"] = 3.0 * float(np.clip(sw_h.min(), 0.0, c) / c)
            terms["lift_sync"] = -1.0 * float(
                min(abs(sw_h[0] - sw_h[1]) / c, 2.0)
            )
        else:
            terms["descend"] = 1.0 * (
                1.0 - float(np.clip(np.mean(sw_h) / c, 0.0, 1.0))
            )
            terms["land_asym"] = -0.5 * abs(float(sw_c[0] - sw_c[1]))
            vz = (sw_h - self.prev_heights[swing]) / self.control_dt
            terms["hard_land"] = -0.5 * float(
                np.mean(np.clip(-vz - 0.5, 0.0, 3.0))
            )

        # Too slow in this phase.
        self.phase_steps += 1
        if self.phase_steps > self.max_phase_steps:
            terms["too_slow"] = -0.2

        # ---- phase switching (events, no timer) ----
        bonus = 0.0
        event = None
        body_speed = float(np.linalg.norm(lin_body))
        ang_speed = float(np.linalg.norm(ang))

        if lift_phase:
            if sw_h.min() >= LIFT_DONE_FRAC * c:
                bonus += 3.0
                event = "lifted"
                self._advance_phase()
        else:
            for k, j in enumerate(swing):
                if contacts[j] > 0 and self.touch_steps[k] < 0:
                    self.touch_steps[k] = self.step_count

            if min(self.touch_steps) >= 0 and not self.land_sync_paid:
                gap = abs(self.touch_steps[0] - self.touch_steps[1])
                self.last_land_gap = int(gap)
                terms["land_sync"] = 1.0 * math.exp(-gap / 2.0)
                self.land_sync_paid = True

            # In place, a landing also needs a calm body. While walking
            # the body is always moving, so touching down is enough.
            calm = self.stage == 4 or (
                body_speed < SETTLE_LIN and ang_speed < SETTLE_ANG
            )

            if sw_c.min() >= 1.0 and calm:
                bonus += 5.0
                event = "landed"
                if self.phase == 3:
                    self.cycle_count += 1
                self._advance_phase()

        return float(sum(terms.values())) + bonus, event

    # ---------------------------------------------------------
    # RESET
    # ---------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.data.xfrc_applied[:] = 0.0
        self.push_steps_left = 0
        self.push_force = np.zeros(3)

        height_noise, tilt_noise, joint_noise = self._limits()

        self.data.qpos[2] += self.np_random.uniform(-height_noise, height_noise)
        roll = self.np_random.uniform(-tilt_noise, tilt_noise)
        pitch = self.np_random.uniform(-tilt_noise, tilt_noise)
        self.data.qpos[3:7] = self._quat(roll, pitch)

        if joint_noise > 0.0:
            for jid, qadr in zip(self.joint_ids, self.joint_qpos_addresses):
                proposed = self.data.qpos[qadr] + self.np_random.uniform(
                    -joint_noise, joint_noise
                )
                if self.model.jnt_limited[jid]:
                    lo, hi = self.model.jnt_range[jid]
                    proposed = np.clip(proposed, lo + 1e-3, hi - 1e-3)
                self.data.qpos[qadr] = proposed

        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.home_ctrl
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.previous_action[:] = 0.0
        self.disturbance_count = 0
        self.next_disturbance_step = self.disturbance_interval

        # Phase state: start with a random pair so there is no side bias.
        start_pair = (
            int(self.np_random.integers(0, 2)) if self.random_start_pair else 0
        )
        self.phase = 2 * start_pair
        self.phase_steps = 0
        self.cycle_count = 0
        self.touch_steps = [-1, -1]
        self.land_sync_paid = False
        self.last_land_gap = -1
        self.prev_heights = self._foot_heights().copy()
        self.prev_foot_xy = self._foot_world()[:, :2].copy()

        return self._obs(), {"stage": self.stage, "phase": self.phase}

    # ---------------------------------------------------------
    # STEP
    # ---------------------------------------------------------

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        ctrl = np.clip(
            self.home_ctrl + self.action_scale * action,
            self.ctrl_lo,
            self.ctrl_hi,
        )
        self.data.ctrl[:] = ctrl

        # ---- push held for several control steps ----
        disturbance_applied = self._maybe_start_push()
        if self.push_steps_left > 0:
            self.data.xfrc_applied[self.base_body_id, :3] = self.push_force
            self.push_steps_left -= 1
        else:
            self.data.xfrc_applied[self.base_body_id, :] = 0.0

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        if self.push_steps_left == 0:
            self.data.xfrc_applied[:] = 0.0

        self.step_count += 1

        # ---- state for reward ----
        roll, pitch = self._roll_pitch()
        height = float(self.data.qpos[2])
        contacts = self._contacts()
        heights = self._foot_heights()
        foot_body = self._foot_body()
        lin_body = self._rot().T @ self.data.qvel[0:3]
        ang = self.data.qvel[3:6].copy()

        joint_offsets = (
            self.data.qpos[self.joint_qpos_addresses]
            - self.home_qpos[self.joint_qpos_addresses]
        )
        joint_vel = self.data.qvel[self.joint_dof_addresses]

        tv = self.target_velocity
        vel_err = float(
            np.linalg.norm([lin_body[0] - tv[0], lin_body[1] - tv[1], lin_body[2]])
        )
        ang_err = float(np.linalg.norm([ang[0], ang[1], ang[2] - tv[2]]))

        upright = math.exp(-2.0 * (roll**2 + pitch**2))
        sigma = 0.06 if self.stage == 1 else 0.08
        height_score = math.exp(-(((height - self.target_height) / sigma) ** 2))

        action_change = float(np.mean((action - self.previous_action) ** 2))
        action_mag = float(np.mean(action**2))
        joint_vel_sq = float(np.sum(joint_vel**2))
        offsets_sq = float(np.mean(joint_offsets**2))

        event = None

        if self.stage == 1:
            reward = (
                0.5
                + 1.5 * upright
                + 1.0 * height_score
                + 0.35 * float(np.mean(contacts))
                - 0.08 * vel_err
                - 0.04 * ang_err
                - 0.03 * offsets_sq
                - 0.005 * joint_vel_sq
                - 0.30 * action_change
                - 0.05 * action_mag
            )
        else:
            gait_reward, event = self._gait_terms(
                heights, contacts, foot_body, lin_body, ang
            )
            reward = (
                0.5
                + 1.5 * upright
                + 0.8 * height_score
                - 0.08 * vel_err
                - 0.04 * ang_err
                - 0.01 * offsets_sq
                - 0.002 * joint_vel_sq
                - 0.10 * action_change
                - 0.02 * action_mag
                + gait_reward
            )

            if self.stage == 4:
                # Reward matching the target forward/sideways speed.
                # sigma = 0.1 m/s is tight enough that standing still
                # earns clearly less than actually walking.
                err_xy = math.hypot(
                    lin_body[0] - tv[0], lin_body[1] - tv[1]
                )
                reward += 4.0 * math.exp(-((err_xy / 0.1) ** 2))

        self.previous_action = action.copy()
        self.prev_heights = heights.copy()
        self.prev_foot_xy = self._foot_world()[:, :2].copy()

        fallen = self._fallen()
        terminated = bool(fallen)
        truncated = bool(self.step_count >= self.max_episode_steps and not terminated)

        if fallen:
            reward -= 5.0

        obs = self._obs()

        info = {
            "stage": self.stage,
            "phase": self.phase,
            "phase_event": event,
            "cycles": self.cycle_count,
            "last_land_gap_steps": self.last_land_gap,
            "height": height,
            "body_vx": float(lin_body[0]),
            "body_vy": float(lin_body[1]),
            "target_vx": float(tv[0]),
            "roll": roll,
            "pitch": pitch,
            "foot_heights": heights.tolist(),
            "foot_contacts": contacts.tolist(),
            "fallen": fallen,
            "mean_abs_joint_vel": float(np.mean(np.abs(joint_vel))),
            "disturbance_applied": disturbance_applied,
            "disturbance_count": self.disturbance_count,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    # ---------------------------------------------------------
    # RENDER / CLOSE
    # ---------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return None
        if self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None