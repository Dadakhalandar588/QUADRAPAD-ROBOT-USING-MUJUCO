import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces


class Go1Env(gym.Env):
    def __init__(self, xml_path="models/unitree_go1/scene.xml"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = 10        # 10 physics steps x 0.002 s = 0.02 s per action
        self.action_scale = 0.25    # how far an action of 1.0 moves the target (rad)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.home = self.data.ctrl.copy()                    # standing pose targets (12)

        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0]   # lowest allowed target
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1]   # highest allowed target

        self.command = np.array([0.5, 0.0, 0.0])             # forward, sideways, turn
        self.last_action = np.zeros(12)

        self.steps = 0
        self.max_steps = 1000                                # 1000 x 0.02 s = 20 s per episode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(45,), dtype=np.float32)

    def _gravity_in_body(self):
        quat = self.data.qpos[3:7]
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, quat)
        g = np.zeros(3)
        mujoco.mju_rotVecQuat(g, np.array([0.0, 0.0, -1.0]), inv)
        return g

    def _get_obs(self):
        obs = np.concatenate([
            self.data.qvel[3:6],                    # body angular velocity (3)
            self._gravity_in_body(),                # tilt (3)
            self.command,                           # what we want (3)
            self.data.qpos[7:] - self.home,         # joint angles vs home (12)
            self.data.qvel[6:],                     # joint velocities (12)
            self.last_action,                       # previous action (12)
        ])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.last_action[:] = 0.0
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        # 1. ACT: action -> target angles -> motors -> physics
        action = np.clip(action, -1.0, 1.0)
        ctrl = np.clip(self.home + self.action_scale * action,
                       self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[:] = ctrl
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        # 2. SCORE: reward
        vx = self.data.qvel[0]                      # forward speed
        vy = self.data.qvel[1]                      # sideways speed
        g = self._gravity_in_body()                 # tilt

        reward = (
            2.0 * np.exp(-4.0 * (vx - self.command[0]) ** 2)    # match target speed
            - 0.5 * vy ** 2                                     # don't drift sideways
            - 0.5 * (1.0 + g[2])                                # stay level
            - 0.05 * np.sum((action - self.last_action) ** 2)   # smooth actions
            + 0.1                                               # stay alive
        )
        self.last_action = action.copy()            # after reward: it needs the OLD value

        # 3. END CHECK
        terminated = bool(g[2] > -0.5 or self.data.qpos[2] < 0.15)   # fell
        truncated = self.steps >= self.max_steps                      # time ran out

        return self._get_obs(), float(reward), terminated, truncated, {}