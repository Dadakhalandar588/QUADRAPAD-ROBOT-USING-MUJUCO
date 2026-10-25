import sys
import time
sys.path.append(".")

import mujoco.viewer
from envs.go1_env import Go1Env

env = Go1Env()
obs, info = env.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        action = env.action_space.sample()          # random action
        obs, r, term, trunc, info = env.step(action)
        viewer.sync()                               # redraw the window
        time.sleep(0.02)                            # one step = 0.02 s of robot time

        if term or trunc:
            print("episode ended, terminated:", term, "reward:", round(r, 3))
            obs, info = env.reset()