# ---------- 1. IMPORTS ----------
import mujoco
import numpy as np

# ---------- 2. LOAD + RESET ----------
XML_PATH = "models/unitree_go1/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)

# ---------- 3. SAVE THE HOME TARGETS ----------
target = data.ctrl.copy()

# ---------- 4. HOW MANY STEPS = 2 SECONDS ----------
dt = model.opt.timestep
sim_time = 2.0
n_steps = int(sim_time / dt)
print("timestep =", dt, " steps =", n_steps)
print("start height =", round(data.qpos[2], 4))

# ---------- 5. THE LOOP ----------
for i in range(n_steps):
    data.ctrl[:] = target
    mujoco.mj_step(model, data)

    if i % 100 == 0:
        print(f"t = {data.time:.2f}s  height = {data.qpos[2]:.4f}")

# ---------- 6. RESULT ----------
print("final height      :", round(data.qpos[2], 4))
print("joint error (rad) :", np.round(data.qpos[7:] - target, 3))
print("joint velocities  :", np.round(data.qvel[6:], 3))