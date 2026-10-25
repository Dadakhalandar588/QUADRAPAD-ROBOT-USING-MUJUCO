# ---------- 1. IMPORTS ----------
import mujoco
import numpy as np

# ---------- 2. LOAD ----------
XML_PATH = "models/unitree_go1/scene.xml"   # check this with `ls models`

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ---------- 3. RESET + INSPECT ----------
mujoco.mj_resetDataKeyframe(model, data, 0)

print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

print("qpos joints:", data.qpos[7:])
print("qvel joints:", data.qvel[6:])
print("ctrl       :", data.ctrl)

# ---------- 4. COMPUTE ----------
spring_torque = 100 * (data.ctrl - data.qpos[7:])
print("spring torque:", np.round(spring_torque, 2))