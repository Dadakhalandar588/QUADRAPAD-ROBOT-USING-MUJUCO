import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = "models/unitree_go1/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
home = data.ctrl.copy()

def find(leg, part):
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name.startswith(leg) and part in name:
            return i
    raise ValueError(f"not found: {leg} {part}")

# trot: diagonal pairs together (FR+RL), then (FL+RR)
offsets = {"FR": 0.0, "RL": 0.0, "FL": 0.5, "RR": 0.5}
legs = []
for leg, offset in offsets.items():
    thigh, calf = find(leg, "thigh"), find(leg, "calf")
    legs.append((thigh, calf, offset))
    print(leg, "thigh", thigh, "calf", calf, "offset", offset)

A = 0.10 # thigh swing (rad)
B = 0.3       # calf lift (rad)
F = 5.0       # cycles per second
SWING = 0.5   # half the cycle in the air, half on the ground

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        target = home.copy()
        for thigh, calf, offset in legs:
            p = (F * data.time + offset) % 1.0

            if p < SWING:                       # swing: foot in the air
                u = p / SWING
                d = A * np.cos(np.pi * u)       # back -> front
                lift = B * np.sin(np.pi * u)    # up and down
            else:                               # stance: foot on the ground
                u = (p - SWING) / (1 - SWING)
                d = -A + 2 * A * u              # front -> b  ack   
                lift = 0.0

            target[thigh] = home[thigh] + d
            target[calf] = home[calf] - lift
        data.ctrl[:] = target

        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)