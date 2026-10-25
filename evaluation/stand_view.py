import time
import mujoco
import mujoco.viewer

XML_PATH = "models/unitree_go1/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

mujoco.mj_resetDataKeyframe(model, data, 0)
target = data.ctrl.copy()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)