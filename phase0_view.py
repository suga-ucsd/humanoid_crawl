import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
import time


MODEL_PATH = "/home/suga/projects/lucky/unitree_ros/robots/g1_description/g1_29dof.xml"


def main():
    model_path = Path(MODEL_PATH).resolve()
    assert model_path.exists(), f"Model not found: {model_path}"

    print(f"[INFO] Loading model: {model_path}")

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print("[INFO] Model loaded successfully")
    print(f"[INFO] nq (positions): {model.nq}")
    print(f"[INFO] nv (velocities): {model.nv}")
    print(f"[INFO] nu (actuators): {model.nu}")
    for i in range(model.njnt):
        print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i))
    # Reset to default pose
    mujoco.mj_resetData(model, data)

    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] Simulation running... Ctrl+C to exit")

        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()

            # Step simulation
            mujoco.mj_step(model, data)

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
