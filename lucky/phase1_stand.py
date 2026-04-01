import mujoco
import mujoco.viewer
import numpy as np
import time
from pathlib import Path

MODEL_PATH = "/home/suga/projects/lucky/unitree_ros/robots/g1_description/g1_29dof.xml"



# -----------------------------
# Helper: get joint indices
# -----------------------------
def get_joint_map(model):
    joint_map = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_map[name] = i
    return joint_map


# -----------------------------
# Define a crawling pose
# -----------------------------
def create_crawl_pose(model, joint_map, filename="g1_pose.txt"):
    """
    Load a saved joint pose from a text file and return qpos_target
    """
    # Start from default pose
    qpos_target = np.copy(model.key_qpos[0]) if model.nkey > 0 else np.copy(model.qpos0)

    # --- Read file ---
    try:
        with open(filename, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                joint_name, value_str = parts
                value = float(value_str)
                # apply if joint exists
                if joint_name in joint_map:
                    idx = joint_map[joint_name]
                    qpos_target[idx] = value
    except FileNotFoundError:
        print(f"[WARNING] Pose file '{filename}' not found. Using default pose.")

    return qpos_target

# -----------------------------
# PD control
# -----------------------------
def pd_control(model, data, q_target, Kp=400.0, Kd=20.0):
    for i in range(model.nu):
        # Get joint ID this actuator controls
        joint_id = model.actuator_trnid[i][0]
        # name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        # print(i, "->", name)
        # Get qpos index
        qpos_adr = model.jnt_qposadr[joint_id]

        # Get qvel index
        qvel_adr = model.jnt_dofadr[joint_id]

        # Errors
        pos_err = q_target[qpos_adr] - data.qpos[qpos_adr]
        vel_err = -data.qvel[qvel_adr]

        # Control
        data.ctrl[i] = Kp * pos_err + Kd * vel_err

# -----------------------------
# Main
# -----------------------------
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    joint_map = get_joint_map(model)

    print("\n[INFO] Joint names:")
    for k in joint_map:
        print(k)

    # Create target pose
    q_target = create_crawl_pose(model, joint_map)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n[INFO] Stabilizing crawl pose...")

        while viewer.is_running():
            step_start = time.time()

            # Apply PD control
            pd_control(model, data, q_target)

            # Step simulation
            mujoco.mj_step(model, data)

            viewer.sync()

            # Real-time sync
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
