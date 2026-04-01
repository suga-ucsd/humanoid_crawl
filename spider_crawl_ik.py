import mujoco
import mujoco.viewer
import numpy as np
import time

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"


# -----------------------------
# Helper: joint map
# -----------------------------
def get_joint_map(model):
    joint_map = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_map[name] = i
    return joint_map


# -----------------------------
# Damped pseudo-inverse
# -----------------------------
def damped_pinv(J, damping=1e-1):
    JT = J.T
    return JT @ np.linalg.inv(J @ JT + damping**2 * np.eye(J.shape[0]))


# -----------------------------
# IK for one limb
# -----------------------------
def ik_limb(model, data, body_name, joint_ids, target_pos, steps=10):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)

    for _ in range(steps):
        mujoco.mj_forward(model, data)

        current_pos = data.xpos[body_id]
        error = target_pos - current_pos

        if np.linalg.norm(error) < 1e-3:
            break

        # Jacobian
        Jp = np.zeros((3, model.nv))
        Jr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, Jp, Jr, body_id)

        # Extract limb DOFs
        dof_ids = [model.jnt_dofadr[j] for j in joint_ids]
        J_limb = Jp[:, dof_ids]

        # IK solve
        J_pinv = damped_pinv(J_limb)
        dq = J_pinv @ error

        # Apply with scaling (IMPORTANT for stability)
        for i, j in enumerate(joint_ids):
            qpos_adr = model.jnt_qposadr[j]
            data.qpos[qpos_adr] += 0.05 * dq[i]

    return data.qpos.copy()


# -----------------------------
# Spider gait (belly-up)
# -----------------------------
def spider_targets(t):
    step_height = 0.05
    step_length = 0.15

    return {
        "left_hand":  np.array([ 0.3,  0.25, -0.3 + step_height*np.sin(t)]),
        "right_hand": np.array([ 0.3, -0.25, -0.3 + step_height*np.sin(t + np.pi)]),
        "left_foot":  np.array([-0.3,  0.25, -0.3 + step_height*np.sin(t + np.pi)]),
        "right_foot": np.array([-0.3, -0.25, -0.3 + step_height*np.sin(t)]),
    }


# -----------------------------
# Initialize stable pose
# -----------------------------
def initialize_pose(model, data, joint_map):
    mujoco.mj_resetData(model, data)

    # Lift base above ground (VERY IMPORTANT)
    data.qpos[2] = 0.5

    # Slight bend for stability
    data.qpos[model.jnt_qposadr[joint_map["left_knee_joint"]]] = 0.5
    data.qpos[model.jnt_qposadr[joint_map["right_knee_joint"]]] = 0.5

    data.qpos[model.jnt_qposadr[joint_map["left_elbow_joint"]]] = 0.5
    data.qpos[model.jnt_qposadr[joint_map["right_elbow_joint"]]] = 0.5

    mujoco.mj_forward(model, data)


# -----------------------------
# MAIN
# -----------------------------
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    joint_map = get_joint_map(model)

    # Limb definitions
    left_leg = [
        joint_map["left_hip_pitch_joint"],
        joint_map["left_hip_roll_joint"],
        joint_map["left_hip_yaw_joint"],
        joint_map["left_knee_joint"],
    ]

    right_leg = [
        joint_map["right_hip_pitch_joint"],
        joint_map["right_hip_roll_joint"],
        joint_map["right_hip_yaw_joint"],
        joint_map["right_knee_joint"],
    ]

    left_arm = [
        joint_map["left_shoulder_pitch_joint"],
        joint_map["left_shoulder_roll_joint"],
        joint_map["left_shoulder_yaw_joint"],
        joint_map["left_elbow_joint"],
    ]

    right_arm = [
        joint_map["right_shoulder_pitch_joint"],
        joint_map["right_shoulder_roll_joint"],
        joint_map["right_shoulder_yaw_joint"],
        joint_map["right_elbow_joint"],
    ]

    initialize_pose(model, data, joint_map)

    t = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] Running spider crawl IK...")

        while viewer.is_running():
            step_start = time.time()

            targets = spider_targets(t)

            # IK solve per limb
            ik_limb(model, data, "left_wrist_yaw_link", left_arm, targets["left_hand"])
            ik_limb(model, data, "right_wrist_yaw_link", right_arm, targets["right_hand"])
            ik_limb(model, data, "left_ankle_roll_link", left_leg, targets["left_foot"])
            ik_limb(model, data, "right_ankle_roll_link", right_leg, targets["right_foot"])

            mujoco.mj_step(model, data)
            viewer.sync()

            t += 0.03

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
