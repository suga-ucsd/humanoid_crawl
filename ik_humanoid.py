import mujoco
import numpy as np
import time

MODEL_PATH = "unitree_ros/robots/g1_description/g1_29dof_lock_waist_rev_1_0.xml"

# -----------------------------
# SITES (must exist in XML)
# -----------------------------
SITES = {
    "RH": "right_hand_site",
    "LH": "left_hand_site",
    "RF": "right_foot_site",
    "LF": "left_foot_site",
}

# -----------------------------
# JOINTS (your robot)
# -----------------------------
LIMB_JOINTS = {
    "RH": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
    ],
    "LH": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
    ],
    "RF": [
        "right_hip_yaw_joint",
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
    ],
    "LF": [
        "left_hip_yaw_joint",
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
    ],
}

TORSO_JOINTS = [
    "waist_yaw_joint",
]

# -----------------------------
# GAINS (safe defaults)
# -----------------------------
KP = 40
KD = 5

IK_ITERS = 5
DAMPING = 1e-3

STEP_HEIGHT = 0.02
STEP_LENGTH = 0.03
CYCLE_TIME = 8.0

# -----------------------------
# LOAD MODEL
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# -----------------------------
# INDEX HELPERS
# -----------------------------
def joint_qpos_ids(names):
    return [model.joint(n).qposadr for n in names]

def joint_dof_ids(names):
    return [model.joint(n).dofadr for n in names]

JOINT_QPOS_IDXS = {k: joint_qpos_ids(v) for k, v in LIMB_JOINTS.items()}
JOINT_DOF_IDXS  = {k: joint_dof_ids(v) for k, v in LIMB_JOINTS.items()}
TORSO_QPOS_IDXS = joint_qpos_ids(TORSO_JOINTS)

# -----------------------------
# UTILS
# -----------------------------
def get_site_pos(name):
    return data.site_xpos[model.site(name).id].copy()

def get_jacobian(site):
    site_id = model.site(site).id
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, Jp, Jr, site_id)
    return Jp

# -----------------------------
# STABLE IK (Damped Least Squares)
# -----------------------------
def ik_solve(limb, target_pos, q_target):
    dof_ids  = JOINT_DOF_IDXS[limb]
    qpos_ids = JOINT_QPOS_IDXS[limb]
    site     = SITES[limb]

    for _ in range(IK_ITERS):
        data.qpos[:] = q_target
        mujoco.mj_forward(model, data)

        current = get_site_pos(site)
        error = target_pos - current

        J = get_jacobian(site)
        J_limb = J[:, dof_ids]

        # Damped least squares
        JT = J_limb.T
        H = JT @ J_limb + DAMPING * np.eye(len(dof_ids))
        dq = np.linalg.solve(H, JT @ error)

        for i, jid in enumerate(qpos_ids):
            q_target[jid] += dq[i]

    return q_target

# -----------------------------
# GAIT
# -----------------------------
GAIT_ORDER = ["RH", "LF", "LH", "RF"]

def gait(t):
    phase_time = CYCLE_TIME / 4
    t_mod = t % CYCLE_TIME

    idx = int(t_mod / phase_time)
    phase = (t_mod % phase_time) / phase_time

    return GAIT_ORDER[idx], phase

# -----------------------------
# TRAJECTORY
# -----------------------------
def swing(p0, phase):
    direction = np.array([STEP_LENGTH, 0, 0])
    p1 = p0 + direction

    pos = (1 - phase) * p0 + phase * p1
    pos[2] += STEP_HEIGHT * np.sin(np.pi * phase)

    return pos

# -----------------------------
# INIT
# -----------------------------
mujoco.mj_forward(model, data)

support_pos = {k: get_site_pos(v) for k, v in SITES.items()}

start = time.time()

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    t = time.time() - start

    active, phase = gait(t)

    q_target = data.qpos.copy()

    for limb in SITES:

        if limb == active:
            target = swing(support_pos[limb], phase)

            if phase > 0.95:
                support_pos[limb] = target.copy()
        else:
            target = support_pos[limb]

        q_target = ik_solve(limb, target, q_target)

    # -----------------------------
    # PD CONTROL
    # -----------------------------
    qpos = data.qpos.copy()
    qvel = data.qvel.copy()

    tau = KP * (q_target - qpos) - KD * qvel

    # -----------------------------
    # TORSO STABILIZATION
    # -----------------------------
    for jid in TORSO_QPOS_IDXS:
        tau[jid] += -80 * qpos[jid] - 8 * qvel[jid]

    # Apply control
    data.ctrl[:] = tau[:model.nu]

    mujoco.mj_step(model, data)
