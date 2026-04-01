import mujoco, mujoco.viewer, numpy as np, time

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

# End-effector sites defined in the XML
EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]

# Torque limits from XML actuatorfrcrange
TORQUE_LIMITS = {
    "left_hip_pitch_joint": 88, "left_hip_roll_joint": 139,
    "left_hip_yaw_joint": 88, "left_knee_joint": 139,
    "left_ankle_pitch_joint": 50,
    "right_hip_pitch_joint": 88, "right_hip_roll_joint": 139,
    "right_hip_yaw_joint": 88, "right_knee_joint": 139,
    "right_ankle_pitch_joint": 50,
    "waist_yaw_joint": 88,
    "left_shoulder_pitch_joint": 25, "left_shoulder_roll_joint": 25,
    "left_shoulder_yaw_joint": 25, "left_elbow_joint": 25,
    "left_wrist_pitch_joint": 5,
    "right_shoulder_pitch_joint": 25, "right_shoulder_roll_joint": 25,
    "right_shoulder_yaw_joint": 25, "right_elbow_joint": 25,
    "right_wrist_pitch_joint": 5,
}

# ──────────────────────────────────────
# Setup
# ──────────────────────────────────────

def load_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data

def configure_physics(model):
    model.opt.timestep = 0.0005
    model.opt.iterations = 50
    model.opt.ls_iterations = 50
    model.opt.solver = 2

    for i in range(model.ngeom):
        model.geom_solref[i] = [0.02, 1.0]
        model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]

    for i in range(model.nv):
        model.dof_damping[i] = 2.0

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_friction[floor_id] = [1.5, 0.005, 0.001]
    for i in range(model.ngeom):
        if i != floor_id:
            model.geom_friction[i] = [1.5, 0.005, 0.001]

def get_joint_map(model):
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap

def load_pose(model, joint_map, filename="g1_pose.txt"):
    q = np.copy(model.qpos0)
    try:
        with open(filename) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in joint_map:
                    q[joint_map[parts[0]]] = float(parts[1])
    except FileNotFoundError:
        print("No pose file, using defaults.")
    return q

def set_belly_up(data, q_target, z=0.25):
    data.qpos[:] = q_target
    data.qpos[0:3] = [0, 0, z]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    q_target[0:3] = data.qpos[0:3]
    q_target[3:7] = data.qpos[3:7]
    data.qvel[:] = 0
    return q_target

# ──────────────────────────────────────
# PD Control
# ──────────────────────────────────────

def build_gains(model):
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 0.3
        kd[i] = kp[i] * 1.0
    return kp, kd, max_tau

def pd_control(model, data, q_target, kp, kd, max_tau):
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i][0]
        qa = model.jnt_qposadr[jnt_id]
        va = model.jnt_dofadr[jnt_id]
        tau = kp[i] * (q_target[qa] - data.qpos[qa]) + kd[i] * (-data.qvel[va])
        data.ctrl[i] = np.clip(tau, -max_tau[i], max_tau[i])

# ──────────────────────────────────────
# Inverse Kinematics
# ──────────────────────────────────────

def get_ee_positions(model, data):
    """Get current end-effector positions from site sensors."""
    positions = {}
    for name in EE_SITES:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        positions[name] = data.site_xpos[site_id].copy()
    return positions

def ik_solve(model, data, q_target, ee_targets, max_iters=50, step_size=0.5, tol=0.001):
    """
    Jacobian-based IK: given target positions for end-effectors,
    compute joint angles that reach them.

    Args:
        model, data: MuJoCo model/data
        q_target: current joint target (modified in place for actuated DOFs)
        ee_targets: dict {site_name: np.array([x,y,z])}
        max_iters: IK iterations
        step_size: gradient step size (smaller = more stable)
        tol: position error tolerance

    Returns:
        q_target with updated joint angles
    """
    # Work on a copy of data to not disturb the simulation
    d = mujoco.MjData(model)
    d.qpos[:] = q_target
    mujoco.mj_forward(model, d)

    nv = model.nv
    jacp = np.zeros((3, nv))  # position Jacobian

    for _ in range(max_iters):
        mujoco.mj_forward(model, d)
        total_err = 0.0

        dq = np.zeros(nv)
        for site_name, target_pos in ee_targets.items():
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            current_pos = d.site_xpos[site_id]
            err = target_pos - current_pos
            total_err += np.linalg.norm(err)

            # Compute Jacobian for this site
            jacp[:] = 0
            mujoco.mj_jacSite(model, d, jacp, None, site_id)

            # Damped least squares: dq += J^T (J J^T + λI)^-1 err
            lam = 0.1
            JJT = jacp @ jacp.T + lam * np.eye(3)
            dq += jacp.T @ np.linalg.solve(JJT, err)

        if total_err < tol:
            break

        # Apply delta, skip the 6 free-joint DOFs (keep base fixed for IK)
        d.qpos[7:] += step_size * dq[6:]
        # Clamp to joint limits
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
                qa = model.jnt_qposadr[i]
                lo, hi = model.jnt_range[i]
                if lo < hi:
                    d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)

    # Copy solved joint angles back (only actuated joints, not free joint)
    q_target[7:] = d.qpos[7:]
    return q_target

# ──────────────────────────────────────
# Main
# ──────────────────────────────────────

def main():
    model, data = load_model(MODEL_PATH)
    configure_physics(model)
    joint_map = get_joint_map(model)

    q_target = load_pose(model, joint_map)
    q_target = set_belly_up(data, q_target, z=0.25)
    mujoco.mj_forward(model, data)

    kp, kd, max_tau = build_gains(model)

    # Record initial EE positions from the loaded pose as IK targets
    ee_targets = get_ee_positions(model, data)
    print("End-effector targets:")
    for name, pos in ee_targets.items():
        print(f"  {name}: {pos}")

    RENDER_EVERY = 10

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            t = time.time()

            # IK: if you update ee_targets, this recomputes joint angles
            # (RL will modify ee_targets, IK solves, PD tracks)
            # q_target = ik_solve(model, data, q_target, ee_targets)

            pd_control(model, data, q_target, kp, kd, max_tau)
            mujoco.mj_step(model, data)
            step += 1

            if step % RENDER_EVERY == 0:
                viewer.sync()
                time.sleep(max(0, model.opt.timestep * RENDER_EVERY - (time.time() - t)))

if __name__ == "__main__":
    main()
