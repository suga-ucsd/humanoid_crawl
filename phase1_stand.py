import mujoco, mujoco.viewer, numpy as np, time

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# --- Stabilize physics ---
model.opt.timestep = 0.0005          # 0.5ms (default 2ms is too coarse)
model.opt.iterations = 50            # more solver iterations
model.opt.ls_iterations = 50         # line search iterations
model.opt.solver = 2                 # Newton solver (most stable)

# Set ground friction (sliding, torsional, rolling)
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
model.geom_friction[floor_id] = [2.0, 0.005, 0.001]

# Also add friction to all robot geoms so contacts grip
for i in range(model.ngeom):
    if i != floor_id:
        model.geom_friction[i] = [1.5, 0.005, 0.001]

# Soften all contacts to reduce bounce/jitter
for i in range(model.ngeom):
    model.geom_solref[i] = [0.02, 1.0]    # slower, damped contact
    model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]

# Add damping to every DOF
for i in range(model.nv):
    model.dof_damping[i] = 2.0

# Joint name -> qpos address
joint_map = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_map[name] = model.jnt_qposadr[i]

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

# Load target pose
q_target = np.copy(model.qpos0)
try:
    with open("g1_pose.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in joint_map:
                q_target[joint_map[parts[0]]] = float(parts[1])
except FileNotFoundError:
    print("No pose file, using defaults.")

# Belly-up, just above ground
data.qpos[:] = q_target
data.qpos[0:3] = [0, 0, 0.25]
data.qpos[3:7] = [0.707, 0, -0.707, 0]
q_target[0:3] = data.qpos[0:3]
q_target[3:7] = data.qpos[3:7]
data.qvel[:] = 0
mujoco.mj_forward(model, data)

# Per-actuator gains
max_tau = np.zeros(model.nu)
kp = np.zeros(model.nu)
kd = np.zeros(model.nu)

for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    frc = TORQUE_LIMITS.get(name, 25.0)
    max_tau[i] = frc
    kp[i] = frc * 0.3
    kd[i] = kp[i] * 1.0   # critically damped ratio

def pd_control():
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i][0]
        qa = model.jnt_qposadr[jnt_id]
        va = model.jnt_dofadr[jnt_id]
        tau = kp[i] * (q_target[qa] - data.qpos[qa]) + kd[i] * (-data.qvel[va])
        data.ctrl[i] = np.clip(tau, -max_tau[i], max_tau[i])

RENDER_EVERY = 10  # render every 10 steps (= every 5ms at 0.5ms timestep)

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t = time.time()
        pd_control()
        mujoco.mj_step(model, data)
        step += 1
        if step % RENDER_EVERY == 0:
            viewer.sync()
            elapsed = time.time() - t
            time.sleep(max(0, model.opt.timestep * RENDER_EVERY - elapsed))
