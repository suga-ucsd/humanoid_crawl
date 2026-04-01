import mujoco
import mujoco.viewer
import numpy as np
import time

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "unitree_ros/robots/g1_description/g1_29dof_lock_waist_rev_1_0.xml"

# -----------------------------
# Load model & data
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Disable gravity for sandbox
model.opt.gravity[:] = 0.0

# Lock floating base
data.qpos[:7] = 0.0
data.qvel[:6] = 0.0

# -----------------------------
# Define crawling joints
# -----------------------------
# Legs: hip_pitch and knee
left_hip = model.jnt_qposadr[model.joint("left_hip_pitch_joint").id]
right_hip = model.jnt_qposadr[model.joint("right_hip_pitch_joint").id]
left_knee = model.jnt_qposadr[model.joint("left_knee_joint").id]
right_knee = model.jnt_qposadr[model.joint("right_knee_joint").id]

# Arms: shoulder_pitch and elbow
left_shoulder = model.jnt_qposadr[model.joint("left_shoulder_pitch_joint").id]
right_shoulder = model.jnt_qposadr[model.joint("right_shoulder_pitch_joint").id]
left_elbow = model.jnt_qposadr[model.joint("left_elbow_joint").id]
right_elbow = model.jnt_qposadr[model.joint("right_elbow_joint").id]

# -----------------------------
# Crawling parameters
# -----------------------------
amplitude_leg = 0.5     # hip swing
amplitude_knee = 1.0    # knee bend
amplitude_arm = 0.5
amplitude_elbow = 1.0
frequency = 0.5         # Hz

# -----------------------------
# Simulation loop
# -----------------------------
t0 = time.time()

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        t = time.time() - t0

        # Simple sinusoidal crawl pattern (legs & arms move opposite)
        data.qpos[left_hip] = amplitude_leg * np.sin(2*np.pi*frequency*t)
        data.qpos[right_hip] = amplitude_leg * np.sin(2*np.pi*frequency*t + np.pi)
        data.qpos[left_knee] = amplitude_knee * np.sin(2*np.pi*frequency*t + np.pi/2)
        data.qpos[right_knee] = amplitude_knee * np.sin(2*np.pi*frequency*t + 3*np.pi/2)

        data.qpos[left_shoulder] = amplitude_arm * np.sin(2*np.pi*frequency*t + np.pi)
        data.qpos[right_shoulder] = amplitude_arm * np.sin(2*np.pi*frequency*t)
        data.qpos[left_elbow] = amplitude_elbow * np.sin(2*np.pi*frequency*t + np.pi/2)
        data.qpos[right_elbow] = amplitude_elbow * np.sin(2*np.pi*frequency*t + 3*np.pi/2)

        mujoco.mj_step(model, data)
        viewer.sync()
