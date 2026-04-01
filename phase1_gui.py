import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import tkinter as tk
from pathlib import Path

MODEL_PATH = "/home/suga/projects/lucky/unitree_ros/robots/g1_description/g1_29dof.xml"



# -----------------------------
# GUI STATE (shared)
# -----------------------------
class Params:
    def __init__(self):
        self.Kp = 200.0
        self.Kd = 10.0
        self.A = 0.3
        self.omega = 2.0

        # Joint offsets (manual tuning)
        self.offsets = {
            "left_hip_pitch": 0.0,
            "right_hip_pitch": 0.0,
            "left_shoulder_pitch": 0.0,
            "right_shoulder_pitch": 0.0,
        }


params = Params()


# -----------------------------
# GUI
# -----------------------------
def start_gui():
    root = tk.Tk()
    root.title("MuJoCo Crawl Controller")

    def slider(label, attr, minv, maxv, row):
        tk.Label(root, text=label).grid(row=row, column=0)
        s = tk.Scale(root, from_=minv, to=maxv,
                     resolution=0.1, orient=tk.HORIZONTAL,
                     command=lambda v: setattr(params, attr, float(v)))
        s.set(getattr(params, attr))
        s.grid(row=row, column=1)

    slider("Kp", "Kp", 0, 500, 0)
    slider("Kd", "Kd", 0, 50, 1)
    slider("Amplitude", "A", 0, 1, 2)
    slider("Frequency", "omega", 0, 5, 3)

    # Joint offset sliders
    row = 4
    for name in params.offsets:
        tk.Label(root, text=name).grid(row=row, column=0)
        s = tk.Scale(root, from_=-1.5, to=1.5,
                     resolution=0.05, orient=tk.HORIZONTAL,
                     command=lambda v, n=name: params.offsets.__setitem__(n, float(v)))
        s.set(0)
        s.grid(row=row, column=1)
        row += 1

    root.mainloop()


# -----------------------------
# Joint mapping
# -----------------------------
def get_joint_map(model):
    joint_map = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_map[name] = i
    return joint_map


# -----------------------------
# Base pose
# -----------------------------
def create_base_pose(model, joint_map):
    qpos = np.copy(model.qpos0)

    def set_joint(name, val):
        if name in joint_map:
            qpos[model.jnt_qposadr[joint_map[name]]] = val

    set_joint("left_hip_pitch", -0.5)
    set_joint("right_hip_pitch", -0.5)
    set_joint("left_knee", 1.0)
    set_joint("right_knee", 1.0)

    set_joint("left_shoulder_pitch", 0.8)
    set_joint("right_shoulder_pitch", 0.8)
    set_joint("left_elbow", -1.2)
    set_joint("right_elbow", -1.2)

    return qpos


# -----------------------------
# Gait
# -----------------------------
def gait_target(base_pose, t, joint_map, model):
    q = base_pose.copy()

    A = params.A
    omega = params.omega

    def apply(name, phase):
        if name in joint_map:
            j = joint_map[name]
            idx = model.jnt_qposadr[j]
            q[idx] += A * np.sin(omega * t + phase)
            q[idx] += params.offsets.get(name, 0.0)

    apply("left_shoulder_pitch", 0)
    apply("right_hip_pitch", 0)

    apply("right_shoulder_pitch", np.pi)
    apply("left_hip_pitch", np.pi)

    return q


# -----------------------------
# PD control (correct mapping)
# -----------------------------
def pd_control(model, data, q_target):
    for i in range(model.nu):
        joint_id = model.actuator_trnid[i][0]

        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]

        pos_err = q_target[qpos_adr] - data.qpos[qpos_adr]
        vel_err = -data.qvel[qvel_adr]

        ctrl = params.Kp * pos_err + params.Kd * vel_err
        data.ctrl[i] = np.clip(ctrl, -300, 300)


# -----------------------------
# Simulation
# -----------------------------
def run_sim():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    joint_map = get_joint_map(model)
    base_pose = create_base_pose(model, joint_map)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t = 0.0

        while viewer.is_running():
            step_start = time.time()

            q_target = gait_target(base_pose, t, joint_map, model)
            pd_control(model, data, q_target)

            mujoco.mj_step(model, data)
            viewer.sync()

            t += model.opt.timestep

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


# -----------------------------
# Run both
# -----------------------------
if __name__ == "__main__":
    threading.Thread(target=start_gui, daemon=True).start()
    run_sim()
