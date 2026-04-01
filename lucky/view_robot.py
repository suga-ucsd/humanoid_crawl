import mujoco
import mujoco.viewer
import numpy as np
import tkinter as tk
import threading

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "unitree_ros/robots/g1_description/g1_29dof.xml"

# -----------------------------
# Load model & data
# -----------------------------
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Disable gravity
model.opt.gravity[:] = 0.0

# Disable collisions
for i in range(model.npair):
    model.eq_data[i].solref[0] = 0.0  # optional, just make collisions ineffective
# Or simpler: just ignore collisions in simulation

# Fix floating base
data.qpos[:7] = 0.0
data.qvel[:6] = 0.0

# -----------------------------
# Joint mapping
# -----------------------------
joint_map = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_idx = model.jnt_qposadr[i]
    qvel_idx = model.jnt_dofadr[i]
    joint_map[name] = qpos_idx
    print(name, "qpos_idx:", qpos_idx)

# -----------------------------
# GUI state
# -----------------------------
joint_sliders = {}

def start_gui():
    root = tk.Tk()
    root.title("G1 Joint Manipulator")

    # --- Create sliders ---
    for name in joint_map.keys():
        s = tk.Scale(root, from_=-10, to=10, resolution=0.01,
                     orient=tk.HORIZONTAL, length=400, label=name)
        s.pack()
        joint_sliders[name] = s

    # --- Update MuJoCo pose from sliders ---
    def update_pose():
        for name, s in joint_sliders.items():
            idx = joint_map[name]
            data.qpos[idx] = s.get()
        root.after(50, update_pose)

    # --- Save current pose to text file ---
    def save_pose():
        filename = "g1_pose.txt"
        with open(filename, "w") as f:
            for name, s in joint_sliders.items():
                f.write(f"{name} {s.get()}\n")
        print(f"Pose saved to {filename}")

    # --- Add save button ---
    save_button = tk.Button(root, text="Save Pose", command=save_pose)
    save_button.pack(pady=10)

    # Start updating
    root.after(50, update_pose)
    root.mainloop()

# -----------------------------
# Simulation loop
# -----------------------------
def run_sim():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

# -----------------------------
# Run GUI + simulation
# -----------------------------
threading.Thread(target=start_gui, daemon=True).start()
run_sim()
