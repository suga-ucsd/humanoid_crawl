import mujoco, mujoco.viewer, numpy as np, tkinter as tk, threading

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
model.opt.gravity[:] = 0.0

# Build joint map
joint_map = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): model.jnt_qposadr[i]
             for i in range(model.njnt)}

# Load saved pose
saved_pose = {}
try:
    with open("g1_pose.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[0] in joint_map:
                saved_pose[parts[0]] = float(parts[1])
except FileNotFoundError:
    print("No pose file found, using defaults.")

for name, val in saved_pose.items():
    data.qpos[joint_map[name]] = val
mujoco.mj_forward(model, data)

# GUI
sliders = {}

def start_gui():
    root = tk.Tk()
    root.title("G1 Joints")
    canvas = tk.Canvas(root)
    sb = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    frame = tk.Frame(canvas)
    frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame, anchor="nw")
    canvas.configure(yscrollcommand=sb.set)
    canvas.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    for name in joint_map:
        s = tk.Scale(frame, from_=-5, to=5, resolution=0.01,
                     orient=tk.HORIZONTAL, length=400, label=name)
        s.set(saved_pose.get(name, 0.0))
        s.pack()
        sliders[name] = s

    def update():
        for n, s in sliders.items():
            data.qpos[joint_map[n]] = s.get()
        root.after(50, update)

    def save():
        with open("g1_pose.txt", "w") as f:
            for n, s in sliders.items():
                f.write(f"{n} {s.get()}\n")
        print("Pose saved.")

    tk.Button(frame, text="Save Pose", command=save).pack(pady=10)
    root.after(50, update)
    root.mainloop()

def run_sim():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)
            viewer.sync()

threading.Thread(target=start_gui, daemon=True).start()
run_sim()
