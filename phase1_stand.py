import mujoco, mujoco.viewer, numpy as np, time, threading, tkinter as tk

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]
EE_LABELS = ["Left Foot", "Right Foot", "Left Hand", "Right Hand"]
EE_COLORS = ["#3b82f6", "#eab308", "#ef4444", "#22c55e"]

# Which joint DOFs each end-effector controls (indices into qvel/dof space)
# Free joint uses DOFs 0-5, then hinge joints start at DOF 6
LIMB_DOFS = {
    "left_foot_site":  [],  # filled at runtime
    "right_foot_site": [],
    "left_hand_site":  [],
    "right_hand_site": [],
}

LIMB_JOINTS = {
    "left_foot_site":  ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
                         "left_knee_joint", "left_ankle_pitch_joint"],
    "right_foot_site": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
                         "right_knee_joint", "right_ankle_pitch_joint"],
    "left_hand_site":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                         "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_pitch_joint"],
    "right_hand_site": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                         "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_pitch_joint"],
}

# ──────────────────────────────────────
# Setup
# ──────────────────────────────────────

def load_model(path):
    model = mujoco.MjModel.from_xml_path(path)
    return model, mujoco.MjData(model)

def get_joint_map(model):
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap

def build_limb_dofs(model):
    """Map each EE to the DOF indices of its kinematic chain."""
    for site_name, joint_names in LIMB_JOINTS.items():
        dofs = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                dofs.append(model.jnt_dofadr[jid])
        LIMB_DOFS[site_name] = dofs

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

# ──────────────────────────────────────
# Per-limb IK (only moves joints in that limb)
# ──────────────────────────────────────

def ik_solve_limb(model, data, site_name, target_pos, max_iters=100, step_size=0.5, tol=0.002):
    """Solve IK for a single end-effector, only modifying its limb's joints."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    dof_ids = LIMB_DOFS[site_name]
    joint_names = LIMB_JOINTS[site_name]
    nv = model.nv
    jacp = np.zeros((3, nv))

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            break

        jacp[:] = 0
        mujoco.mj_jacSite(model, data, jacp, None, sid)

        # Extract only columns for this limb's DOFs
        J = jacp[:, dof_ids]  # 3 x n_limb_dofs

        # Damped least squares
        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        # Apply to only this limb's joints
        for i, dof in enumerate(dof_ids):
            # dof -> qpos: for hinge joints, qposadr = dofadr + 7 - 6 = dofadr + 1
            # Actually need to find the joint and its qposadr
            jname = joint_names[i]
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qa = model.jnt_qposadr[jid]
            data.qpos[qa] += step_size * dq[i]
            # Clamp to joint limits
            lo, hi = model.jnt_range[jid]
            if lo < hi:
                data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)

# ──────────────────────────────────────
# GUI
# ──────────────────────────────────────

class EEController:
    def __init__(self):
        self.offsets = {name: np.zeros(3) for name in EE_SITES}
        self.home = None
        self.active = {name: False for name in EE_SITES}  # which EEs have moved

    def get_targets(self):
        if self.home is None:
            return {}
        targets = {}
        for name in EE_SITES:
            off = self.offsets[name]
            if np.any(np.abs(off) > 0.001):
                targets[name] = self.home[name] + off
                self.active[name] = True
            elif self.active[name]:
                targets[name] = self.home[name]
                self.active[name] = False
        return targets

def start_gui(ctrl):
    root = tk.Tk()
    root.title("G1 IK Controller")
    root.configure(bg="#1a1a1a")

    sliders = {}
    RANGE = 0.20

    for idx, name in enumerate(EE_SITES):
        frame = tk.LabelFrame(root, text=f" {EE_LABELS[idx]} ",
                               font=("Arial", 11, "bold"),
                               fg=EE_COLORS[idx], bg="#1a1a1a", padx=5, pady=3)
        frame.pack(padx=8, pady=3, fill="x")

        sliders[name] = {}
        for axis, label in enumerate(["X", "Y", "Z"]):
            row = tk.Frame(frame, bg="#1a1a1a")
            row.pack(fill="x")
            tk.Label(row, text=label, font=("Arial", 9, "bold"), fg="white",
                     bg="#1a1a1a", width=2).pack(side="left")
            s = tk.Scale(row, from_=-RANGE, to=RANGE, resolution=0.005,
                         orient=tk.HORIZONTAL, length=220,
                         bg="#2d2d2d", fg="white", troughcolor="#404040",
                         highlightbackground="#1a1a1a", showvalue=True)
            s.set(0.0)
            s.pack(side="left", fill="x", expand=True)
            sliders[name][axis] = s

    def update():
        for name in EE_SITES:
            for axis in range(3):
                ctrl.offsets[name][axis] = sliders[name][axis].get()
        root.after(20, update)

    def reset():
        for name in EE_SITES:
            for axis in range(3):
                sliders[name][axis].set(0.0)

    tk.Button(root, text="Reset All", command=reset, bg="#dc2626", fg="white",
              font=("Arial", 10, "bold"), padx=20, pady=5).pack(pady=8)

    root.after(20, update)
    root.mainloop()

# ──────────────────────────────────────
# Main — pure kinematics, no physics
# ──────────────────────────────────────

def main():
    model, data = load_model(MODEL_PATH)
    model.opt.gravity[:] = 0  # not needed since no mj_step, but just in case
    joint_map = get_joint_map(model)
    build_limb_dofs(model)

    # Set spider pose, belly up on ground
    q_pose = load_pose(model, joint_map)
    data.qpos[:] = q_pose
    data.qpos[0:3] = [0, 0, 0.25]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0

    # Save the fixed base pose
    base_qpos = data.qpos[0:7].copy()

    mujoco.mj_forward(model, data)

    # Controller
    ctrl = EEController()
    ctrl.home = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        ctrl.home[name] = data.site_xpos[sid].copy()
    print("Home positions:")
    for name, pos in ctrl.home.items():
        print(f"  {name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # Start GUI
    threading.Thread(target=start_gui, args=(ctrl,), daemon=True).start()

    print("\nReady! Move sliders to control end-effectors.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t = time.time()

            # Always lock the base in place
            data.qpos[0:7] = base_qpos

            # Solve IK per-limb for any moved end-effector
            targets = ctrl.get_targets()
            for site_name, target_pos in targets.items():
                ik_solve_limb(model, data, site_name, target_pos)

            # Kinematics only — no physics, perfectly stable
            mujoco.mj_forward(model, data)
            viewer.sync()

            time.sleep(max(0, 0.02 - (time.time() - t)))  # ~50fps

if __name__ == "__main__":
    main()
