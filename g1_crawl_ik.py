import mujoco, glfw, numpy as np, time

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]
EE_LABELS = ["Left Foot", "Right Foot", "Left Hand", "Right Hand"]

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

def get_joint_map(model):
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap

def build_limb_dofs(model):
    limb_dofs = {}
    for site_name, joint_names in LIMB_JOINTS.items():
        dofs = []
        for jname in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                dofs.append(model.jnt_dofadr[jid])
        limb_dofs[site_name] = dofs
    return limb_dofs

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
# Per-limb IK
# ──────────────────────────────────────

def ik_solve_limb(model, data, site_name, target_pos, limb_dofs, max_iters=100, step_size=0.5, tol=0.002):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    dof_ids = limb_dofs[site_name]
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

        J = jacp[:, dof_ids]
        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        dq = J.T @ np.linalg.solve(JJT, err)

        for i, jname in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qa = model.jnt_qposadr[jid]
            data.qpos[qa] += step_size * dq[i]
            lo, hi = model.jnt_range[jid]
            if lo < hi:
                data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)

# ──────────────────────────────────────
# Camera helpers
# ──────────────────────────────────────

def get_camera_vectors(cam):
    """Get right/up vectors in world space from camera azimuth/elevation."""
    az = np.radians(cam.azimuth)
    el = np.radians(cam.elevation)
    fwd = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    world_up = np.array([0, 0, 1])
    right = np.cross(fwd, world_up)
    n = np.linalg.norm(right)
    if n < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right /= n
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)
    return right, up

# ──────────────────────────────────────
# Interactive Viewer
# ──────────────────────────────────────

class Viewer:
    def __init__(self, model, data, limb_dofs):
        self.model = model
        self.data = data
        self.limb_dofs = limb_dofs

        # Mouse state
        self.selected = None       # selected EE site name
        self.target_pos = None     # 3D target for selected EE
        self.dragging = False
        self.rotating = False
        self.panning = False
        self.lastx = 0
        self.lasty = 0

        # Init GLFW
        if not glfw.init():
            raise RuntimeError("GLFW init failed")
        self.window = glfw.create_window(1280, 720, "G1 IK — Right-click to grab end-effectors", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # MuJoCo rendering
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [0, 0, 0.2]
        self.cam.distance = 2.0
        self.cam.azimuth = 90
        self.cam.elevation = -25

        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = False

        # Callbacks
        glfw.set_mouse_button_callback(self.window, self._on_button)
        glfw.set_cursor_pos_callback(self.window, self._on_move)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    def _find_nearest_ee(self, click_3d):
        """Find the nearest end-effector site to a 3D click point."""
        best, best_dist = None, 0.2  # 20cm threshold
        for name in EE_SITES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            d = np.linalg.norm(click_3d - self.data.site_xpos[sid])
            if d < best_dist:
                best_dist = d
                best = name
        return best

    def _on_button(self, window, button, action, mods):
        self.lastx, self.lasty = glfw.get_cursor_pos(window)

        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_RIGHT:
                # Raycast to find what we clicked on
                w, h = glfw.get_framebuffer_size(window)
                relx = self.lastx / w
                rely = (h - self.lasty) / h
                selpnt = np.zeros(3)
                geomid = np.zeros(1, dtype=np.int32)
                flexid = np.zeros(1, dtype=np.int32)
                skinid = np.zeros(1, dtype=np.int32)

                body = mujoco.mjv_select(self.model, self.data, self.opt,
                                          w / h, relx, rely, self.scn,
                                          selpnt, geomid, flexid, skinid)
                if body >= 0:
                    ee = self._find_nearest_ee(selpnt)
                    if ee:
                        self.selected = ee
                        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, ee)
                        self.target_pos = self.data.site_xpos[sid].copy()
                        self.dragging = True
                        idx = EE_SITES.index(ee)
                        print(f"Grabbed: {EE_LABELS[idx]}")

            elif button == glfw.MOUSE_BUTTON_LEFT:
                if mods == glfw.MOD_SHIFT:
                    self.panning = True
                else:
                    self.rotating = True

            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self.panning = True

        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_RIGHT and self.dragging:
                idx = EE_SITES.index(self.selected) if self.selected else -1
                if idx >= 0:
                    print(f"Released: {EE_LABELS[idx]}")
                self.dragging = False
                self.selected = None
                self.target_pos = None

            elif button == glfw.MOUSE_BUTTON_LEFT:
                self.rotating = False
                self.panning = False

            elif button == glfw.MOUSE_BUTTON_MIDDLE:
                self.panning = False

    def _on_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        w, h = glfw.get_framebuffer_size(window)

        if self.dragging and self.target_pos is not None:
            # Move target in camera plane
            right, up = get_camera_vectors(self.cam)
            scale = 0.002 * self.cam.distance
            self.target_pos += right * dx * scale
            self.target_pos -= up * dy * scale

        elif self.rotating:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ROTATE_H,
                                   dx / w, dy / h, self.scn, self.cam)

        elif self.panning:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_MOVE_H,
                                   dx / w, dy / h, self.scn, self.cam)

    def _on_scroll(self, window, xoff, yoff):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                               0, -0.05 * yoff, self.scn, self.cam)

    def render(self):
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None,
                                self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
        mujoco.mjr_render(viewport, self.scn, self.con)

        # Draw HUD
        if self.selected:
            idx = EE_SITES.index(self.selected)
            text = f"Dragging: {EE_LABELS[idx]}"
        else:
            text = "Right-click an end-effector to grab it"
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                            viewport, text, "", self.con)

        glfw.swap_buffers(self.window)

    def is_running(self):
        return not glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()

# ──────────────────────────────────────
# Main
# ──────────────────────────────────────

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0

    joint_map = get_joint_map(model)
    limb_dofs = build_limb_dofs(model)

    # Spider pose, belly up
    data.qpos[:] = load_pose(model, joint_map)
    data.qpos[0:3] = [0, 0, 0.25]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0
    base_qpos = data.qpos[0:7].copy()
    mujoco.mj_forward(model, data)

    print("Controls:")
    print("  Right-click + drag  →  grab & move end-effector")
    print("  Left-click + drag   →  orbit camera")
    print("  Middle/Shift+left   →  pan camera")
    print("  Scroll              →  zoom\n")

    viewer = Viewer(model, data, limb_dofs)

    while viewer.is_running():
        glfw.poll_events()

        # Lock base
        data.qpos[0:7] = base_qpos

        # IK for grabbed end-effector
        if viewer.dragging and viewer.selected and viewer.target_pos is not None:
            ik_solve_limb(model, data, viewer.selected, viewer.target_pos, limb_dofs)

        mujoco.mj_forward(model, data)
        viewer.render()

    viewer.close()

if __name__ == "__main__":
    main()
