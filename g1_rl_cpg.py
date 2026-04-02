#!/usr/bin/env python3
"""
G1 Spider Crawl — CPG + RL (v2)

Redesigned CPG with proper stance/swing cycle.
Manual mode with live sliders to tune CPG parameters.

Usage:
    python g1_rl_cpg.py --mode manual      # tune CPG with sliders
    python g1_rl_cpg.py --mode test         # random RL agent
    python g1_rl_cpg.py --mode train --steps 5000000 --n_envs 8
    python g1_rl_cpg.py --mode eval --checkpoint models/best/best_model
"""

import argparse, os, time, threading
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]

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

# Which joints each limb uses for locomotion
LIMB_JOINTS = {
    "left_leg":  ["left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint"],
    "right_leg": ["right_hip_pitch_joint", "right_hip_roll_joint", "right_knee_joint"],
    "left_arm":  ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_joint"],
    "right_arm": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_joint"],
}

LIMB_NAMES = ["left_leg", "right_leg", "left_arm", "right_arm"]

# Trot gait: diagonal pairs move together
DEFAULT_PHASES = [0.0, np.pi, np.pi, 0.0]

# ──────────────────────────────────────
# Core
# ──────────────────────────────────────

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

def configure_physics(model):
    model.opt.timestep = 0.002
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    model.opt.solver = 2
    for i in range(model.ngeom):
        model.geom_solref[i] = [0.02, 1.0]
        model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]
    for i in range(6, model.nv):
        model.dof_damping[i] = 1.0
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    model.geom_friction[floor_id] = [1.5, 0.005, 0.001]
    for i in range(model.ngeom):
        if i != floor_id:
            model.geom_friction[i] = [1.5, 0.005, 0.001]

def build_gains(model):
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 0.5
        kd[i] = kp[i] * 0.5
    return kp, kd, max_tau

def pd_control_vec(model, data, q_target, kp, kd, max_tau):
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

# ──────────────────────────────────────
# CPG v2: Proper stance/swing cycle
# ──────────────────────────────────────

class CPG:
    """
    Generates crawling motion with distinct stance and swing phases.

    Each limb cycle:
        SWING  (phase 0→π):   Foot lifts, leg swings forward (repositioning)
        STANCE (phase π→2π):  Foot plants, leg pushes backward (propulsion)

    The key asymmetry: during stance the foot is on the ground and pushes,
    during swing the foot is in the air and repositions. This creates net
    forward force.

    Joint roles per limb:
        joint_0 (pitch):  Forward/back swing. Positive during swing, negative during stance.
        joint_1 (roll):   Lateral lift during swing for ground clearance.
        joint_2 (knee):   Bend extra during swing to clear ground.
    """

    def __init__(self, model, home_qpos, joint_map):
        self.model = model
        self.home_qpos = home_qpos
        self.joint_map = joint_map
        self.phase = 0.0

        self.limb_qpos_ids = {}
        for limb_name, joint_names in LIMB_JOINTS.items():
            ids = []
            for jname in joint_names:
                if jname in joint_map:
                    ids.append(joint_map[jname])
            self.limb_qpos_ids[limb_name] = ids

        self.joint_limits = {}
        for jname, qa in joint_map.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                lo, hi = model.jnt_range[jid]
                self.joint_limits[qa] = (lo, hi)

        # Sign convention: legs push one way, arms push the other
        # because they're on opposite sides of the body
        # For belly-up spider: legs are at the "bottom" (hip side),
        # arms are at the "top" (shoulder side)
        self.pitch_signs = {
            "left_leg": -1.0,   # negative pitch = push backward for legs
            "right_leg": -1.0,
            "left_arm": 1.0,    # positive pitch = push backward for arms
            "right_arm": 1.0,
        }
        self.roll_signs = {
            "left_leg": 1.0,
            "right_leg": -1.0,  # mirror for right side
            "left_arm": 1.0,
            "right_arm": -1.0,
        }

    def step(self, dt, frequency):
        self.phase += 2.0 * np.pi * frequency * dt
        if self.phase > 2.0 * np.pi:
            self.phase -= 2.0 * np.pi

    def generate(self, amplitudes, phase_offsets, frequency, swing_height, duty_cycle=0.4):
        """
        Args:
            amplitudes: [4] per-limb amplitude (0-1)
            phase_offsets: [4] per-limb phase offset
            frequency: gait frequency Hz (unused here, just for API compat)
            swing_height: how much to lift during swing (0-1)
            duty_cycle: fraction of cycle spent in swing (0.3-0.5)
        """
        q_target = self.home_qpos.copy()

        for i, limb_name in enumerate(LIMB_NAMES):
            ids = self.limb_qpos_ids[limb_name]
            if len(ids) < 3:
                continue

            amp = amplitudes[i] * 0.35          # max 0.35 rad swing
            limb_phase = self.phase + DEFAULT_PHASES[i] + phase_offsets[i]

            # Normalize phase to [0, 2π]
            lp = limb_phase % (2.0 * np.pi)
            swing_end = duty_cycle * 2.0 * np.pi

            if lp < swing_end:
                # ── SWING PHASE: foot in air, leg moves forward ──
                t = lp / swing_end                     # 0→1 through swing
                swing_curve = np.sin(t * np.pi)        # smooth bell: 0→1→0

                pitch_offset = amp * self.pitch_signs[limb_name] * swing_curve
                roll_offset = swing_height * 0.15 * self.roll_signs[limb_name] * swing_curve
                knee_offset = swing_height * 0.3 * swing_curve
            else:
                # ── STANCE PHASE: foot on ground, leg pushes back ──
                t = (lp - swing_end) / (2.0 * np.pi - swing_end)  # 0→1 through stance
                stance_curve = np.sin(t * np.pi)       # smooth push

                pitch_offset = -amp * self.pitch_signs[limb_name] * stance_curve * 0.7
                roll_offset = 0.0                      # keep foot planted
                knee_offset = 0.0                      # keep foot on ground

            q_target[ids[0]] += pitch_offset
            q_target[ids[1]] += roll_offset
            q_target[ids[2]] += knee_offset

        # Clamp to limits
        for qa, (lo, hi) in self.joint_limits.items():
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

        return q_target

# ──────────────────────────────────────
# Manual mode with tkinter sliders
# ──────────────────────────────────────

class CPGParams:
    """Shared state for sliders → sim loop."""
    def __init__(self):
        self.frequency = 0.8
        self.swing_height = 0.5
        self.duty_cycle = 0.4
        self.amplitudes = [0.4, 0.4, 0.4, 0.4]
        self.phase_offsets = [0.0, 0.0, 0.0, 0.0]

def start_manual_gui(params):
    import tkinter as tk

    root = tk.Tk()
    root.title("CPG Tuner")
    root.configure(bg="#1a1a1a")

    style = {"bg": "#1a1a1a", "fg": "white"}

    def make_slider(parent, label, from_, to, resolution, initial, callback):
        tk.Label(parent, text=label, font=("Arial", 9), **style).pack(anchor="w")
        s = tk.Scale(parent, from_=from_, to=to, resolution=resolution,
                     orient="horizontal", length=250,
                     bg="#2d2d2d", fg="white", troughcolor="#404040",
                     highlightbackground="#1a1a1a", command=callback)
        s.set(initial)
        s.pack(fill="x", padx=5)
        return s

    # Global params
    gf = tk.LabelFrame(root, text=" Global ", font=("Arial", 11, "bold"),
                        fg="#22c55e", bg="#1a1a1a", padx=5, pady=3)
    gf.pack(padx=8, pady=3, fill="x")

    make_slider(gf, "Frequency (Hz)", 0.1, 3.0, 0.05, params.frequency,
                lambda v: setattr(params, 'frequency', float(v)))
    make_slider(gf, "Swing Height", 0.0, 1.0, 0.05, params.swing_height,
                lambda v: setattr(params, 'swing_height', float(v)))
    make_slider(gf, "Duty Cycle (swing fraction)", 0.2, 0.6, 0.05, params.duty_cycle,
                lambda v: setattr(params, 'duty_cycle', float(v)))

    # Per-limb params
    colors = ["#3b82f6", "#eab308", "#ef4444", "#a855f7"]
    labels = ["Left Leg", "Right Leg", "Left Arm", "Right Arm"]

    for idx in range(4):
        lf = tk.LabelFrame(root, text=f" {labels[idx]} ",
                            font=("Arial", 10, "bold"),
                            fg=colors[idx], bg="#1a1a1a", padx=5, pady=3)
        lf.pack(padx=8, pady=2, fill="x")

        def make_amp_cb(i):
            return lambda v: params.amplitudes.__setitem__(i, float(v))
        def make_phase_cb(i):
            return lambda v: params.phase_offsets.__setitem__(i, float(v))

        make_slider(lf, "Amplitude", 0.0, 1.0, 0.05, params.amplitudes[idx],
                     make_amp_cb(idx))
        make_slider(lf, "Phase Offset", -1.57, 1.57, 0.05, params.phase_offsets[idx],
                     make_phase_cb(idx))

    def reset():
        params.frequency = 0.8
        params.swing_height = 0.5
        params.duty_cycle = 0.4
        params.amplitudes[:] = [0.4, 0.4, 0.4, 0.4]
        params.phase_offsets[:] = [0.0, 0.0, 0.0, 0.0]

    tk.Button(root, text="Reset", command=reset,
              bg="#dc2626", fg="white", font=("Arial", 10, "bold")).pack(pady=8)

    root.mainloop()

def run_manual(args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)
    kp, kd, max_tau = build_gains(model)

    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 0.25]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    cpg = CPG(model, home_qpos, joint_map)
    params = CPGParams()

    # Start GUI
    threading.Thread(target=start_manual_gui, args=(params,), daemon=True).start()

    # EE site ids for contact check
    ee_sids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in EE_SITES]
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    print("Manual CPG mode — tune sliders to find a crawling gait")
    print("Watch the terminal for velocity and contact info\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t = time.time()

                q_target = cpg.generate(
                    np.array(params.amplitudes),
                    np.array(params.phase_offsets),
                    params.frequency,
                    params.swing_height,
                    params.duty_cycle,
                )
                q_target[0:7] = data.qpos[0:7]

                cpg.step(0.02, params.frequency)

                for _ in range(10):
                    pd_control_vec(model, data, q_target, kp, kd, max_tau)
                    mujoco.mj_step(model, data)

                viewer.sync()
                step += 1

                if step % 50 == 0:
                    # Check contacts
                    contacts = [0, 0, 0, 0]
                    for ci in range(data.ncon):
                        c = data.contact[ci]
                        g1, g2 = c.geom1, c.geom2
                        if g1 != floor_id and g2 != floor_id:
                            continue
                        rg = g2 if g1 == floor_id else g1
                        rb = model.geom_bodyid[rg]
                        for j, sid in enumerate(ee_sids):
                            eb = model.site_bodyid[sid]
                            b = eb
                            while b > 0:
                                if b == rb:
                                    contacts[j] = 1
                                    break
                                b = model.body_parentid[b]

                    vx, vy = data.qvel[0], data.qvel[1]
                    print(f"  step={step:5d}  vel=[{vx:+.3f},{vy:+.3f}]  "
                          f"h={data.qpos[2]:.3f}  contacts={contacts}  "
                          f"freq={params.frequency:.1f}  amps={[f'{a:.2f}' for a in params.amplitudes]}")

                time.sleep(max(0, 0.02 - (time.time() - t)))
        except KeyboardInterrupt:
            print("\nDone.")

# ──────────────────────────────────────
# Environment for RL
# ──────────────────────────────────────

class G1CPGEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    CONTROL_DT = 0.02
    N_SUBSTEPS = 10
    MAX_STEPS = 1000

    def __init__(self, render_mode=None, target_dir=None):
        super().__init__()
        self.render_mode = render_mode

        td = np.array(target_dir or [1.0, 0.0], dtype=np.float64)
        self.target_dir = td / (np.linalg.norm(td) + 1e-8)

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        self.joint_map = get_joint_map(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)
        self.kp, self.kd, self.max_tau = build_gains(self.model)

        self.act_qpos = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        self.ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in EE_SITES
        ]
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.cpg = CPG(self.model, self.home_qpos, self.joint_map)

        # 10D action: 4 amps + 4 phases + freq + swing_height
        self.action_space = spaces.Box(-1.0, 1.0, shape=(10,), dtype=np.float32)

        obs_dim = 4 + 3 + 3 + self.model.nu * 2 + 4 + 4 + 2 + 2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(10)
        self.viewer = None

    def _get_ee_contacts(self):
        contacts = np.zeros(4)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != self.floor_id and g2 != self.floor_id:
                continue
            robot_geom = g2 if g1 == self.floor_id else g1
            robot_body = self.model.geom_bodyid[robot_geom]
            for j, sid in enumerate(self.ee_site_ids):
                ee_body = self.model.site_bodyid[sid]
                b = ee_body
                while b > 0:
                    if b == robot_body:
                        contacts[j] = 1.0
                        break
                    b = self.model.body_parentid[b]
        return contacts

    def _get_obs(self):
        d = self.data
        joint_off = np.zeros(self.model.nu)
        joint_vel = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            va = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            joint_off[i] = d.qpos[qa] - self.home_qpos[qa]
            joint_vel[i] = d.qvel[va]

        ee_h = np.array([d.site_xpos[sid][2] for sid in self.ee_site_ids])
        contacts = self._get_ee_contacts()
        clock = np.array([np.sin(self.cpg.phase), np.cos(self.cpg.phase)])

        return np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3],
            joint_off, joint_vel,
            ee_h, contacts, clock, self.target_dir, [d.qpos[2]]
        ]).astype(np.float32)

    def _decode_action(self, action):
        action = np.clip(action, -1.0, 1.0)
        amplitudes = (action[0:4] + 1.0) * 0.5
        phase_offsets = action[4:8] * (np.pi / 4.0)
        frequency = 1.0 + action[8] * 0.8      # 0.2 to 1.8 Hz
        swing_height = (action[9] + 1.0) * 0.5
        return amplitudes, phase_offsets, frequency, swing_height

    def _get_reward(self, action):
        d = self.data

        alive = 1.0

        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = max(0.0, -pelvis_rot[2, 2])

        vel_xy = d.qvel[0:2]
        forward_vel = np.dot(vel_xy, self.target_dir)
        lateral_vel = abs(vel_xy[0] * self.target_dir[1] - vel_xy[1] * self.target_dir[0])

        height_err = abs(d.qpos[2] - 0.15)
        height_rew = max(0.0, 1.0 - 5.0 * height_err)

        contacts = self._get_ee_contacts()
        contact_rew = min(np.sum(contacts), 3) / 3.0

        ang_vel = np.sum(np.square(d.qvel[3:6]))
        action_rate = np.sum(np.square(action - self.prev_action))
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        reward = (
            1.0 * alive
            + 1.0 * belly_up
            + 5.0 * forward_vel
            + 0.5 * height_rew
            + 0.3 * contact_rew
            - 1.0 * lateral_vel
            - 0.01 * ang_vel
            - 0.05 * action_rate
            - 0.0005 * energy
        )
        return reward

    def _is_terminated(self):
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True
        h = self.data.qpos[2]
        return h > 0.8 or h < -0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02,
                                                          size=self.data.qpos[7:].shape)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.cpg.phase = 0.0
        self.step_count = 0
        self.prev_action = np.zeros(10)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        amps, phases, freq, sh = self._decode_action(action)

        q_target = self.cpg.generate(amps, phases, freq, sh)
        q_target[0:7] = self.data.qpos[0:7]
        self.cpg.step(self.CONTROL_DT, freq)

        for _ in range(self.N_SUBSTEPS):
            pd_control_vec(self.model, self.data, q_target,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        reward = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.MAX_STEPS
        self.prev_action = action.copy()
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

# ──────────────────────────────────────
# Train / Eval / Test (same as before)
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    target = [float(x) for x in args.target]

    def make_env():
        def _init():
            return G1CPGEnv(target_dir=target)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    eval_env = G1CPGEnv(target_dir=target)

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        model = PPO.load(args.checkpoint, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=args.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-0.5,
            ),
        )

    callbacks = [
        CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                           save_path="models/", name_prefix="g1_cpg"),
        EvalCallback(eval_env, best_model_save_path="models/best/",
                     log_path="logs/eval/",
                     eval_freq=max(50_000 // args.n_envs, 1),
                     n_eval_episodes=5, deterministic=True),
    ]

    print(f"\nCPG+RL Training | {args.steps:,} steps | {args.n_envs} envs | {args.device}\n")
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save("models/g1_cpg_final")
    env.close()

def evaluate(args):
    from stable_baselines3 import PPO
    target = [float(x) for x in args.target]
    if not args.checkpoint:
        print("Error: --checkpoint required")
        return
    model = PPO.load(args.checkpoint)
    env = G1CPGEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()
    total_reward, steps, episodes = 0, 0, 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.02)
            if terminated or truncated:
                episodes += 1
                print(f"Ep {episodes} — steps: {steps}, reward: {total_reward:.1f}")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

def test_random(args):
    target = [float(x) for x in args.target]
    env = G1CPGEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()
    print("Random CPG agent\n")
    total_reward, steps = 0, 0
    try:
        while True:
            if steps % 20 == 0:
                action = env.action_space.sample() * 0.5
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            if steps % 100 == 0:
                amps, phases, freq, sh = env._decode_action(action)
                print(f"  step={steps:4d}  rew={total_reward:7.1f}  "
                      f"freq={freq:.2f}  h={env.data.qpos[2]:.3f}")
            if terminated or truncated:
                print(f"Episode done — steps: {steps}, reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider — CPG + RL v2")
    parser.add_argument("--mode", choices=["train", "eval", "test", "manual"], required=True,
                        help="manual: tune CPG with sliders | test: random | train: RL | eval: run model")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--target", nargs=2, default=["1.0", "0.0"])
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    {"train": train, "eval": evaluate, "test": test_random, "manual": run_manual}[args.mode](args)

if __name__ == "__main__":
    main()
