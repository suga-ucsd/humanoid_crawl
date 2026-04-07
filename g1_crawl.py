#!/usr/bin/env python3
"""
G1 Spider Crawl — 8 DOF, Sine Wave Gait

Only 2 joints per limb: pitch (swing) + knee/elbow (lift).
Step 1: Built-in sine gait — no RL, no reference files.
Step 2: RL optimizes gait parameters (amplitude, frequency, phases).

Usage:
    python g1_crawl.py --mode sine          # watch sine wave gait (no RL)
    python g1_crawl.py --mode tune          # slider GUI to tune gait
    python g1_crawl.py --mode test          # random RL agent
    python g1_crawl.py --mode train --steps 5000000 --n_envs 16
    python g1_crawl.py --mode eval --checkpoint models/best/best_model
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
    "right_hip_pitch_joint": 88, "right_hip_roll_joint": 139,
    "right_hip_yaw_joint": 88, "right_knee_joint": 139,
    "waist_yaw_joint": 88,
    "left_shoulder_pitch_joint": 60, "left_shoulder_roll_joint": 60,
    "left_shoulder_yaw_joint": 25, "left_elbow_joint": 60,
    "left_wrist_pitch_joint": 10,
    "right_shoulder_pitch_joint": 60, "right_shoulder_roll_joint": 60,
    "right_shoulder_yaw_joint": 25, "right_elbow_joint": 60,
    "right_wrist_pitch_joint": 10,
}

# 8 active joints: 2 per limb (pitch + bend)
ACTIVE_JOINTS = [
    "left_hip_pitch_joint", "left_knee_joint",
    "right_hip_pitch_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_elbow_joint",
]

# Limb structure: [pitch_joint, bend_joint]
LIMBS = {
    "left_leg":  ("left_hip_pitch_joint", "left_knee_joint"),
    "right_leg": ("right_hip_pitch_joint", "right_knee_joint"),
    "left_arm":  ("left_shoulder_pitch_joint", "left_elbow_joint"),
    "right_arm": ("right_shoulder_pitch_joint", "right_elbow_joint"),
}
LIMB_NAMES = ["left_leg", "right_leg", "left_arm", "right_arm"]

# Wave gait: each limb offset by 90° so only 1 swings at a time
PHASE_OFFSETS = [0.0, np.pi/2, np.pi, 3*np.pi/2]
# PHASE_OFFSETS = [0.0, np.pi, np.pi, 0.0]

SWING_DUTY = 0.25  # 25% swing, 75% stance

BELLY_UP_QUAT = np.array([0.707, 0.0, -0.707, 0.0])

# ──────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────
def get_joint_map(model):
    """
    Maps joint names → qpos indices in MuJoCo state.

    This is critical because MuJoCo stores all joint states
    in a flat vector (qpos), so we need indexing.
    """
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap

def load_pose(model, joint_map, filename="g1_pose.txt"):
    """
    Loads a reference pose from file.

    If file doesn't exist → defaults to model.qpos0.

    Used as:
    → neutral crawling configuration
    → PD target baseline
    """    
    q = np.copy(model.qpos0)
    try:
        with open(filename) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in joint_map:
                    q[joint_map[parts[0]]] = float(parts[1])
    except FileNotFoundError:
        pass
    return q

def configure_physics(model):
    """
    Sets MuJoCo physics parameters.

    Important tweaks:
    - Smaller timestep → more stable simulation
    - Higher solver iterations → better contact resolution
    - Increased friction → prevents slipping during crawl
    """
    model.opt.timestep = 0.002
    model.opt.iterations = 50
    model.opt.ls_iterations = 20
    model.opt.solver = 2
    for i in range(6, model.nv):
        model.dof_damping[i] = 3.0
    for i in range(model.ngeom):
        if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
            model.geom_friction[i] = [3.0, 0.5, 0.5]

def build_gains(model):
    """
    Builds PD gains and torque limits.

    kp: position stiffness
    kd: damping
    max_tau: actuator saturation

    Gains are scaled based on actuator strength.
    """
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 3.0
        kd[i] = kp[i] * 0.3
    return kp, kd, max_tau

def pd_control_all(model, data, q_target, kp, kd, max_tau):
    """
    Core PD controller.

    Computes:
        tau = kp*(q_target - q) + kd*(-qdot)

    Then clips torques to actuator limits.

    This is what actually drives the robot.
    """
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

def find_spawn_height(model, data, home_qpos):
    """
    Computes correct spawn height so robot is just above ground.

    Prevents:
    - initial penetration
    - floating too high

    Uses lowest end-effector point.
    """
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    lowest = min(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)][2]
                 for n in EE_SITES)
    return data.qpos[2] - lowest + 0.03

def gentle_gravity_settle(model, data, home_qpos, kp, kd, max_tau, duration=2.0):
    """
    Smoothly introduces gravity.

    WHY:
    If gravity is applied instantly → robot collapses violently.

    Instead:
    → gradually ramp gravity from 0 → full
    → PD stabilizes during this phase
    """
    real_gravity = model.opt.gravity.copy()
    n_steps = int(duration / model.opt.timestep)
    jnt_ids = model.actuator_trnid[:, 0]
    act_qa = model.jnt_qposadr[jnt_ids]
    for step in range(n_steps):
        blend = min(1.0, step / (n_steps * 0.7))
        model.opt.gravity[:] = real_gravity * blend
        q_target = home_qpos.copy()
        q_target[0:7] = data.qpos[0:7]
        for i, qa in enumerate(act_qa):
            q_target[qa] = home_qpos[qa]
        pd_control_all(model, data, q_target, kp, kd, max_tau)
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = real_gravity
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


# ──────────────────────────────────────
# Contact Detection
# ──────────────────────────────────────
def get_ee_contacts(model, data):
    """
    Returns which end-effectors are touching the ground.

    Output: [4] binary vector
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    contacts = np.zeros(4)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != floor_id and g2 != floor_id:
            continue
        robot_geom = g2 if g1 == floor_id else g1
        robot_body = model.geom_bodyid[robot_geom]
        for j in range(4):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[j])
            b = model.site_bodyid[sid]
            while b > 0:
                if b == robot_body:
                    contacts[j] = 1.0
                    break
                b = model.body_parentid[b]
    return contacts

def get_torso_ground(model, data):
    """
    Detects if torso/pelvis touches ground.

    This is BAD for crawling → heavy penalty in reward.
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    bad = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"),
           mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")}
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != floor_id and g2 != floor_id:
            continue
        body = model.geom_bodyid[g2 if g1 == floor_id else g1]
        if body in bad:
            return True
    return False

# -------------------------------------
# Triangle wave
# -------------------------------------

def generate_triangle_targets(home_qpos, joint_map, phase, params):
    """
    Triangle gait:
    - Fast swing (lift + move forward quickly)
    - Slow stance (push backward longer → propulsion)

    params:
        frequency
        pitch_amp
        bend_amp
        swing_ratio  (0.1–0.3)  ← CRITICAL
    """

    q_target = home_qpos.copy()

    freq = params["frequency"]
    pitch_amp = params["pitch_amp"]
    bend_amp = params["bend_amp"]
    swing_ratio = params.get("swing_ratio", 0.2)

    pitch_signs = params.get("pitch_signs", [-1, -1, 1, 1])
    bend_signs  = params.get("bend_signs",  [1, -1, -1, 1])

    cycle = 2 * np.pi

    for i, limb_name in enumerate(LIMB_NAMES):
        pitch_joint, bend_joint = LIMBS[limb_name]
        if pitch_joint not in joint_map:
            continue

        limb_phase = (phase + PHASE_OFFSETS[i]) % cycle
        t = limb_phase / cycle  # normalize 0→1

        # ───────────────
        # SWING: fast
        # ───────────────
        if t < swing_ratio:
            s = t / swing_ratio  # 0→1

            # Triangle: linear up/down (fast!)
            if s < 0.5:
                lift = 2 * s
            else:
                lift = 2 * (1 - s)

            pitch_offset = pitch_amp * pitch_signs[i] * (2*s - 1)  # forward sweep
            bend_offset  = bend_amp  * bend_signs[i]  * lift

        # ───────────────
        # STANCE: slow push
        # ───────────────
        else:
            s = (t - swing_ratio) / (1 - swing_ratio)  # 0→1

            # Slow backward push (this creates forward motion)
            pitch_offset = pitch_amp * pitch_signs[i] * (1 - 2*s) * 0.7
            bend_offset  = 0.0

        q_target[joint_map[pitch_joint]] += pitch_offset
        q_target[joint_map[bend_joint]]  += bend_offset

    return q_target


# ──────────────────────────────────────
# Sine Wave Gait Generator
# ──────────────────────────────────────

def generate_sine_targets(home_qpos, joint_map, phase, params):
    """
    Generate joint targets from sine wave gait parameters.

    params dict:
        frequency:     Hz (0.3 - 2.0)
        pitch_amp:     rad, how much pitch joints swing (0.05 - 0.4)
        bend_amp:      rad, how much knee/elbow bends during swing (0.05 - 0.4)
        pitch_signs:   [4] direction multiplier per limb for pitch
        bend_signs:    [4] direction multiplier per limb for bend
    """
    q_target = home_qpos.copy()
    freq = params["frequency"]
    pitch_amp = params["pitch_amp"]
    bend_amp = params["bend_amp"]

    # Signs: legs pitch one way, arms the other (because of body geometry)
    # These are tunable — the sine/tune modes let you find the right ones
    pitch_signs = params.get("pitch_signs", [-1.0, -1.0, 1.0, 1.0])
    bend_signs = params.get("bend_signs", [1.0, 1.0, 1.0, 1.0])

    for i, limb_name in enumerate(LIMB_NAMES):
        pitch_joint, bend_joint = LIMBS[limb_name]
        if pitch_joint not in joint_map or bend_joint not in joint_map:
            continue

        limb_phase = (phase + PHASE_OFFSETS[i]) % (2 * np.pi)
        swing_end = SWING_DUTY * 2 * np.pi

        if limb_phase < swing_end:
            # SWING: lift + swing forward
            t = limb_phase / swing_end  # 0→1
            swing_curve = np.sin(t * np.pi)  # bell: 0→1→0

            pitch_offset = pitch_amp * pitch_signs[i] * swing_curve
            bend_offset = bend_amp * bend_signs[i] * swing_curve
        else:
            # STANCE: push back
            t = (limb_phase - swing_end) / (2 * np.pi - swing_end)  # 0→1
            stance_curve = np.sin(t * np.pi)

            pitch_offset = -pitch_amp * pitch_signs[i] * stance_curve * 0.6
            bend_offset = 0.0

        q_target[joint_map[pitch_joint]] += pitch_offset
        q_target[joint_map[bend_joint]] += bend_offset

    return q_target

# ──────────────────────────────────────
# Mode: SINE
# ──────────────────────────────────────

def run_sine(args):
    """
    Runs the robot with a fixed procedural gait.

    Purpose:
    - Verify gait works WITHOUT RL
    - Debug contact patterns
    - Tune amplitudes manually
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    # REDUCE damping for crawling — stand values are too stiff
    for i in range(6, model.nv):
        model.dof_damping[i] = 0.5  # was 3.0 — way too high for motion

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)

    # Lower gains for crawling — stand gains kill motion
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 1.0   # was 3.0 — too stiff for motion
        kd[i] = kp[i] * 0.03  # was 0.3 — less damping
    
    spawn_h = find_spawn_height(model, data, home_qpos)

    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, spawn_h]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0

    # Settle with stiff gains, then switch to soft for crawling
    kp_stiff = kp * 3.0
    kd_stiff = kd * 3.0
    gentle_gravity_settle(model, data, home_qpos, kp_stiff, kd_stiff, max_tau, duration=2.0)

    start_x = data.qpos[0]
    
    # Print diagnostic: home pose values for active joints
    print(f"\nSine wave gait — settled h={data.qpos[2]:.3f}m")
    print(f"  freq={args.freq}Hz  pitch_amp={args.pitch_amp}  bend_amp={args.bend_amp}")
    print(f"\n  Active joint home values:")
    for name in ACTIVE_JOINTS:
        qa = joint_map[name]
        print(f"    {name:30s}: home={home_qpos[qa]:+.3f}  current={data.qpos[qa]:+.3f}")

    params = {
        "frequency": args.freq,
        "pitch_amp": args.pitch_amp,
        "bend_amp": args.bend_amp,
        "pitch_signs": [-1, -1, 1, 1],
        "bend_signs": [1, 1, 1, 1],
    }

    phase = 0.0
    print(f"\n  Watching for limb motion...\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t0 = time.time()

                q_target = generate_triangle_targets(home_qpos, joint_map, phase, params)
                q_target[0:7] = data.qpos[0:7]

                # Print what offsets are being applied (first few seconds)
                if step % 250 == 0 and step < 3000:
                    print(f"  step={step:5d}  phase={phase:.2f}")
                    for name in ACTIVE_JOINTS:
                        qa = joint_map[name]
                        offset = q_target[qa] - home_qpos[qa]
                        actual = data.qpos[qa]
                        error = q_target[qa] - actual
                        if abs(offset) > 0.001:
                            print(f"    {name:30s}: offset={offset:+.3f}  "
                                  f"target={q_target[qa]:+.3f}  actual={actual:+.3f}  "
                                  f"error={error:+.3f}")

                pd_control_all(model, data, q_target, kp, kd, max_tau)
                mujoco.mj_step(model, data)
                viewer.sync()

                phase += 2 * np.pi * params["frequency"] * 0.002
                step += 1

                if step % 500 == 0:
                    h = data.qpos[2]
                    progress = data.qpos[0] - start_x
                    contacts = get_ee_contacts(model, data)
                    torso = "⚠TRS" if get_torso_ground(model, data) else ""
                    print(f"\n  t={step*0.002:5.1f}s  h={h:.3f}  prog={progress:+.3f}m  "
                          f"vel={data.qvel[0]:+.3f}  EE={int(np.sum(contacts))}/4  {torso}")

                time.sleep(max(0, 0.002 - (time.time() - t0)))
        except KeyboardInterrupt:
            print(f"\nFinal progress: {data.qpos[0] - start_x:+.3f}m")

# ──────────────────────────────────────
# Mode: TUNE — slider GUI
# ──────────────────────────────────────

class GaitParams:
    def __init__(self):
        self.frequency = 0.6
        self.pitch_amp = 0.15
        self.bend_amp = 0.15
        self.pitch_signs = [-1.0, -1.0, 1.0, 1.0]
        self.bend_signs = [1.0, 1.0, 1.0, 1.0]

    def as_dict(self):
        return {
            "frequency": self.frequency,
            "pitch_amp": self.pitch_amp,
            "bend_amp": self.bend_amp,
            "pitch_signs": self.pitch_signs,
            "bend_signs": self.bend_signs,
        }

def start_tune_gui(params):
    import tkinter as tk
    root = tk.Tk()
    root.title("Gait Tuner — 8 DOF")
    root.configure(bg="#1a1a1a")

    def slider(parent, label, from_, to, res, init, cb):
        tk.Label(parent, text=label, fg="white", bg="#1a1a1a",
                 font=("Arial", 9)).pack(anchor="w")
        s = tk.Scale(parent, from_=from_, to=to, resolution=res,
                     orient="horizontal", length=250,
                     bg="#2d2d2d", fg="white", troughcolor="#404040",
                     highlightbackground="#1a1a1a", command=cb)
        s.set(init)
        s.pack(fill="x", padx=5)
        return s

    gf = tk.LabelFrame(root, text=" Global ", fg="#22c55e", bg="#1a1a1a",
                        font=("Arial", 11, "bold"), padx=5, pady=3)
    gf.pack(padx=8, pady=3, fill="x")

    slider(gf, "Frequency (Hz)", 0.1, 2.0, 0.05, params.frequency,
           lambda v: setattr(params, 'frequency', float(v)))
    slider(gf, "Pitch Amplitude", 0.0, 0.5, 0.01, params.pitch_amp,
           lambda v: setattr(params, 'pitch_amp', float(v)))
    slider(gf, "Bend Amplitude", 0.0, 0.5, 0.01, params.bend_amp,
           lambda v: setattr(params, 'bend_amp', float(v)))

    colors = ["#3b82f6", "#eab308", "#ef4444", "#a855f7"]
    labels = ["Left Leg", "Right Leg", "Left Arm", "Right Arm"]
    for idx in range(4):
        lf = tk.LabelFrame(root, text=f" {labels[idx]} ", fg=colors[idx],
                            bg="#1a1a1a", font=("Arial", 10, "bold"), padx=5, pady=2)
        lf.pack(padx=8, pady=2, fill="x")
        def mk_ps(i):
            return lambda v: params.pitch_signs.__setitem__(i, float(v))
        def mk_bs(i):
            return lambda v: params.bend_signs.__setitem__(i, float(v))
        slider(lf, "Pitch sign", -1.0, 1.0, 0.5, params.pitch_signs[idx], mk_ps(idx))
        slider(lf, "Bend sign", -1.0, 1.0, 0.5, params.bend_signs[idx], mk_bs(idx))

    root.mainloop()

def run_tune(args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    # Lower damping for crawling
    for i in range(6, model.nv):
        model.dof_damping[i] = 0.5

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)
    _, _, max_tau = build_gains(model)

    # Stiff for settle, soft for crawl
    kp_stiff = np.zeros(model.nu)
    kd_stiff = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        kp_stiff[i] = frc * 3.0
        kd_stiff[i] = kp_stiff[i] * 0.3
        kp[i] = frc * 1.0
        kd[i] = kp[i] * 0.1

    spawn_h = find_spawn_height(model, data, home_qpos)

    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, spawn_h]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    gentle_gravity_settle(model, data, home_qpos, kp_stiff, kd_stiff, max_tau, duration=2.0)

    params = GaitParams()
    threading.Thread(target=start_tune_gui, args=(params,), daemon=True).start()

    start_x = data.qpos[0]
    phase = 0.0
    print("Tune mode — adjust sliders to find working gait\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t0 = time.time()
                q_target = generate_triangle_targets(home_qpos, joint_map, phase, params.as_dict())
                q_target[0:7] = data.qpos[0:7]
                pd_control_all(model, data, q_target, kp, kd, max_tau)
                mujoco.mj_step(model, data)
                viewer.sync()

                phase += 2 * np.pi * params.frequency * 0.002
                step += 1

                if step % 500 == 0:
                    prog = data.qpos[0] - start_x
                    contacts = get_ee_contacts(model, data)
                    print(f"  t={step*0.002:5.1f}s  prog={prog:+.3f}m  "
                          f"vel={data.qvel[0]:+.3f}  h={data.qpos[2]:.3f}  "
                          f"EE={int(np.sum(contacts))}/4  "
                          f"f={params.frequency:.2f}  pa={params.pitch_amp:.2f}  "
                          f"ba={params.bend_amp:.2f}")

                time.sleep(max(0, 0.002 - (time.time() - t0)))
        except KeyboardInterrupt:
            print(f"\nBest params found:")
            print(f"  frequency: {params.frequency}")
            print(f"  pitch_amp: {params.pitch_amp}")
            print(f"  bend_amp: {params.bend_amp}")
            print(f"  pitch_signs: {params.pitch_signs}")
            print(f"  bend_signs: {params.bend_signs}")

# ──────────────────────────────────────
# RL Environment — 8 DOF
# ──────────────────────────────────────

class G1CrawlEnv(gym.Env):
    """
    RL ENVIRONMENT

    Key Design:
    - Base motion = triangle gait (hardcoded)
    - RL adds residual corrections

    Why?
    → Pure RL fails (too unstable)
    → IK + RL conflicts
    → Residual RL = best of both worlds

    Action:
        8D vector:
        [pitch offsets (4 limbs), bend offsets (4 limbs)]

    Observation:
        - orientation
        - velocities
        - joint states
        - contact info
        - phase clock
        - height
    """
    metadata = {"render_modes": ["human"], "render_fps": 50}

    RESIDUAL_SCALE = 0.3
    N_SUBSTEPS = 10
    MAX_STEPS = 1500
    BLEND_STEPS = 100

    DEFAULT_PARAMS = {
        "frequency": 1.2,
        "pitch_amp": 0.4,
        "bend_amp": 0.25,
        "pitch_signs": [-1.0, -1.0, 1.0, 1.0],
        "bend_signs": [1.0, 1.0, 1.0, 1.0],
    }

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        # Lower damping for crawling (stand used 3.0)
        for i in range(6, self.model.nv):
            self.model.dof_damping[i] = 0.5

        self.joint_map = get_joint_map(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)

        # Two gain sets: stiff for settling, soft for crawling
        self.kp_stiff, self.kd_stiff, self.max_tau = build_gains(self.model)
        self.kp = np.zeros(self.model.nu)
        self.kd = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            frc = TORQUE_LIMITS.get(name, 25.0)
            self.kp[i] = frc * 1.0
            self.kd[i] = self.kp[i] * 0.03
        self.spawn_h = find_spawn_height(self.model, self.data, self.home_qpos)
        self.contact_history = np.zeros((4, 50))  # 50-step window
        self.hist_idx = 0

        # Map active joint names to qpos addresses
        self.active_qpos_map = {}
        for name in ACTIVE_JOINTS:
            if name in self.joint_map:
                self.active_qpos_map[name] = self.joint_map[name]

        # 8D action: pitch offset (4) + bend offset (4)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

        # Obs: quat(4) + angvel(3) + linvel(3) + 8 joint pos + 8 joint vel
        #      + ee_contacts(4) + clock(2) + height(1)
        obs_dim = 4 + 3 + 3 + 8 + 8 + 4 + 2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(8)
        self.settled_height = 0.1
        self.start_x = 0.0
        self.prev_x = 0.0       # for per-step displacement
        self.prev_vel = 0.0     # for jerk detection
        self.viewer = None

    def _get_active_state(self):
        """Get positions and velocities of 8 active joints."""
        pos = np.zeros(8)
        vel = np.zeros(8)
        for idx, name in enumerate(ACTIVE_JOINTS):
            qa = self.joint_map[name]
            pos[idx] = self.data.qpos[qa] - self.home_qpos[qa]
            # Find dof address
            for i in range(self.model.njnt):
                if self.model.jnt_qposadr[i] == qa:
                    vel[idx] = self.data.qvel[self.model.jnt_dofadr[i]]
                    break
        return pos, vel

    def _get_obs(self):
        d = self.data
        j_pos, j_vel = self._get_active_state()
        contacts = get_ee_contacts(self.model, self.data)
        clock = np.array([np.sin(self.phase), np.cos(self.phase)])
        return np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3],
            j_pos, j_vel,
            contacts, clock, [d.qpos[2]]
        ]).astype(np.float32)

    def _get_reward(self, action):
        """
        Computes the reward signal for crawling behavior.

        DESIGN PHILOSOPHY:
        The reward is heavily shaped to enforce *true crawling* rather than
        degenerate behaviors like jumping, sliding, or vibrating in place.

        The structure is:
            1. PRIMARY OBJECTIVE → forward displacement
            2. GAIT CONSTRAINTS → enforce proper contact patterns
            3. STABILITY TERMS → maintain physically plausible posture
            4. PENALTIES → suppress cheating behaviors (jumping, imbalance, etc.)

        ──────────────────────────────────────
        1. PRIMARY SIGNAL (DOMINANT)
        ──────────────────────────────────────
        - displacement:
            Reward is proportional to forward movement per step.
            Only positive displacement is counted (no backward reward).
            This is scaled VERY HIGH (×200) to dominate learning.

            → Ensures the agent focuses on locomotion above all else.

        ──────────────────────────────────────
        2. GAIT ENFORCEMENT (CRITICAL)
        ──────────────────────────────────────
        - contact count (n_contacts):
            Ideal crawling requires 2–3 limbs on the ground.

        - lift_reward:
            Encourages lifting limbs (prevents all-legs-planted shuffling).
            Hands are weighted higher → encourages active pulling.

        - limb_usage:
            Encourages limbs to alternate between contact and swing.
            Prevents limbs from getting "stuck" permanently.

        - good_contact:
            Reward if 2–3 limbs are in contact (stable crawl regime).

        - all_down_penalty:
            Penalizes all 4 limbs being on ground (no stepping).

        - airborne_penalty:
            Strong penalty if ≤1 contact → prevents jumping/hopping.

        ──────────────────────────────────────
        3. STABILITY & POSTURE
        ──────────────────────────────────────
        - height reward (h_rew):
            Keeps torso near nominal settled height.

        - belly_up:
            Encourages correct orientation (belly-up crawling).

        - alignment:
            Encourages forward-facing orientation.

        - torso_bad:
            Penalizes torso/pelvis touching ground.

        ──────────────────────────────────────
        4. MOTION QUALITY PENALTIES
        ──────────────────────────────────────
        - z_penalty:
            Penalizes vertical velocity → suppresses jumping.

        - vel_jerk:
            Penalizes sudden velocity changes → smoother motion.

        - lat_vel, yaw_rate:
            Penalize sideways drift and spinning.

        - energy:
            Penalizes excessive torque usage.

        - act_rate:
            Penalizes rapid action changes (encourages smooth control).

        ──────────────────────────────────────
        5. SYMMETRY & BALANCE
        ──────────────────────────────────────
        - lr_sym:
            Penalizes left-right asymmetry.

        - diag_sym:
            Penalizes breaking diagonal coordination (important for crawl gait).

        - imbalance_penalty:
            Penalizes uneven load between arms and legs.

        ──────────────────────────────────────
        6. TEMPORAL CONTACT CONSISTENCY
        ──────────────────────────────────────
        - history_penalty:
            Based on contact history variance per limb.
            Encourages balanced usage across all limbs over time.

        ──────────────────────────────────────
        OVERALL EFFECT:
        ──────────────────────────────────────
        The reward strongly pushes the agent to:
        ✔ Move forward continuously
        ✔ Maintain 2–3 contacts (crawl regime)
        ✔ Lift limbs and alternate properly
        ✔ Stay low and stable
        ✔ Avoid jumping or exploiting physics

        While penalizing:
        ✘ Jumping / hopping
        ✘ Sliding with all limbs down
        ✘ Imbalanced or asymmetric gaits
        ✘ High-energy or jerky motion

        RETURNS:
            reward (float):
                Scalar reward for current step

            info (dict):
                Debug information including:
                    - forward velocity
                    - displacement (mm)
                    - height
                    - number of contacts
                    - gait indicators (lifting, all_down)
                    - torso contact status
                    - alignment metric
        """

        d = self.data

        # ═══════════════════════════════════════
        # PRIMARY: ACTUAL DISPLACEMENT THIS STEP
        # ═══════════════════════════════════════
        step_dx = d.qpos[0] - self.prev_x
        # Only positive displacement counts, capped
        displacement = max(0.0, step_dx)

        # ═══════════════════════════════════════
        # LIMB LIFTING — must lift at least 1 EE
        # ═══════════════════════════════════════
        contacts = get_ee_contacts(self.model, self.data)
        n_contacts = int(np.sum(contacts))
        feet_contacts = contacts[0:2]
        hand_contacts = contacts[2:4]

        feet_lift = 2 - np.sum(feet_contacts)
        hands_lift = 2 - np.sum(hand_contacts)

        # EE heights — check if any foot is actually lifting
        ee_max_h = 0.0
        for j in range(4):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITES[j])
            ee_h = self.data.site_xpos[sid][2]
            ee_max_h = max(ee_max_h, ee_h)

        ee_vels = []

        # Reward: at least 1 limb in the air (2-3 on ground)
        lifting = 1.0 if n_contacts <= 3 and ee_max_h > 0.03 else 0.0

        # --- Lifting rewards ---
        lift_reward = (
            2.0 * feet_lift +
            4.0 * hands_lift
        )

        # Penalty: all 4 feet planted = dancing in place, not walking
        all_down_penalty = 1.0 if n_contacts == 4 else 0.0

        # --- Diagonal gait ---
        diag1 = (contacts[0] == 0 and contacts[3] == 0)
        diag2 = (contacts[1] == 0 and contacts[2] == 0)
        # diag_reward = 2.0 * (diag1 + diag2)

        # ═══════════════════════════════════════
        # SECONDARY: stability stuff
        # ═══════════════════════════════════════
        h = d.qpos[2]
        h_rew = np.exp(-30.0 * (h - self.settled_height) ** 2)

        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        R = d.xmat[pid].reshape(3, 3)
        belly_up = max(0.0, -R[2, 2])
        alignment = R[0, 0]

        torso_bad = 1.0 if get_torso_ground(self.model, self.data) else 0.0

        # Jerk: penalize oscillating in place
        fwd_vel = d.qvel[0]
        vel_jerk = abs(fwd_vel - self.prev_vel)

        lat_vel = abs(d.qvel[1])
        yaw_rate = abs(d.qvel[5])
        act_rate = np.sum(np.square(action - self.prev_action))
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        # Encourage each limb to NOT be stuck
        limb_usage = 0.0
        for c in contacts:
            limb_usage += (1.0 - abs(c - 0.5))  # best when switching between 0 and 1

        limb_usage /= 4.0
        # Penalize flight HARD
        airborne_penalty = 1.0 if n_contacts <= 1 else 0.0

        # Ideal crawl: 2–3 contacts
        good_contact = 1.0 if 2 <= n_contacts <= 3 else 0.0

        # Penalize all 4 planted (no movement)
        all_down_penalty = 1.0 if n_contacts == 4 else 0.0

        # Left-right symmetry
        lr_sym = abs(contacts[0] - contacts[1]) + abs(contacts[2] - contacts[3])

        # Diagonal symmetry (natural crawling)
        diag_sym = abs(contacts[0] - contacts[3]) + abs(contacts[1] - contacts[2])

        # Difference between limb groups
        feet_load = np.sum(feet_contacts)
        hand_load = np.sum(hand_contacts)

        imbalance = abs(feet_load - hand_load)  # 0 = perfect balance

        imbalance_penalty = imbalance
        
        # --- vertical motion ---
        z_vel = abs(d.qvel[2])
        z_penalty = z_vel

        contact_usage = np.mean(self.contact_history, axis=1)  # per limb

        usage_balance = np.std(contact_usage)  # low = balanced

        history_penalty = usage_balance

        # ═══════════════════════════════════════
        # REWARD — displacement is king
        # ═══════════════════════════════════════
        reward = (
            # Displacement is the DOMINANT signal
            200.0 * displacement     # 1cm = +2.0 reward. HUGE.

            # Must lift limbs to walk
            + 2.0 * lift_reward          # bonus for having a limb in the air
            + 2.0 * limb_usage

            # + 3.0 * forward_consistency     # limbs moving forward
            # - 4.0 * backward_penalty        # punish backward limbs
            # + 2.0 * consistency

            # Keep body healthy (small weights)
            + 0.5 * h_rew
            + 0.5 * belly_up
            + 0.5 * alignment
            + 0.1                    # alive

                # Contact behavior (VERY IMPORTANT)
            + 3.0 * good_contact
            - 6.0 * airborne_penalty
            - 2.0 * all_down_penalty

            # Kill jumping
            - 4.0 * z_penalty


            # Penalties
            - 2.0 * torso_bad
            - 0.3 * vel_jerk        # don't oscillate
            - 0.1 * lat_vel
            - 0.1 * yaw_rate
            - 0.001 * energy
            - 0.02 * act_rate
            - 2.0 * history_penalty
            - 3.0 * all_down_penalty # strong penalty for all feet planted
            - 2.0 * lr_sym
            - 2.0 * diag_sym
            - 4.0 * imbalance_penalty
        )

        # Update trackers
        self.prev_x = d.qpos[0]
        self.prev_vel = fwd_vel

        return reward, {
            "fwd_vel": float(fwd_vel), "step_dx": float(step_dx * 1000),  # mm
            "displacement_mm": float(displacement * 1000),
            "height": float(h), "n_contacts": n_contacts,
            "lifting": bool(lifting), "all_down": bool(all_down_penalty),
            "torso_ground": bool(torso_bad), "alignment": float(alignment),
        }

    def _is_terminated(self):
        """
        Determines whether the current episode should terminate early.

        DESIGN PHILOSOPHY:
        Termination conditions are used to:
        - Prevent the agent from exploring physically invalid states
        - Stop episodes where recovery is unlikely
        - Improve training efficiency by avoiding wasted rollout steps

        ──────────────────────────────────────
        TERMINATION CONDITIONS
        ──────────────────────────────────────

        1. NaN STATE CHECK
            - If any position value becomes NaN, simulation is unstable.
            → Immediate termination to avoid corrupt gradients.

        2. COLLAPSE (TOO LOW)
            - If base height drops below threshold (~2 cm)
            → Robot has collapsed onto the ground.

        3. LAUNCH (TOO HIGH)
            - If base height exceeds threshold (~0.6 m)
            → Robot is jumping or has become unstable.

        4. FLIP DETECTION
            - Uses pelvis orientation matrix
            - belly = -R[2,2]
            → If too negative → robot flipped incorrectly

        ──────────────────────────────────────
        WHY THIS MATTERS
        ──────────────────────────────────────
        Without these:
        ✘ RL will exploit unstable physics
        ✘ Episodes waste time in unrecoverable states
        ✘ Training becomes noisy and slow

        RETURNS:
            terminated (bool):
                True if episode should end

            reason (str):
                Human-readable termination reason for debugging/logging
        """
        d = self.data
        if np.any(np.isnan(d.qpos)):
            return True, "NaN"
        h = d.qpos[2]
        if h < 0.02:
            return True, f"collapsed h={h:.3f}"
        if h > 0.60:
            return True, f"launched h={h:.3f}"
        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        belly = -d.xmat[pid].reshape(3, 3)[2, 2]
        if belly < -0.2:
            return True, f"flipped belly={belly:.2f}"
        return False, ""

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial stable crawling state.

        DESIGN PHILOSOPHY:
        The reset is carefully designed to:
        - Start from a consistent, physically valid configuration
        - Avoid immediate instability (no sudden gravity shocks)
        - Provide slight randomness for generalization

        ──────────────────────────────────────
        RESET PROCEDURE
        ──────────────────────────────────────

        1. RESET SIMULATION STATE
            - Clears MuJoCo internal buffers and state variables

        2. INITIALIZE POSE
            - Set joint positions to predefined "home" pose
            - Place robot at computed spawn height (just above ground)
            - Set orientation to belly-up configuration

        3. RANDOMIZATION (DOMAIN RANDOMIZATION)
            - Small noise added to joint positions
            → Prevents overfitting to exact initial state

        4. ZERO VELOCITIES
            - Ensures no initial momentum

        5. GRAVITY SETTLING (CRITICAL)
            - Gradually introduces gravity over time
            - Uses stiff PD gains to stabilize robot

            WHY:
            ✘ Without this → robot collapses violently
            ✔ With this → stable starting configuration

        6. TRACKING VARIABLES RESET
            - settled_height → reference for height reward
            - start_x / prev_x → displacement tracking
            - prev_vel → jerk computation
            - phase → gait phase reset
            - prev_action → smoothness tracking

        ──────────────────────────────────────
        WHY THIS IS IMPORTANT
        ──────────────────────────────────────
        A bad reset leads to:
        ✘ noisy reward signals
        ✘ unstable early steps
        ✘ poor RL convergence

        RETURNS:
            observation (np.ndarray):
                Initial observation vector

            info (dict):
                Empty dictionary (standard Gym API)
        """
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, self.spawn_h]
        self.data.qpos[3:7] = BELLY_UP_QUAT
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.01, 0.01,
                                                          size=self.data.qpos[7:].shape)
        self.data.qvel[:] = 0

        gentle_gravity_settle(self.model, self.data, self.home_qpos,
                              self.kp_stiff, self.kd_stiff, self.max_tau, duration=2.0)

        self.settled_height = self.data.qpos[2]
        self.start_x = self.data.qpos[0]
        self.prev_x = self.data.qpos[0]
        self.prev_vel = 0.0
        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(8)

        return self._get_obs(), {}

    def step(self, action):
        """
        Executes one environment step.

        DESIGN PHILOSOPHY:
        This is a HYBRID CONTROL pipeline:
            Base controller → procedural gait (triangle wave)
            RL agent → residual corrections

        This avoids:
            ✘ learning locomotion from scratch (too hard)
            ✘ instability from pure RL

        ──────────────────────────────────────
        STEP PIPELINE
        ──────────────────────────────────────

        1. ACTION PROCESSING
            - Clip action to valid range [-1, 1]
            - Action space:
                [0:4] → pitch corrections
                [4:8] → bend corrections

        2. BASE GAIT GENERATION
            - Uses triangle wave generator
            - Produces structured crawling motion

        3. RESIDUAL RL CORRECTIONS
            - RL modifies joint targets on top of base gait
            - Scaled by RESIDUAL_SCALE

            WHY:
            ✔ Keeps motion stable
            ✔ Allows RL to refine instead of invent

        4. JOINT LIMIT ENFORCEMENT
            - Ensures targets stay within physical limits
            - Prevents unrealistic or unsafe configurations

        5. PHYSICS SIMULATION
            - Runs multiple substeps (N_SUBSTEPS)
            → Improves stability and control fidelity

            Each substep:
                - Apply PD control
                - Step MuJoCo simulation

        6. PHASE UPDATE
            - Advances gait cycle based on frequency

        7. REWARD COMPUTATION
            - Calls _get_reward() for shaped reward

        8. TERMINATION CHECK
            - Calls _is_terminated()

        9. CONTACT HISTORY UPDATE
            - Maintains rolling window of contact states
            → used for gait regularization

        ──────────────────────────────────────
        WHY THIS DESIGN WORKS
        ──────────────────────────────────────
        Pure RL:
            ✘ unstable, learns jumping

        Pure hand-designed gait:
            ✘ limited performance

        Hybrid:
            ✔ stable
            ✔ learnable
            ✔ high performance

        ──────────────────────────────────────
        RETURNS:
            observation (np.ndarray):
                Next state observation

            reward (float):
                Scalar reward

            terminated (bool):
                True if episode ended due to failure

            truncated (bool):
                True if episode ended due to time limit

            info (dict):
                Debug information including termination reason
        """
        action = np.clip(action, -1.0, 1.0)

        # Base gait from sine wave
        q_target = generate_triangle_targets(
            self.home_qpos, self.joint_map, self.phase, self.DEFAULT_PARAMS)
        q_target[0:7] = self.data.qpos[0:7]

        # Add RL corrections on top of sine targets
        # action[0:4] = pitch corrections, action[4:8] = bend corrections
        for i, limb_name in enumerate(LIMB_NAMES):
            pitch_joint, bend_joint = LIMBS[limb_name]
            if pitch_joint in self.joint_map:
                qa = self.joint_map[pitch_joint]
                q_target[qa] += action[i] * self.RESIDUAL_SCALE
            if bend_joint in self.joint_map:
                qa = self.joint_map[bend_joint]
                q_target[qa] += action[4 + i] * self.RESIDUAL_SCALE

        # Clamp to joint limits
        for name in ACTIVE_JOINTS:
            qa = self.joint_map[name]
            for j in range(self.model.njnt):
                if self.model.jnt_qposadr[j] == qa:
                    lo, hi = self.model.jnt_range[j]
                    if lo < hi:
                        q_target[qa] = np.clip(q_target[qa], lo, hi)
                    break

        for _ in range(self.N_SUBSTEPS):
            pd_control_all(self.model, self.data, q_target, self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        self.phase += 2 * np.pi * self.DEFAULT_PARAMS["frequency"] * 0.02
        self.step_count += 1

        reward, info = self._get_reward(action)
        term, reason = self._is_terminated()
        if term:
            info["term_reason"] = reason
        trunc = self.step_count >= self.MAX_STEPS

        self.prev_action = action.copy()


        self.contact_history[:, self.hist_idx] = get_ee_contacts(self.model, self.data)
        self.hist_idx = (self.hist_idx + 1) % 50

        return self._get_obs(), reward, term, trunc, info

    def render(self):
        """
        Renders the simulation using MuJoCo viewer.

        DESIGN PHILOSOPHY:
        - Lazy initialization → viewer is only created when needed
        - Supports interactive visualization for debugging

        ──────────────────────────────────────
        BEHAVIOR
        ──────────────────────────────────────

        1. VIEWER CREATION
            - If viewer does not exist AND render_mode == "human"
            → launch passive MuJoCo viewer

        2. FRAME UPDATE
            - Syncs viewer with current simulation state

        ──────────────────────────────────────
        WHY THIS IS IMPORTANT
        ──────────────────────────────────────
        Visualization is critical for:
        ✔ debugging reward issues
        ✔ understanding gait failures
        ✔ detecting unwanted behaviors (jumping, slipping)

        NOTE:
        - This does NOT affect training (unless rendering enabled)
        - Mainly used in eval/debug modes
        """
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

# ──────────────────────────────────────
# Train / Eval / Test
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = SubprocVecEnv([lambda: G1CrawlEnv() for _ in range(args.n_envs)])
    eval_env = G1CrawlEnv()

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        model = PPO.load(args.checkpoint, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, verbose=1, device=args.device,
            tensorboard_log="logs/",
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-1.0,
            ),
        )

    print(f"\n8-DOF Crawl Training")
    print(f"  Action: 8D (4 pitch + 4 bend corrections)")
    print(f"  Base gait: built-in sine wave")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs}\n")

    model.learn(
        total_timesteps=args.steps,
        callback=[
            CheckpointCallback(save_freq=max(200_000 // args.n_envs, 1),
                               save_path="models/", name_prefix="g1_crawl"),
            EvalCallback(eval_env, best_model_save_path="models/best/",
                         log_path="logs/eval/",
                         eval_freq=max(50_000 // args.n_envs, 1),
                         n_eval_episodes=5, deterministic=True),
        ],
        progress_bar=True,
    )
    model.save("models/g1_crawl_final")
    env.close()

def evaluate(args):
    from stable_baselines3 import PPO
    if not args.checkpoint:
        print("Error: --checkpoint required"); return
    model = PPO.load(args.checkpoint)
    env = G1CrawlEnv(render_mode="human")
    obs, _ = env.reset()
    total, steps, episodes = 0, 0, 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            total += rew; steps += 1
            env.render(); time.sleep(0.02)
            if steps % 50 == 0:
                print(f"  step={steps:4d}  rew={total:.1f}  fwd={info['fwd_vel']:+.3f}  "
                      f"h={info['height']:.3f}  EE={info['n_contacts']}/4  ")
                      # f"prog={info['progress']:+.3f}m")
            if term or trunc:
                episodes += 1; r = info.get("term_reason", "truncated")
                # print(f"\n  Ep {episodes} [{r}] steps={steps} prog={'progress':+.3f}m\n")
                print(f"\n  Ep {episodes} [{r}] steps={steps} prog={info.get('progress', 0):+.3f}m\n")
                total, steps = 0, 0; obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

def test_random(args):
    env = G1CrawlEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"Random test — 8D action, settled h={env.settled_height:.3f}m\n")
    total, steps, ep = 0, 0, 0
    ep_lens, ep_rews = [], []
    try:
        while True:
            obs, rew, term, trunc, info = env.step(env.action_space.sample() * 0.3)
            total += rew; steps += 1; env.render()
            if steps % 100 == 0:
                ps = total / max(steps, 1)
                print(f"  step={steps:4d}  rew/s={ps:+.2f}  fwd={info['fwd_vel']:+.3f}  "
                      f"h={info['height']:.3f}  EE={info['n_contacts']}/4  ")
                      # f"prog={info['progress']:+.3f}")
            if term or trunc:
                ep += 1; ep_lens.append(steps); ep_rews.append(total)
                r = info.get("term_reason", "truncated")
                v = "✅" if steps >= 500 and total/max(steps,1) > 0 else "❌"
                print(f"  Ep {ep} [{r}] steps={steps} rew/s={total/max(steps,1):+.2f} {v}")
                if ep >= 3:
                    al, ar = np.mean(ep_lens), np.mean([r/max(l,1) for r,l in zip(ep_rews,ep_lens)])
                    print(f"\n  Summary: avg_len={al:.0f} avg_rew/s={ar:+.2f} "
                          f"→ {'✅ READY' if ar > 0 and al > 300 else '❌ NOT READY'}\n")
                total, steps = 0, 0; obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# Mode: SEARCH — try all sign combos
# ──────────────────────────────────────

def run_search(args):
    """
    Perform a brute-force search over predefined gait parameter combinations to identify
    effective crawling behaviors for the robot.

    This function systematically evaluates different combinations of:
    - Pitch direction signs for each limb ([-1, 0, +1]^4)
    - Bend direction patterns (predefined symmetric/asymmetric sets)
    - Oscillation amplitudes (pitch and bend)
    - Gait frequencies

    The search is executed in two stages:

    1. Coarse Search (Sign Combinations):
        - Iterates over all pitch_sign and bend_sign combinations.
        - Runs a short simulation (~5 seconds) for each configuration.
        - Evaluates forward displacement as the primary metric.
        - Tracks stability (alive/dead), lateral drift, and final height.
        - Ranks configurations based on forward progress.

    2. Fine Search (Amplitude & Frequency Tuning):
        - Uses the best sign configuration from the coarse search.
        - Sweeps over different pitch amplitudes, bend amplitudes, and frequencies.
        - Again evaluates forward progress and stability.
        - Identifies the best overall gait parameters.

    Simulation Details:
        - Robot is reset to a belly-up crawling pose before each trial.
        - A short "settling phase" with stiff PD gains stabilizes the robot.
        - Crawling is driven by a triangle-wave gait generator.
        - PD control tracks joint targets at each timestep.
        - Episodes terminate early if instability is detected (NaNs, collapse, or excessive height).

    Metrics Collected:
        - progress: forward displacement along x-axis
        - lat_drift: deviation along y-axis
        - final_h: final height of the robot base
        - alive: whether the robot remained stable during rollout

    Output:
        - Prints ranked results of top-performing configurations.
        - Displays the best pitch/bend sign combination.
        - Reports optimal amplitude and frequency settings.
        - Suggests parameters to plug into DEFAULT_PARAMS for further testing.

    Args:
        args: Command-line or configuration arguments (not directly used here,
              but included for compatibility with the overall script structure).

    Returns:
        None
    """    
    import itertools

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    # Lower damping for crawling
    for i in range(6, model.nv):
        model.dof_damping[i] = 0.5

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)

    # Two gain sets: stiff for settling, soft for crawling
    _, _, max_tau = build_gains(model)
    kp_stiff = np.zeros(model.nu)
    kd_stiff = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        kp_stiff[i] = frc * 3.0
        kd_stiff[i] = kp_stiff[i] * 0.3
        kp[i] = frc * 1.0
        kd[i] = kp[i] * 0.1

    spawn_h = find_spawn_height(model, data, home_qpos)

    # Try all 81 pitch_sign combos (each limb: -1, 0, +1)
    sign_values = [-1.0, 0.0, 1.0]
    pitch_combos = list(itertools.product(sign_values, repeat=4))
    bend_combos = [[1, 1, 1, 1], [1, 1, -1, -1], [-1, -1, 1, 1], [-1, -1, -1, -1]]

    amps = [0.15, 0.25, 0.35, 0.45]
    freqs = [0.3, 0.5, 0.7, 1.0]

    # Quick search: just pitch_signs with fixed bend/amp/freq
    print(f"Searching {len(pitch_combos)} pitch_sign combos × {len(bend_combos)} bend_sign combos")
    print(f"Each trial: 5 seconds of simulation\n")

    SIM_DURATION = 5.0
    SIM_STEPS = int(SIM_DURATION / model.opt.timestep)

    results = []
    total = len(pitch_combos) * len(bend_combos)
    done = 0

    for ps in pitch_combos:
        for bs in bend_combos:
            # Reset
            mujoco.mj_resetData(model, data)
            data.qpos[:] = home_qpos
            data.qpos[0:3] = [0, 0, spawn_h]
            data.qpos[3:7] = BELLY_UP_QUAT
            data.qvel[:] = 0

            # Settle with stiff gains
            gentle_gravity_settle(model, data, home_qpos, kp_stiff, kd_stiff, max_tau, duration=1.0)
            start_x = data.qpos[0]
            start_h = data.qpos[2]

            params = {
                "frequency": 0.5,
                "pitch_amp": 0.3,
                "bend_amp": 0.2,
                "pitch_signs": list(ps),
                "bend_signs": list(bs),
            }

            phase = 0.0
            alive = True
            for step in range(SIM_STEPS):
                q_target = generate_triangle_targets(home_qpos, joint_map, phase, params)
                q_target[0:7] = data.qpos[0:7]
                pd_control_all(model, data, q_target, kp, kd, max_tau)
                mujoco.mj_step(model, data)
                phase += 2 * np.pi * params["frequency"] * model.opt.timestep

                if np.any(np.isnan(data.qpos)) or data.qpos[2] < 0.01 or data.qpos[2] > 0.6:
                    alive = False
                    break

            progress = data.qpos[0] - start_x
            lat_drift = abs(data.qpos[1])
            final_h = data.qpos[2]

            results.append({
                "pitch_signs": list(ps),
                "bend_signs": list(bs),
                "progress": progress,
                "lat_drift": lat_drift,
                "final_h": final_h,
                "alive": alive,
            })

            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} tested...")

    # Sort by forward progress
    results.sort(key=lambda r: r["progress"], reverse=True)

    print(f"\n{'rank':>4}  {'progress':>9}  {'lat':>6}  {'h':>6}  {'alive':>5}  "
          f"{'pitch_signs':>20}  {'bend_signs':>20}")
    print("-" * 85)
    for i, r in enumerate(results[:20]):
        a = "✅" if r["alive"] else "❌"
        print(f"  {i+1:3d}  {r['progress']:+9.4f}m  {r['lat_drift']:6.3f}  "
              f"{r['final_h']:6.3f}  {a}  "
              f"{r['pitch_signs']}  {r['bend_signs']}")

    best = results[0]
    print(f"\n  Best: pitch_signs={best['pitch_signs']}, bend_signs={best['bend_signs']}")
    print(f"        progress={best['progress']:+.4f}m, alive={best['alive']}")

    # Now try different amplitudes and frequencies with the best signs
    print(f"\n  Tuning amp/freq with best signs...\n")

    fine_results = []
    for amp_p in amps:
        for amp_b in amps:
            for freq in freqs:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = home_qpos
                data.qpos[0:3] = [0, 0, spawn_h]
                data.qpos[3:7] = BELLY_UP_QUAT
                data.qvel[:] = 0
                gentle_gravity_settle(model, data, home_qpos, kp_stiff, kd_stiff, max_tau, duration=1.0)
                start_x = data.qpos[0]

                params = {
                    "frequency": freq,
                    "pitch_amp": amp_p,
                    "bend_amp": amp_b,
                    "pitch_signs": best["pitch_signs"],
                    "bend_signs": best["bend_signs"],
                }

                phase = 0.0
                alive = True
                for step in range(SIM_STEPS):
                    q_target = generate_triangle_targets(home_qpos, joint_map, phase, params)
                    q_target[0:7] = data.qpos[0:7]
                    pd_control_all(model, data, q_target, kp, kd, max_tau)
                    mujoco.mj_step(model, data)
                    phase += 2 * np.pi * freq * model.opt.timestep
                    if np.any(np.isnan(data.qpos)) or data.qpos[2] < 0.01:
                        alive = False; break

                progress = data.qpos[0] - start_x
                fine_results.append({
                    "freq": freq, "pitch_amp": amp_p, "bend_amp": amp_b,
                    "progress": progress, "alive": alive,
                })

    fine_results.sort(key=lambda r: r["progress"], reverse=True)
    print(f"  {'rank':>4}  {'freq':>5}  {'p_amp':>5}  {'b_amp':>5}  {'progress':>9}  {'alive':>5}")
    print("  " + "-" * 45)
    for i, r in enumerate(fine_results[:10]):
        a = "✅" if r["alive"] else "❌"
        print(f"  {i+1:3d}  {r['freq']:5.2f}  {r['pitch_amp']:5.2f}  "
              f"{r['bend_amp']:5.2f}  {r['progress']:+9.4f}m  {a}")

    top = fine_results[0]
    print(f"\n  === BEST OVERALL ===")
    print(f"  pitch_signs = {best['pitch_signs']}")
    print(f"  bend_signs  = {best['bend_signs']}")
    print(f"  frequency   = {top['freq']}")
    print(f"  pitch_amp   = {top['pitch_amp']}")
    print(f"  bend_amp    = {top['bend_amp']}")
    print(f"  progress    = {top['progress']:+.4f}m in {SIM_DURATION}s")
    print(f"\n  Copy these into DEFAULT_PARAMS in the code, then run --mode sine to verify.")


def record_video_from_checkpoint(checkpoint_path, output="rollout.mp4", steps=500, fps=60):
    """
    Runs a policy from checkpoint and saves an MP4 video.
    """

    import imageio
    from stable_baselines3 import PPO

    # Load env + model
    env = G1CrawlEnv(render_mode=None)
    model = PPO.load(checkpoint_path)

    obs, _ = env.reset()

    model_mj = env.model
    data = env.data

    cam = mujoco.MjvCamera()

    # Zoom OUT (higher distance = more zoomed out)
    cam.distance = 4.0        # try 3.0–6.0 depending on how wide you want
    cam.azimuth = 90          # side view (0 = front, 90 = side)
    cam.elevation = -20       # slight downward tilt
    cam.lookat[:] = [0, 0, 0.2]  # center point (slightly above ground)

    # Offscreen renderer
    renderer = mujoco.Renderer(model_mj, width=640, height=480)

    frames = []

    print(f"Recording {steps} steps to {output}...")

    FRAME_SKIP = int(1 / (fps * model_mj.opt.timestep)) 

    for step in range(steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)

        renderer.update_scene(
            data,
            camera=cam
        )        
        # Render frame
        frame = renderer.render()  # RGB array
        import cv2
        cv2.putText(frame, f"step {step}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        frames.append(frame)

        if step % FRAME_SKIP == 0:
            # renderer.update_scene(data)
            frames.append(renderer.render())


        frames.append(frame)

        if term or trunc:
            obs, _ = env.reset()

    renderer.close()

    # Save video
    imageio.mimsave(output, frames, fps=fps)

    print(f"Saved video: {output}")


# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl — 8 DOF")
    parser.add_argument("--mode", choices=["sine", "tune", "search", "train", "eval", "test"],
                        required=True,
                        help="sine: watch | tune: sliders | search: find best signs | train/eval/test: RL")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--freq", type=float, default=0.5)
    parser.add_argument("--pitch_amp", type=float, default=0.3)
    parser.add_argument("--bend_amp", type=float, default=0.2)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--name", type=str, default="rollout.mp4")
    args = parser.parse_args()
    if args.mode == "eval" and args.video:
        record_video_from_checkpoint(args.checkpoint, output=args.name)
    else:
        {"sine": run_sine, "tune": run_tune, "search": run_search,
         "train": train, "eval": evaluate, "test": test_random}[args.mode](args)

    # {"sine": run_sine, "tune": run_tune, "search": run_search,
    #  "train": train, "eval": evaluate, "test": test_random}[args.mode](args)

if __name__ == "__main__":
    main()
