#!/usr/bin/env python3
"""
G1 Spider Crawl — Reference Motion Generator

Generates crawling reference trajectories for imitation learning.
All limbs push in the SAME direction (opposite to desired movement).
Saves body-frame joint trajectories that work for any direction.

NOTE: This reference generator matches the 16 active DOF used in g1_crawl.py.
Waist (waist_yaw_joint) and wrists (left/right_wrist_pitch_joint) are excluded
from the actuated set — they are locked at home pose during RL training.

Usage:
    python g1_crawl_ref.py                          # visualise forward crawl
    python g1_crawl_ref.py --dir 0 1                # crawl in +Y direction
    python g1_crawl_ref.py --stride 0.06 --lift 0.05
    python g1_crawl_ref.py --save                   # save forward reference
    python g1_crawl_ref.py --save-all               # save 8 directions
"""

import argparse, time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

EE_SITES  = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]
EE_LABELS = ["LF", "RF", "LH", "RH"]

# ── 16 actuated joints matching ACTIVE_JOINTS in g1_crawl.py ────────────────
# Excluded vs original: waist_yaw_joint, left_wrist_pitch_joint,
#                       right_wrist_pitch_joint
ACTUATED_JOINTS = [
    "left_hip_pitch_joint",   "left_hip_roll_joint",
    "left_hip_yaw_joint",     "left_knee_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",
    "right_hip_yaw_joint",    "right_knee_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
]

# IK degrees-of-freedom per limb — includes passive ankle joints so the IK
# can set a realistic foot pose, but the reference saves only actuated angles.
LIMB_JOINTS = {
    "left_foot_site": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint",
        # "left_ankle_pitch_joint",   # passive — IK only, not actuated
    ],
    "right_foot_site": [
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint",
        # "right_ankle_pitch_joint",  # passive — IK only, not actuated
    ],
    "left_hand_site": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",   "left_elbow_joint",
        # "left_wrist_pitch_joint",   # locked — not actuated in RL
    ],
    "right_hand_site": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",   "right_elbow_joint",
        # "right_wrist_pitch_joint",  # locked — not actuated in RL
    ],
}

# Wave gait: 1 limb swings at a time, 3 on ground
# Order: LF → LH → RF → RH  (cross-body pattern for stability)
PHASE_OFFSETS = [0.0, np.pi, np.pi / 2, 3 * np.pi / 2]
SWING_DUTY    = 0.25   # 25 % swing, 75 % stance

# Crawl height: pelvis z when belly-up on the floor
CRAWL_HEIGHT = 0.25

# ──────────────────────────────────────
# Setup helpers
# ──────────────────────────────────────

def get_joint_map(model):
    jmap = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name:
            jmap[name] = model.jnt_qposadr[i]
    return jmap


def build_limb_dofs(model):
    """Return {site_name: [dof_addresses]} for every IK joint."""
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
        print("No pose file found — using model defaults.")
    return q


def get_actuated_qpos_indices(model):
    """
    Return the qpos address for each of the 16 active joints, in the order
    they appear in ACTUATED_JOINTS (matching ACTIVE_JOINTS in g1_crawl.py).
    Joints not in ACTUATED_JOINTS (waist, wrists) are skipped.
    """
    indices = []
    # Build a name→actuator-index map
    act_name_to_idx = {}
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            act_name_to_idx[name] = i

    for jname in ACTUATED_JOINTS:
        if jname in act_name_to_idx:
            ai  = act_name_to_idx[jname]
            jid = model.actuator_trnid[ai, 0]
            indices.append(model.jnt_qposadr[jid])
        else:
            print(f"  WARNING: actuated joint '{jname}' not found in model")

    return np.array(indices)


# ──────────────────────────────────────
# Per-limb IK
# ──────────────────────────────────────

def ik_solve_limb(model, data, site_name, target_pos, limb_dofs,
                  max_iters=80, step_size=0.5, tol=0.002):
    sid     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    dof_ids = limb_dofs[site_name]
    joint_names = LIMB_JOINTS[site_name]
    nv   = model.nv
    jacp = np.zeros((3, nv))

    for _ in range(max_iters):
        mujoco.mj_forward(model, data)
        err = target_pos - data.site_xpos[sid]
        if np.linalg.norm(err) < tol:
            break
        jacp[:] = 0
        mujoco.mj_jacSite(model, data, jacp, None, sid)
        J   = jacp[:, dof_ids]
        lam = 0.01
        JJT = J @ J.T + lam * np.eye(3)
        dq  = J.T @ np.linalg.solve(JJT, err)
        for i, jname in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            qa  = model.jnt_qposadr[jid]
            data.qpos[qa] += step_size * dq[i]
            lo, hi = model.jnt_range[jid]
            if lo < hi:
                data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)


# ──────────────────────────────────────
# Trajectory: D-shaped hook with wave gait
# ──────────────────────────────────────

def generate_ee_trajectory(home_pos, phase, stride_length, lift_height, move_dir, site_name=None):
    p        = phase % (2.0 * np.pi)
    swing_end = SWING_DUTY * 2.0 * np.pi

    is_swing = p < swing_end

    if is_swing:
        t = p / swing_end
        forward_offset = stride_length * (-1.0 + 2.0 * t)

        # Hands should NOT lift too much
        if site_name in ["left_hand_site", "right_hand_site"]:
            lift = 0.5 * lift_height * np.sin(t * np.pi)
        else:
            lift = lift_height * np.sin(t * np.pi)

    else:
        t = (p - swing_end) / (2.0 * np.pi - swing_end)
        forward_offset = stride_length * (1.0 - 2.0 * t)
        lift = 0.0

    up_dir = np.array([0.0, 0.0, 1.0])
    pos = home_pos + forward_offset * move_dir + lift * up_dir

    # Hand-specific contact shaping
    if site_name in ["left_hand_site", "right_hand_site"]:
        if is_swing:
            pos[2] -= 0.015
        else:
            pos[2] -= 0.04
            pos += 0.01 * move_dir

    return pos


# ──────────────────────────────────────
# Record reference trajectory
# ──────────────────────────────────────

def generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                        move_dir, speed, stride, lift, n_cycles=4):
    """
    Generate one reference trajectory for a given movement direction.

    Saves only the 16 ACTIVE joint angles (in ACTUATED_JOINTS order) so the
    imitation env can directly index them via active_ids.

    Returns dict with:
        times       : [N]         timestamps
        joint_qpos  : [N, 16]     active joint angles per frame
        ee_positions: [N, 4, 3]   end-effector world positions per frame
        move_dir    : [2]
        phase       : [N]
        speed, stride, lift
    """
    # Crawl pose: belly-up at CRAWL_HEIGHT
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0.0, 0.0, CRAWL_HEIGHT]
    data.qpos[3:7] = [0.707, 0.0, -0.707, 0.0]
    data.qvel[:] = 0
    base_qpos = data.qpos[0:7].copy()
    mujoco.mj_forward(model, data)

    # Home EE positions for this pose
    home_positions = {}
    ee_sids = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        home_positions[name] = data.site_xpos[sid].copy()
        ee_sids[name]        = sid

    # 3-D movement direction (XY plane)
    move_dir_3d = np.array([move_dir[0], move_dir[1], 0.0])
    norm = np.linalg.norm(move_dir_3d)
    if norm > 1e-6:
        move_dir_3d /= norm
    else:
        move_dir_3d = np.array([1.0, 0.0, 0.0])

    dt       = 0.02
    duration = n_cycles / speed
    n_steps  = int(duration / dt)

    # Indices into qpos for the 16 active joints
    act_qa = get_actuated_qpos_indices(model)

    times_list      = []
    joint_qpos_list = []
    ee_pos_list     = []
    phase_list      = []

    phase = 0.0

    for step in range(n_steps):
        # Keep base fixed (zero-gravity reference generation)
        data.qpos[0:7] = base_qpos

        for i, name in enumerate(EE_SITES):
            limb_phase = phase + PHASE_OFFSETS[i]
            target = generate_ee_trajectory(
                home_positions[name], limb_phase, stride, lift, move_dir_3d, name
            )
            ik_solve_limb(model, data, name, target, limb_dofs)

        mujoco.mj_forward(model, data)

        times_list.append(step * dt)
        joint_qpos_list.append(data.qpos[act_qa].copy())   # 16 active joints only

        ee_pos = np.zeros((4, 3))
        for j, name in enumerate(EE_SITES):
            ee_pos[j] = data.site_xpos[ee_sids[name]].copy()
        ee_pos_list.append(ee_pos)

        phase_list.append(phase % (2 * np.pi))
        phase += 2.0 * np.pi * speed * dt

    return {
        "times":        np.array(times_list),
        "joint_qpos":   np.array(joint_qpos_list),   # [N, 16]
        "ee_positions": np.array(ee_pos_list),        # [N, 4, 3]
        "move_dir":     np.array(move_dir),
        "phase":        np.array(phase_list),
        "speed":        speed,
        "stride":       stride,
        "lift":         lift,
    }


def save_reference(ref, filename):
    np.savez(filename, **ref)
    print(f"  Saved {len(ref['times'])} frames → {filename}")
    print(f"    Duration: {ref['times'][-1]:.1f}s | "
          f"Dir: [{ref['move_dir'][0]:+.2f},{ref['move_dir'][1]:+.2f}] | "
          f"joint_qpos shape: {ref['joint_qpos'].shape}")


# ──────────────────────────────────────
# Main
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl Reference Motion")
    parser.add_argument("--speed",    type=float, default=0.8,
                        help="gait frequency Hz")
    parser.add_argument("--stride",   type=float, default=0.06,
                        help="stride half-length metres")
    parser.add_argument("--lift",     type=float, default=0.04,
                        help="foot lift height metres")
    parser.add_argument("--dir",      nargs=2, type=float, default=[1.0, 0.0],
                        help="movement direction X Y")
    parser.add_argument("--save",     action="store_true",
                        help="save forward reference")
    parser.add_argument("--save-all", action="store_true",
                        help="save 8 directions + stationary")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)
    model.opt.gravity[:] = 0          # zero-g for kinematic reference generation

    joint_map  = get_joint_map(model)
    limb_dofs  = build_limb_dofs(model)
    home_qpos  = load_pose(model, joint_map)

    # Crawl starting pose
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0.0, 0.0, CRAWL_HEIGHT]
    data.qpos[3:7] = [0.707, 0.0, -0.707, 0.0]
    data.qvel[:] = 0
    base_qpos = data.qpos[0:7].copy()
    mujoco.mj_forward(model, data)

    # Normalise movement direction
    move_dir = np.array(args.dir, dtype=float)
    norm = np.linalg.norm(move_dir)
    if norm > 1e-6:
        move_dir /= norm
    move_dir_3d = np.array([move_dir[0], move_dir[1], 0.0])

    print(f"Spider Crawl Reference (Wave Gait) — 16 DOF")
    print(f"  Active joints   : {len(ACTUATED_JOINTS)}  (waist + wrists locked/excluded)")
    print(f"  Model actuators : {model.nu}")
    print(f"  Speed: {args.speed} Hz | Stride: {args.stride}m | Lift: {args.lift}m")
    print(f"  Movement dir: [{move_dir[0]:.2f}, {move_dir[1]:.2f}]")
    print(f"  Pelvis height: {CRAWL_HEIGHT}m  (contact sphere r=0.04m)")

    print(f"\nActive joints ({len(ACTUATED_JOINTS)}):")
    for i, name in enumerate(ACTUATED_JOINTS):
        print(f"  [{i:2d}] {name}")

    print(f"\nHome EE positions (belly-up pose):")
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid]
        print(f"  {name:20s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

    # ── Save references ──
    if args.save_all:
        print("\nGenerating references for 8 directions + stationary...")
        directions = {
            "forward":   [ 1.0,  0.0],
            "backward":  [-1.0,  0.0],
            "left":      [ 0.0,  1.0],
            "right":     [ 0.0, -1.0],
            "fwd_left":  [ 0.707,  0.707],
            "fwd_right": [ 0.707, -0.707],
            "bwd_left":  [-0.707,  0.707],
            "bwd_right": [-0.707, -0.707],
        }
        for dir_name, d in directions.items():
            print(f"\n  [{dir_name}] ...")
            ref = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                     d, args.speed, args.stride, args.lift)
            save_reference(ref, f"g1_ref_{dir_name}.npz")

        print(f"\n  [stationary] (zero stride) ...")
        ref_idle = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                      [1.0, 0.0], args.speed, 0.0, 0.0)
        save_reference(ref_idle, "g1_ref_stationary.npz")
        print(f"\nSaved {len(directions) + 1} reference files.")

        # Reset for visualisation after saving
        data.qpos[:] = home_qpos
        data.qpos[0:3] = [0.0, 0.0, CRAWL_HEIGHT]
        data.qpos[3:7] = [0.707, 0.0, -0.707, 0.0]
        mujoco.mj_forward(model, data)

    elif args.save:
        print("\nGenerating forward reference...")
        ref = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                 move_dir.tolist(), args.speed, args.stride, args.lift)
        save_reference(ref, "g1_ref_forward.npz")
        data.qpos[:] = home_qpos
        data.qpos[0:3] = [0.0, 0.0, CRAWL_HEIGHT]
        data.qpos[3:7] = [0.707, 0.0, -0.707, 0.0]
        mujoco.mj_forward(model, data)

    # ── Live visualisation ──
    print("\nVisualisation running — Ctrl+C to stop\n")

    # Re-record home EE positions (may have changed during reference generation)
    home_positions = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        home_positions[name] = data.site_xpos[sid].copy()

    phase = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t0 = time.time()

                data.qpos[0:7] = base_qpos   # lock base for visualisation

                for i, name in enumerate(EE_SITES):
                    limb_phase = phase + PHASE_OFFSETS[i]
                    target = generate_ee_trajectory(
                        home_positions[name], limb_phase,
                        args.stride, args.lift, move_dir_3d, name
                    )
                    ik_solve_limb(model, data, name, target, limb_dofs)

                mujoco.mj_forward(model, data)
                viewer.sync()

                phase += 2.0 * np.pi * args.speed * 0.02
                step  += 1

                if step % 50 == 0:
                    status = []
                    for i in range(4):
                        lp   = (phase + PHASE_OFFSETS[i]) % (2 * np.pi)
                        mode = "SWING " if lp < SWING_DUTY * 2 * np.pi else "STANCE"
                        status.append(f"{EE_LABELS[i]}:{mode}")
                    print(f"  step={step:5d}  {' | '.join(status)}  "
                          f"dir=[{move_dir[0]:+.2f},{move_dir[1]:+.2f}]")

                time.sleep(max(0.0, 0.02 - (time.time() - t0)))

        except KeyboardInterrupt:
            print("\nDone.")


if __name__ == "__main__":
    main()
