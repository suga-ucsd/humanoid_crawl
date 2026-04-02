#!/usr/bin/env python3
"""
G1 Spider Crawl — Reference Motion Generator

Generates crawling reference trajectories for imitation learning.
All limbs push in the SAME direction (opposite to desired movement).
Saves body-frame joint trajectories that work for any direction.

Usage:
    python g1_crawl_ref.py                          # visualize forward crawl
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

EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]
EE_LABELS = ["LF", "RF", "LH", "RH"]

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

# Wave gait: 1 limb swings at a time, 3 on ground
# Order: LF → LH → RF → RH (cross pattern for balance)
PHASE_OFFSETS = [0.0, np.pi, np.pi / 2, 3 * np.pi / 2]
SWING_DUTY = 0.25  # 25% swing, 75% stance

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

def ik_solve_limb(model, data, site_name, target_pos, limb_dofs,
                  max_iters=80, step_size=0.5, tol=0.002):
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
# Trajectory: D-shaped hook with wave gait
# ──────────────────────────────────────

def generate_ee_trajectory(home_pos, phase, stride_length, lift_height, move_dir):
    """
    D-shaped hook trajectory. ALL limbs use the SAME move_dir.

    During STANCE: foot drags from front to rear (opposite to move_dir)
                   → pushes body in move_dir
    During SWING:  foot scoops from rear to front (along move_dir)
                   → repositions for next push

    This means ALL 4 limbs contribute to pushing the body in move_dir,
    regardless of whether they're front or back limbs.

    Wave gait: 25% swing (quick scoop), 75% stance (slow push).

    Args:
        home_pos: [3] resting EE position
        phase: limb phase (with offset applied)
        stride_length: half the total stroke distance
        lift_height: peak height during swing
        move_dir: [3] MOVEMENT direction (same for all limbs!)
    """
    p = phase % (2.0 * np.pi)
    swing_end = SWING_DUTY * 2.0 * np.pi

    if p < swing_end:
        # SWING: quick scoop rear→front
        t = p / swing_end  # 0→1
        forward_offset = stride_length * (-1.0 + 2.0 * t)
        lift = lift_height * np.sin(t * np.pi)
    else:
        # STANCE: slow drag front→rear (propulsion)
        t = (p - swing_end) / (2.0 * np.pi - swing_end)  # 0→1
        forward_offset = stride_length * (1.0 - 2.0 * t)
        lift = 0.0

    up_dir = np.array([0.0, 0.0, 1.0])
    target = home_pos + forward_offset * move_dir + lift * up_dir
    return target

# ──────────────────────────────────────
# Record reference trajectory
# ──────────────────────────────────────

def generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                       move_dir, speed, stride, lift, n_cycles=4):
    """
    Generate one reference trajectory for a given movement direction.

    Returns dict with:
        times: [N] timestamps
        joint_qpos: [N, n_joints] actuated joint angles per frame
        ee_positions: [N, 4, 3] end-effector positions per frame
        move_dir: [2] the movement direction
        phase: [N] gait phase per frame
    """
    # Reset to spider pose
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 0.25]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0
    base_qpos = data.qpos[0:7].copy()
    mujoco.mj_forward(model, data)

    # Record home EE positions
    home_positions = {}
    ee_sids = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        home_positions[name] = data.site_xpos[sid].copy()
        ee_sids[name] = sid

    # 3D move direction (XY plane)
    move_dir_3d = np.array([move_dir[0], move_dir[1], 0.0])
    norm = np.linalg.norm(move_dir_3d)
    if norm > 1e-6:
        move_dir_3d /= norm
    else:
        move_dir_3d = np.array([1.0, 0.0, 0.0])

    dt = 0.02  # 50 Hz
    duration = n_cycles / speed
    n_steps = int(duration / dt)

    times = []
    joint_qpos_list = []
    ee_pos_list = []
    phase_list = []

    phase = 0.0

    # Get actuated joint qpos indices
    act_qa = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        act_qa.append(model.jnt_qposadr[jid])
    act_qa = np.array(act_qa)

    for step in range(n_steps):
        data.qpos[0:7] = base_qpos

        for i, name in enumerate(EE_SITES):
            limb_phase = phase + PHASE_OFFSETS[i]
            target = generate_ee_trajectory(
                home_positions[name], limb_phase,
                stride, lift, move_dir_3d
            )
            ik_solve_limb(model, data, name, target, limb_dofs)

        mujoco.mj_forward(model, data)

        # Record
        times.append(step * dt)
        joint_qpos_list.append(data.qpos[act_qa].copy())

        ee_pos = np.zeros((4, 3))
        for j, name in enumerate(EE_SITES):
            ee_pos[j] = data.site_xpos[ee_sids[name]].copy()
        ee_pos_list.append(ee_pos)

        phase_list.append(phase % (2 * np.pi))

        phase += 2.0 * np.pi * speed * dt

    return {
        "times": np.array(times),
        "joint_qpos": np.array(joint_qpos_list),
        "ee_positions": np.array(ee_pos_list),
        "move_dir": np.array(move_dir),
        "phase": np.array(phase_list),
        "speed": speed,
        "stride": stride,
        "lift": lift,
    }

def save_reference(ref, filename):
    np.savez(filename, **ref)
    print(f"  Saved {len(ref['times'])} frames to {filename}")
    print(f"    Duration: {ref['times'][-1]:.1f}s | Dir: {ref['move_dir']} | "
          f"Shape: {ref['joint_qpos'].shape}")

# ──────────────────────────────────────
# Main
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl Reference Motion")
    parser.add_argument("--speed", type=float, default=0.8, help="gait frequency Hz")
    parser.add_argument("--stride", type=float, default=0.06, help="stride length meters")
    parser.add_argument("--lift", type=float, default=0.04, help="foot lift height meters")
    parser.add_argument("--dir", nargs=2, type=float, default=[1.0, 0.0],
                        help="movement direction X Y")
    parser.add_argument("--save", action="store_true", help="save forward reference")
    parser.add_argument("--save-all", action="store_true", help="save 8 directions")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0

    joint_map = get_joint_map(model)
    limb_dofs = build_limb_dofs(model)
    home_qpos = load_pose(model, joint_map)

    # Setup initial pose
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 0.25]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0
    base_qpos = data.qpos[0:7].copy()
    mujoco.mj_forward(model, data)

    # Normalize movement direction
    move_dir = np.array(args.dir)
    norm = np.linalg.norm(move_dir)
    if norm > 1e-6:
        move_dir /= norm
    move_dir_3d = np.array([move_dir[0], move_dir[1], 0.0])

    print("Spider Crawl Reference Motion (Wave Gait)")
    print(f"  Speed: {args.speed} Hz | Stride: {args.stride}m | Lift: {args.lift}m")
    print(f"  Movement direction: [{move_dir[0]:.2f}, {move_dir[1]:.2f}]")
    print(f"  Gait: wave (3 on ground, 1 swinging)")
    print(f"  All limbs push in SAME direction for forward motion")

    # Print EE home positions
    print(f"\nHome EE positions:")
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid]
        print(f"  {name:20s}: [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]")

    # ── Save references ──
    if args.save_all:
        print("\nGenerating references for 8 directions + stationary...")
        directions = {
            "forward":  [1.0, 0.0],
            "backward": [-1.0, 0.0],
            "left":     [0.0, 1.0],
            "right":    [0.0, -1.0],
            "fwd_left": [0.707, 0.707],
            "fwd_right":[0.707, -0.707],
            "bwd_left": [-0.707, 0.707],
            "bwd_right":[-0.707, -0.707],
        }
        all_refs = {}
        for dir_name, d in directions.items():
            print(f"\n  Generating {dir_name} [{d[0]:.2f}, {d[1]:.2f}]...")
            ref = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                     d, args.speed, args.stride, args.lift)
            all_refs[dir_name] = ref
            save_reference(ref, f"g1_ref_{dir_name}.npz")

        # Also save stationary (zero stride) as the "idle" reference
        print(f"\n  Generating stationary...")
        ref_idle = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                      [1, 0], args.speed, 0.0, 0.0)
        save_reference(ref_idle, "g1_ref_stationary.npz")

        print(f"\nSaved {len(directions)+1} reference files!")
        print("Files: g1_ref_forward.npz, g1_ref_backward.npz, etc.")

        # Reset for visualization
        data.qpos[:] = home_qpos
        data.qpos[0:3] = [0, 0, 0.25]
        data.qpos[3:7] = [0.707, 0, -0.707, 0]
        mujoco.mj_forward(model, data)

    elif args.save:
        print("\nGenerating forward reference...")
        ref = generate_reference(model, data, joint_map, limb_dofs, home_qpos,
                                 move_dir.tolist(), args.speed, args.stride, args.lift)
        save_reference(ref, "g1_ref_forward.npz")

        data.qpos[:] = home_qpos
        data.qpos[0:3] = [0, 0, 0.25]
        data.qpos[3:7] = [0.707, 0, -0.707, 0]
        mujoco.mj_forward(model, data)

    # ── Visualization ──
    print("\nVisualization running — Ctrl+C to stop\n")

    # Record home EE positions for visualization
    home_positions = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        home_positions[name] = data.site_xpos[sid].copy()

    phase = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t = time.time()

                data.qpos[0:7] = base_qpos

                for i, name in enumerate(EE_SITES):
                    limb_phase = phase + PHASE_OFFSETS[i]
                    target = generate_ee_trajectory(
                        home_positions[name], limb_phase,
                        args.stride, args.lift, move_dir_3d
                    )
                    ik_solve_limb(model, data, name, target, limb_dofs)

                mujoco.mj_forward(model, data)
                viewer.sync()

                phase += 2.0 * np.pi * args.speed * 0.02

                step += 1
                if step % 50 == 0:
                    status = []
                    for i in range(4):
                        lp = (phase + PHASE_OFFSETS[i]) % (2 * np.pi)
                        mode = "SWING " if lp < SWING_DUTY * 2 * np.pi else "STANCE"
                        status.append(f"{EE_LABELS[i]}:{mode}")
                    print(f"  step={step:5d}  {' | '.join(status)}  "
                          f"dir=[{move_dir[0]:+.2f},{move_dir[1]:+.2f}]")

                time.sleep(max(0, 0.02 - (time.time() - t)))

        except KeyboardInterrupt:
            print("\nDone.")

if __name__ == "__main__":
    main()
