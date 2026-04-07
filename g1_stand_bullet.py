#!/usr/bin/env python3
"""
G1 Spider Stand — PyBullet version
Port of the MuJoCo g1_stand.py to PyBullet + simple_g1.urdf.

Spider pose: robot belly-up, arms + legs touching ground as 4-point support.
PD control in TORQUE_CONTROL mode, gentle gravity ramp on reset.

Usage:
    python g1_stand_pybullet.py --mode hold
    python g1_stand_pybullet.py --mode hold_debug
    python g1_stand_pybullet.py --mode test
    python g1_stand_pybullet.py --mode train --steps 2000000 --n_envs 8
    python g1_stand_pybullet.py --mode eval  --checkpoint models/best/best_model
"""

import argparse, os, time
import numpy as np
import pybullet as p
import pybullet_data

URDF_PATH = os.path.join(os.path.dirname(__file__), "simple_g1.urdf")

# End-effector link names in the URDF (contact spheres)
EE_LINK_NAMES = [
    "left_foot_link",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
]

# Torque limits (N·m) — same as original Python script
TORQUE_LIMITS = {
    "left_hip_pitch_joint":    88,
    "left_hip_roll_joint":    139,
    "left_hip_yaw_joint":      88,
    "left_knee_joint":        139,
    "right_hip_pitch_joint":   88,
    "right_hip_roll_joint":   139,
    "right_hip_yaw_joint":     88,
    "right_knee_joint":       139,
    "waist_yaw_joint":         88,
    "left_shoulder_pitch_joint":  60,
    "left_shoulder_roll_joint":   60,
    "left_shoulder_yaw_joint":    25,
    "left_elbow_joint":           60,
    "left_wrist_pitch_joint":     10,
    "right_shoulder_pitch_joint": 60,
    "right_shoulder_roll_joint":  60,
    "right_shoulder_yaw_joint":   25,
    "right_elbow_joint":          60,
    "right_wrist_pitch_joint":    10,
}

# Spider pose joint targets (radians) — same as original
POSE_DEFAULTS = {
    "left_hip_pitch_joint":      -0.90,
    "left_hip_roll_joint":        0.30,
    "left_hip_yaw_joint":         0.00,
    "left_knee_joint":            1.60,
    "right_hip_pitch_joint":     -0.90,
    "right_hip_roll_joint":      -0.30,
    "right_hip_yaw_joint":        0.00,
    "right_knee_joint":           1.60,
    "waist_yaw_joint":            0.00,
    "left_shoulder_pitch_joint": -2.80,
    "left_shoulder_roll_joint":   0.80,
    "left_shoulder_yaw_joint":    0.00,
    "left_elbow_joint":           0.30,
    "left_wrist_pitch_joint":     0.00,
    "right_shoulder_pitch_joint":-2.80,
    "right_shoulder_roll_joint": -0.80,
    "right_shoulder_yaw_joint":   0.00,
    "right_elbow_joint":          0.30,
    "right_wrist_pitch_joint":    0.00,
}

# Belly-up orientation: MuJoCo (w,x,y,z) = (0.707, 0, -0.707, 0)  →  PyBullet (x,y,z,w)
BELLY_UP_QUAT = (0.0, -0.7071068, 0.0, 0.7071068)

TIMESTEP  = 0.002
N_SOLVER  = 50

# ──────────────────────────────────────────────────────────────────────────────
# Physics setup
# ──────────────────────────────────────────────────────────────────────────────

def init_physics(gui: bool = True) -> int:
    """Connect to PyBullet, return client id."""
    client = p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
    p.setTimeStep(TIMESTEP, physicsClientId=client)
    p.setPhysicsEngineParameter(
        numSolverIterations=N_SOLVER,
        numSubSteps=1,
        physicsClientId=client,
    )
    p.setGravity(0, 0, -9.81, physicsClientId=client)
    return client


def load_scene(client: int):
    """Load floor plane. Returns floor body id."""
    floor = p.loadURDF("plane.urdf", physicsClientId=client)
    # High friction to match MuJoCo floor friction=3
    p.changeDynamics(floor, -1,
                     lateralFriction=3.0,
                     spinningFriction=0.5,
                     rollingFriction=0.5,
                     physicsClientId=client)
    return floor


def load_robot(client: int) -> int:
    """Load URDF, configure contact dynamics."""
    robot = p.loadURDF(
        URDF_PATH,
        basePosition=[0, 0, 1.0],
        baseOrientation=BELLY_UP_QUAT,
        useFixedBase=False,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        physicsClientId=client,
    )
    # High damping + friction on all links
    num_joints = p.getNumJoints(robot, physicsClientId=client)
    for j in range(num_joints):
        p.changeDynamics(
            robot, j,
            lateralFriction=3.0,
            spinningFriction=0.5,
            rollingFriction=0.5,
            jointDamping=3.0,
            physicsClientId=client,
        )
    p.changeDynamics(robot, -1,  # base (pelvis)
                     lateralFriction=3.0,
                     physicsClientId=client)
    return robot

# ──────────────────────────────────────────────────────────────────────────────
# Joint utilities
# ──────────────────────────────────────────────────────────────────────────────

def build_joint_map(robot: int, client: int) -> dict:
    """
    Returns {joint_name: joint_index} for all non-fixed joints.
    Also returns a separate link-name→link-index map (includes fixed joints).
    """
    joint_map  = {}   # name -> index, revolute/prismatic only
    link_map   = {}   # link_name -> link_index (link_index == joint_index in PyBullet)
    n = p.getNumJoints(robot, physicsClientId=client)
    for i in range(n):
        info = p.getJointInfo(robot, i, physicsClientId=client)
        jname  = info[1].decode()
        lname  = info[12].decode()
        jtype  = info[2]
        link_map[lname] = i
        if jtype != p.JOINT_FIXED:
            joint_map[jname] = i
    return joint_map, link_map


def build_gains(joint_names):
    """Build kp, kd, max_tau arrays aligned to joint_names list."""
    n = len(joint_names)
    max_tau = np.zeros(n)
    kp      = np.zeros(n)
    kd      = np.zeros(n)
    for i, name in enumerate(joint_names):
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i]  = frc * 3.0
        kd[i]  = kp[i] * 0.3
    return kp, kd, max_tau


def home_angles_array(joint_names):
    """Return target angles for `joint_names` from POSE_DEFAULTS (0 if not listed)."""
    return np.array([POSE_DEFAULTS.get(n, 0.0) for n in joint_names])


def disable_default_motors(robot: int, joint_indices, client: int):
    """Disable PyBullet's built-in position/velocity controllers on actuated joints."""
    for j in joint_indices:
        p.setJointMotorControl2(
            robot, j, p.VELOCITY_CONTROL, force=0.0,
            physicsClientId=client,
        )

# ──────────────────────────────────────────────────────────────────────────────
# PD control
# ──────────────────────────────────────────────────────────────────────────────

def pd_control(robot: int, joint_indices, q_target, kp, kd, max_tau, client: int):
    """
    Compute tau = kp*(q_target - q) + kd*(0 - qd), clip, apply TORQUE_CONTROL.
    Returns raw computed torques.
    """
    states    = p.getJointStates(robot, joint_indices, physicsClientId=client)
    positions = np.array([s[0] for s in states])
    velocities= np.array([s[1] for s in states])

    tau = kp * (q_target - positions) + kd * (0.0 - velocities)
    tau = np.clip(tau, -max_tau, max_tau)

    p.setJointMotorControlArray(
        robot,
        joint_indices,
        p.TORQUE_CONTROL,
        forces=tau.tolist(),
        physicsClientId=client,
    )
    return tau, positions, velocities

# ──────────────────────────────────────────────────────────────────────────────
# EE contacts
# ──────────────────────────────────────────────────────────────────────────────

def get_ee_contacts(robot: int, ee_link_indices, floor_id: int, client: int):
    """Returns float array [0|1] per EE — 1 if that link touches the floor."""
    contacts = np.zeros(len(ee_link_indices))
    for i, lnk in enumerate(ee_link_indices):
        pts = p.getContactPoints(
            bodyA=floor_id, bodyB=robot, linkIndexB=lnk,
            physicsClientId=client,
        )
        if pts:
            contacts[i] = 1.0
    return contacts

# ──────────────────────────────────────────────────────────────────────────────
# Spawn height
# ──────────────────────────────────────────────────────────────────────────────

def find_spawn_height(robot: int, joint_map: dict, link_map: dict,
                      joint_names, home_q, client: int) -> float:
    """
    Place robot at belly-up with pelvis at z=1.0, set all joints to home,
    read EE world positions, return spawn height so lowest EE is at z=0.03.
    """
    # Reset to reference height
    p.resetBasePositionAndOrientation(
        robot, [0, 0, 1.0], BELLY_UP_QUAT, physicsClientId=client)

    # Set all joints to home pose
    jidx_list = [joint_map[n] for n in joint_names]
    for jidx, ang in zip(jidx_list, home_q):
        p.resetJointState(robot, jidx, ang, 0.0, physicsClientId=client)

    # Step once with zero gravity so FK updates without falling
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.stepSimulation(physicsClientId=client)

    pelvis_pos, _ = p.getBasePositionAndOrientation(robot, physicsClientId=client)
    pelvis_h = pelvis_pos[2]

    ee_z = {}
    print("EE heights (pelvis reference z=1.0m):")
    for ee_name in EE_LINK_NAMES:
        if ee_name not in link_map:
            print(f"  WARNING: {ee_name} not found in link_map")
            continue
        lnk_idx = link_map[ee_name]
        state   = p.getLinkState(robot, lnk_idx, physicsClientId=client)
        # getLinkState returns (worldPos, worldOrn, ...) — index 4 is world link frame pos
        # index 0 = world pos of link CoM, index 4 = world pos of link frame
        world_pos = state[4]  # link frame world position (joint origin)
        # For foot/hand, the sphere is 0.05m below the joint. To get the sphere's
        # world position we need to apply the local offset. However since we placed
        # contact spheres in the collision/visual at a local offset, getLinkState[4]
        # gives the link origin. We want the sphere center.
        # Offsets from URDF: foot at (0,0,-0.05), hand at (0.08,0,0) in link frame.
        # Get link world orientation to rotate local offset.
        link_world_orn = state[5]  # world orientation of link frame
        if "foot" in ee_name:
            local_offset = np.array([0.0, 0.0, -0.05])
        else:
            local_offset = np.array([0.08, 0.0, 0.0])
        rot_mat = np.array(p.getMatrixFromQuaternion(link_world_orn)).reshape(3, 3)
        sphere_world = np.array(world_pos) + rot_mat @ local_offset
        ee_z[ee_name] = sphere_world[2]
        print(f"  {ee_name:22s}: z={sphere_world[2]:.4f}m")

    if not ee_z:
        print("  No EE links found, using default spawn height 0.25m")
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        return 0.25

    lowest_ee  = min(ee_z.values())
    highest_ee = max(ee_z.values())
    spread     = highest_ee - lowest_ee

    spawn_h = pelvis_h - lowest_ee + 0.03
    print(f"  Lowest EE z={lowest_ee:.4f}m → spawn pelvis at {spawn_h:.3f}m")

    if spread > 0.03:
        print(f"  ⚠ EE spread: {spread:.3f}m — some EEs won't touch ground")
        for name, z in ee_z.items():
            at_ground = z - lowest_ee + 0.03
            status = "✅ on ground" if at_ground < 0.06 else f"❌ at {at_ground:.3f}m (too high)"
            print(f"    {name:22s}: {status}")

    p.setGravity(0, 0, -9.81, physicsClientId=client)
    return spawn_h

# ──────────────────────────────────────────────────────────────────────────────
# Gravity settle
# ──────────────────────────────────────────────────────────────────────────────

def gentle_gravity_settle(robot: int, joint_indices, home_q,
                           kp, kd, max_tau, client: int,
                           duration: float = 3.0):
    """
    Gradually ramp gravity from 0 → -9.81 m/s² while holding PD on home pose.
    Zero all velocities at the end.
    """
    n_steps = int(duration / TIMESTEP)
    for step in range(n_steps):
        blend   = min(1.0, step / (n_steps * 0.7))
        gravity = -9.81 * blend
        p.setGravity(0, 0, gravity, physicsClientId=client)
        pd_control(robot, joint_indices, home_q, kp, kd, max_tau, client)
        p.stepSimulation(physicsClientId=client)

    p.setGravity(0, 0, -9.81, physicsClientId=client)

    # Zero all joint velocities
    for j in joint_indices:
        state = p.getJointState(robot, j, physicsClientId=client)
        p.resetJointState(robot, j, state[0], 0.0, physicsClientId=client)

    # Zero base velocity
    pos, orn = p.getBasePositionAndOrientation(robot, physicsClientId=client)
    p.resetBasePositionAndOrientation(robot, pos, orn, physicsClientId=client)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0], physicsClientId=client)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_robot_pose(robot: int, joint_map: dict, joint_names,
                   home_q, spawn_h: float, client: int):
    """Place robot at spawn_h, belly-up, all joints at home pose."""
    p.resetBasePositionAndOrientation(
        robot, [0, 0, spawn_h], BELLY_UP_QUAT, physicsClientId=client)
    p.resetBaseVelocity(robot, [0, 0, 0], [0, 0, 0], physicsClientId=client)
    for name, ang in zip(joint_names, home_q):
        j = joint_map[name]
        p.resetJointState(robot, j, ang, 0.0, physicsClientId=client)


def get_base_state(robot: int, client: int):
    """Returns pos (3,), orn_xyzw (4,), lin_vel (3,), ang_vel (3,)."""
    pos, orn  = p.getBasePositionAndOrientation(robot, physicsClientId=client)
    lv,  av   = p.getBaseVelocity(robot, physicsClientId=client)
    return np.array(pos), np.array(orn), np.array(lv), np.array(av)


def get_pelvis_up_dot(robot: int, client: int) -> float:
    """
    Returns dot of pelvis local -Z with world +Z.
    +1 when belly-up (robot inverted), -1 when upright.
    """
    _, orn, _, _ = get_base_state(robot, client)
    mat  = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    # Pelvis local +Z (mat[:,2]) should point DOWN in belly-up → dot with world-Z is -1
    # We want belly_up metric: max(0, -pelvis_z_world_dot)
    belly_up = max(0.0, -mat[2, 2])
    return belly_up

# ──────────────────────────────────────────────────────────────────────────────
# Hold mode
# ──────────────────────────────────────────────────────────────────────────────

def hold_pose(args):
    client = init_physics(gui=True)
    floor  = load_scene(client)
    robot  = load_robot(client)

    joint_map, link_map = build_joint_map(robot, client)
    joint_names  = list(POSE_DEFAULTS.keys())
    joint_indices= [joint_map[n] for n in joint_names]
    home_q       = home_angles_array(joint_names)
    kp, kd, max_tau = build_gains(joint_names)
    ee_link_indices = [link_map[n] for n in EE_LINK_NAMES if n in link_map]

    # Disable built-in motors
    disable_default_motors(robot, joint_indices, client)

    spawn_h = find_spawn_height(robot, joint_map, link_map, joint_names, home_q, client)
    print(f"Computed spawn height: {spawn_h:.3f}m")

    set_robot_pose(robot, joint_map, joint_names, home_q, spawn_h, client)
    disable_default_motors(robot, joint_indices, client)  # re-disable after reset

    print("\nEE positions before settle:")
    p.setGravity(0, 0, 0, physicsClientId=client)
    p.stepSimulation(physicsClientId=client)
    for ee_name in EE_LINK_NAMES:
        if ee_name in link_map:
            s = p.getLinkState(robot, link_map[ee_name], physicsClientId=client)
            print(f"  {ee_name:22s}: world_z={s[4][2]:.4f}m")

    print("\nSettling with gradual gravity (3s)...")
    gentle_gravity_settle(robot, joint_indices, home_q, kp, kd, max_tau, client, duration=3.0)

    pos, _, _, _ = get_base_state(robot, client)
    settled_h    = pos[2]
    start_xy     = pos[:2].copy()
    contacts     = get_ee_contacts(robot, ee_link_indices, floor, client)
    print(f"\nSettled at h={settled_h:.3f}m  EE contacts: {contacts.astype(int).tolist()}")
    print(f"\nHolding pose — Ctrl-C to quit\n")

    p.resetDebugVisualizerCamera(
        cameraDistance=1.8, cameraYaw=30, cameraPitch=-20,
        cameraTargetPosition=[0, 0, 0.2], physicsClientId=client)

    step = 0
    try:
        while True:
            t0 = time.time()
            tau, qpos, qvel = pd_control(robot, joint_indices, home_q,
                                         kp, kd, max_tau, client)
            p.stepSimulation(physicsClientId=client)
            step += 1

            if step % 500 == 0:
                pos, orn, lv, av = get_base_state(robot, client)
                h     = pos[2]
                drift = np.linalg.norm(pos[:2] - start_xy)
                vel   = np.linalg.norm(lv[:2])
                cts   = get_ee_contacts(robot, ee_link_indices, floor, client)
                n_sat = int(np.sum(np.abs(tau) >= max_tau * 0.95))
                status = "✅ STABLE" if drift < 0.03 and vel < 0.05 else "❌ SLIDING"
                print(f"  t={step*TIMESTEP:5.1f}s  h={h:.3f}m  "
                      f"drift={drift:.3f}m  vel_xy={vel:.3f}  "
                      f"EE={int(np.sum(cts))}/4  sat={n_sat}/{len(joint_names)}  {status}")

            elapsed = time.time() - t0
            time.sleep(max(0.0, TIMESTEP - elapsed))

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        p.disconnect(client)

# ──────────────────────────────────────────────────────────────────────────────
# Hold debug
# ──────────────────────────────────────────────────────────────────────────────

def hold_debug(args):
    client = init_physics(gui=False)   # headless for fast debug run
    floor  = load_scene(client)
    robot  = load_robot(client)

    joint_map, link_map = build_joint_map(robot, client)
    joint_names   = list(POSE_DEFAULTS.keys())
    joint_indices = [joint_map[n] for n in joint_names]
    home_q        = home_angles_array(joint_names)
    kp, kd, max_tau = build_gains(joint_names)
    ee_link_indices = [link_map[n] for n in EE_LINK_NAMES if n in link_map]

    disable_default_motors(robot, joint_indices, client)
    spawn_h = find_spawn_height(robot, joint_map, link_map, joint_names, home_q, client)
    set_robot_pose(robot, joint_map, joint_names, home_q, spawn_h, client)
    disable_default_motors(robot, joint_indices, client)

    print("Settling with gradual gravity...")
    gentle_gravity_settle(robot, joint_indices, home_q, kp, kd, max_tau, client, duration=3.0)

    pos0, _, _, _ = get_base_state(robot, client)
    start_xy = pos0[:2].copy()

    print(f"\nDebug run — 3s after settle")
    print(f"{'step':>6}  {'h':>6}  {'EE':>4}  {'drift':>6}  top drifting joints")
    print("-" * 80)

    N = int(3.0 / TIMESTEP)
    max_drift  = np.zeros(len(joint_names))
    sat_count  = np.zeros(len(joint_names))

    for s in range(N):
        tau, qpos, _ = pd_control(robot, joint_indices, home_q, kp, kd, max_tau, client)
        p.stepSimulation(physicsClientId=client)

        drift_j   = np.abs(qpos - home_q)
        max_drift = np.maximum(max_drift, drift_j)
        sat_count += (np.abs(tau) >= max_tau * 0.95).astype(float)

        if s % 500 == 0:
            pos, _, lv, _ = get_base_state(robot, client)
            h = pos[2]
            xy_drift = np.linalg.norm(pos[:2] - start_xy)
            cts = get_ee_contacts(robot, ee_link_indices, floor, client)
            top3 = np.argsort(drift_j)[::-1][:3]
            top_str = "  ".join(
                f"{joint_names[j]}={drift_j[j]:.3f}"
                for j in top3 if drift_j[j] > 0.01)
            print(f"  {s:5d}  {h:6.3f}  {int(np.sum(cts))}/4  {xy_drift:6.3f}  {top_str}")

    print(f"\n── Joint Analysis ──")
    print(f"{'Joint':<35s}  {'max_drift':>10s}  {'sat%':>6s}  {'kp':>7s}  {'max_tau':>8s}")
    print("-" * 75)
    for idx in np.argsort(max_drift)[::-1]:
        name = joint_names[idx]
        pct  = 100 * sat_count[idx] / N
        print(f"  {name:<35s}  {max_drift[idx]:10.4f}  {pct:5.1f}%  "
              f"{kp[idx]:7.1f}  {max_tau[idx]:8.1f}")

    bad = [(joint_names[i], max_drift[i], 100 * sat_count[i] / N)
           for i in range(len(joint_names))
           if max_drift[i] > 0.05 and sat_count[i] / N > 0.3]
    if bad:
        print(f"\n⚠ {len(bad)} joints struggling:")
        for name, d, s in bad:
            print(f"  {name}: drift={d:.3f}rad, saturated {s:.0f}% of time")
    else:
        print("\n✅ All joints holding well!")

    p.disconnect(client)

# ──────────────────────────────────────────────────────────────────────────────
# RL Environment (Gymnasium)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


if GYM_AVAILABLE:
    class G1StandEnv(gym.Env):
        metadata = {"render_modes": ["human"], "render_fps": 50}
        RESIDUAL_SCALE = 0.10
        N_SUBSTEPS     = 10
        MAX_STEPS      = 500

        def __init__(self, render_mode=None):
            super().__init__()
            self.render_mode = render_mode
            self._gui = (render_mode == "human")

            self._client = init_physics(gui=self._gui)
            self._floor  = load_scene(self._client)
            self._robot  = load_robot(self._client)

            self._jmap, self._lmap = build_joint_map(self._robot, self._client)
            self._jnames  = list(POSE_DEFAULTS.keys())
            self._jidx    = [self._jmap[n] for n in self._jnames]
            self._home_q  = home_angles_array(self._jnames)
            self._kp, self._kd, self._max_tau = build_gains(self._jnames)
            self._ee_idx  = [self._lmap[n] for n in EE_LINK_NAMES if n in self._lmap]

            disable_default_motors(self._robot, self._jidx, self._client)

            self._spawn_h       = 0.25
            self._settled_h     = 0.15
            self._start_xy      = np.zeros(2)
            self._step_count    = 0
            self._prev_action   = np.zeros(len(self._jnames))
            self._spawn_h_ready = False

            nj = len(self._jnames)
            self.action_space = spaces.Box(-1.0, 1.0, shape=(nj,), dtype=np.float32)
            # quat(4) + angvel(3) + linvel(3) + joint_off(nj) + joint_vel(nj) + ee_contact(4) + h(1)
            obs_dim = 4 + 3 + 3 + nj * 2 + 4 + 1
            self.observation_space = spaces.Box(
                -np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # ── helpers ──────────────────────────────────────────────────────────

        def _ensure_spawn_h(self):
            if not self._spawn_h_ready:
                disable_default_motors(self._robot, self._jidx, self._client)
                self._spawn_h = find_spawn_height(
                    self._robot, self._jmap, self._lmap,
                    self._jnames, self._home_q, self._client)
                self._spawn_h_ready = True

        def _get_obs(self):
            pos, orn, lv, av = get_base_state(self._robot, self._client)
            states    = p.getJointStates(self._robot, self._jidx,
                                          physicsClientId=self._client)
            jpos = np.array([s[0] for s in states])
            jvel = np.array([s[1] for s in states])
            joint_off = jpos - self._home_q
            contacts  = get_ee_contacts(self._robot, self._ee_idx,
                                         self._floor, self._client)
            return np.concatenate([
                orn, av, lv,
                joint_off, jvel,
                contacts, [pos[2]],
            ]).astype(np.float32)

        def _get_reward(self, action):
            pos, orn, lv, av = get_base_state(self._robot, self._client)
            h = pos[2]

            height_rew   = np.exp(-30.0 * (h - self._settled_h) ** 2)
            height_bonus = min(h / max(self._settled_h, 0.05), 1.5)
            contacts     = get_ee_contacts(self._robot, self._ee_idx,
                                            self._floor, self._client)
            contact_rew  = np.sum(contacts) / 4.0
            belly_up     = get_pelvis_up_dot(self._robot, self._client)

            states   = p.getJointStates(self._robot, self._jidx,
                                          physicsClientId=self._client)
            jpos = np.array([s[0] for s in states])
            joint_dev  = np.sum(np.square(jpos - self._home_q))
            pose_rew   = np.exp(-2.0 * joint_dev / len(self._jnames))

            vel_xy  = np.sum(np.square(lv[:2]))
            ang_vel = np.sum(np.square(av))
            drift   = np.sum(np.square(pos[:2] - self._start_xy))
            energy  = np.sum(np.square(
                p.getJointStates(self._robot, self._jidx,
                                  physicsClientId=self._client))) / len(self._jnames)

            # use ctrl array from last PD step — approximate via joint positions
            # (energy penalty: sum squared torques)
            act_rate = np.sum(np.square(action - self._prev_action))

            reward = (
                  3.0 * height_rew
                + 3.0 * height_bonus
                + 2.0 * contact_rew
                + 1.0 * belly_up
                + 1.0 * pose_rew
                + 0.5
                - 0.2 * vel_xy
                - 0.1 * ang_vel
                - 0.5 * drift
                - 0.01 * act_rate
            )
            return float(reward)

        def _is_terminated(self):
            pos, orn, lv, av = get_base_state(self._robot, self._client)
            h = pos[2]
            if np.any(np.isnan(pos)) or np.any(np.isnan(orn)):
                return True
            return h > 0.70 or h < -0.05

        # ── gym interface ─────────────────────────────────────────────────────

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self._ensure_spawn_h()

            p.resetSimulation(physicsClientId=self._client)
            self._floor = load_scene(self._client)
            self._robot = load_robot(self._client)
            self._jmap, self._lmap = build_joint_map(self._robot, self._client)
            self._jidx  = [self._jmap[n] for n in self._jnames]
            self._ee_idx = [self._lmap[n] for n in EE_LINK_NAMES if n in self._lmap]
            disable_default_motors(self._robot, self._jidx, self._client)

            # Small random perturbation
            if self.np_random is not None:
                home_noisy = self._home_q + self.np_random.uniform(
                    -0.02, 0.02, size=self._home_q.shape)
            else:
                home_noisy = self._home_q.copy()

            set_robot_pose(self._robot, self._jmap, self._jnames,
                           home_noisy, self._spawn_h, self._client)
            disable_default_motors(self._robot, self._jidx, self._client)

            gentle_gravity_settle(self._robot, self._jidx, self._home_q,
                                   self._kp, self._kd, self._max_tau,
                                   self._client, duration=1.5)

            pos, _, _, _ = get_base_state(self._robot, self._client)
            self._settled_h   = pos[2]
            self._start_xy    = pos[:2].copy()
            self._step_count  = 0
            self._prev_action = np.zeros(len(self._jnames))

            return self._get_obs(), {}

        def step(self, action):
            action = np.clip(action, -1.0, 1.0)

            # Residual target around home pose
            q_target = self._home_q + action * self.RESIDUAL_SCALE

            for _ in range(self.N_SUBSTEPS):
                pd_control(self._robot, self._jidx, q_target,
                            self._kp, self._kd, self._max_tau, self._client)
                p.stepSimulation(physicsClientId=self._client)

            self._step_count  += 1
            reward     = self._get_reward(action)
            terminated = self._is_terminated()
            truncated  = self._step_count >= self.MAX_STEPS
            self._prev_action = action.copy()
            return self._get_obs(), reward, terminated, truncated, {}

        def render(self):
            pass  # GUI updates automatically in PyBullet GUI mode

        def close(self):
            p.disconnect(self._client)

# ──────────────────────────────────────────────────────────────────────────────
# Train / Eval / Test
# ──────────────────────────────────────────────────────────────────────────────

def train(args):
    if not GYM_AVAILABLE:
        raise ImportError("gymnasium not installed")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs",   exist_ok=True)

    env      = SubprocVecEnv([lambda: G1StandEnv() for _ in range(args.n_envs)])
    eval_env = G1StandEnv()

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        model = PPO.load(args.checkpoint, env=env, device=args.device)
        print(f"Resumed from {args.checkpoint}")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
            max_grad_norm=0.5, verbose=1, device=args.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-1.5,
            ),
        )

    print(f"\nSpider Stand Training  |  steps={args.steps:,}  envs={args.n_envs}  device={args.device}\n")
    model.learn(
        total_timesteps=args.steps,
        callback=[
            CheckpointCallback(save_freq=max(100_000 // args.n_envs, 1),
                               save_path="models/", name_prefix="g1_stand"),
            EvalCallback(eval_env, best_model_save_path="models/best/",
                         log_path="logs/eval/",
                         eval_freq=max(50_000 // args.n_envs, 1),
                         n_eval_episodes=5, deterministic=True),
        ],
        progress_bar=True,
    )
    model.save("models/g1_stand_final")
    env.close()
    eval_env.close()


def evaluate(args):
    if not GYM_AVAILABLE:
        raise ImportError("gymnasium not installed")
    from stable_baselines3 import PPO
    if not args.checkpoint:
        print("Error: --checkpoint required for eval mode")
        return
    model    = PPO.load(args.checkpoint)
    env      = G1StandEnv(render_mode="human")
    obs, _   = env.reset()
    total_r, steps, eps = 0.0, 0, 0
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            steps   += 1
            if steps % 50 == 0:
                pos, _, lv, _ = get_base_state(env._robot, env._client)
                cts = get_ee_contacts(env._robot, env._ee_idx, env._floor, env._client)
                drift = np.linalg.norm(pos[:2] - env._start_xy)
                print(f"  step={steps}  h={pos[2]:.3f}  EE={int(np.sum(cts))}/4  "
                      f"drift={drift:.3f}  rew={total_r:.1f}")
            time.sleep(0.02)
            if term or trunc:
                eps += 1
                print(f"\nEp {eps}: steps={steps}  reward={total_r:.1f}\n")
                total_r, steps = 0.0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


def test_random(args):
    if not GYM_AVAILABLE:
        raise ImportError("gymnasium not installed")
    env    = G1StandEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"Settled height: {env._settled_h:.3f}m\n")
    total_r, steps = 0.0, 0
    try:
        while True:
            action = env.action_space.sample() * 0.1
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            steps   += 1
            time.sleep(0.02)
            if steps % 50 == 0:
                pos, _, lv, _ = get_base_state(env._robot, env._client)
                cts = get_ee_contacts(env._robot, env._ee_idx, env._floor, env._client)
                drift = np.linalg.norm(pos[:2] - env._start_xy)
                print(f"  step={steps:4d}  h={pos[2]:.3f}  EE={int(np.sum(cts))}/4  "
                      f"drift={drift:.3f}  rew/step={total_r/steps:+.2f}")
            if term or trunc:
                print(f"Done — steps={steps}  total={total_r:.1f}  "
                      f"per_step={total_r/max(steps,1):+.2f}\n")
                total_r, steps = 0.0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Stand — PyBullet")
    parser.add_argument("--mode", required=True,
                        choices=["hold", "hold_debug", "train", "eval", "test"])
    parser.add_argument("--steps",      type=int, default=2_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_envs",     type=int, default=8)
    parser.add_argument("--device",     type=str, default="cpu")
    args = parser.parse_args()

    dispatch = {
        "hold":       hold_pose,
        "hold_debug": hold_debug,
        "train":      train,
        "eval":       evaluate,
        "test":       test_random,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
