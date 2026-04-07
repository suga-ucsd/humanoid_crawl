#!/usr/bin/env python3
"""
G1 Spider Stand v3 — Hold spider pose at height, no collapse.

Key fix: find the exact spawn height where EEs touch ground,
start there with PD on. No dropping, no impact, no collapse.

Usage:
    python g1_stand.py --mode hold
    python g1_stand.py --mode hold_debug
    python g1_stand.py --mode test
    python g1_stand.py --mode train --steps 2000000 --n_envs 8
    python g1_stand.py --mode eval --checkpoint models/best/best_model
"""

import argparse, os, time
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

# Spider pose — arms at maximum reach to touch ground
POSE_DEFAULTS = {
    "left_hip_pitch_joint": -0.90,
    "left_hip_roll_joint": 0.30,
    "left_hip_yaw_joint": 0.00,
    "left_knee_joint": 1.60,
    "right_hip_pitch_joint": -0.90,
    "right_hip_roll_joint": -0.30,
    "right_hip_yaw_joint": 0.00,
    "right_knee_joint": 1.60,
    "waist_yaw_joint": 0.00,
    "left_shoulder_pitch_joint": -2.80,   # near limit -3.09 — fully extended back
    "left_shoulder_roll_joint": 0.80,     # wide splay
    "left_shoulder_yaw_joint": 0.00,
    "left_elbow_joint": 0.30,            # nearly straight — max reach
    "left_wrist_pitch_joint": 0.00,
    "right_shoulder_pitch_joint": -2.80,
    "right_shoulder_roll_joint": -0.80,
    "right_shoulder_yaw_joint": 0.00,
    "right_elbow_joint": 0.30,
    "right_wrist_pitch_joint": 0.00,
}

BELLY_UP_QUAT = np.array([0.707, 0.0, -0.707, 0.0])  # -90° around Y = belly up

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

    if os.path.exists(filename):
        print(f"Loading pose from {filename}")
        with open(filename) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0] in joint_map:
                    q[joint_map[parts[0]]] = float(parts[1])
    else:
        print(f"⚠ {filename} not found — using POSE_DEFAULTS")
        for jname, val in POSE_DEFAULTS.items():
            if jname in joint_map:
                q[joint_map[jname]] = val

    return q

def configure_physics(model):
    model.opt.timestep = 0.002
    model.opt.iterations = 50
    model.opt.ls_iterations = 20
    model.opt.solver = 2
    for i in range(6, model.nv):
        model.dof_damping[i] = 3.0  # high passive damping
    for i in range(model.ngeom):
        if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
            model.geom_friction[i] = [3.0, 0.5, 0.5]

def build_gains(model):
    """Very stiff PD with high damping — resist collapse."""
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 3.0    # very stiff
        kd[i] = kp[i] * 0.3  # strong damping
    return kp, kd, max_tau

def pd_control_vec(model, data, q_target, kp, kd, max_tau):
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)
    return tau

def get_ee_contacts(model, data):
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    ee_site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in EE_SITES]
    contacts = np.zeros(4)
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != floor_id and g2 != floor_id:
            continue
        robot_geom = g2 if g1 == floor_id else g1
        robot_body = model.geom_bodyid[robot_geom]
        for j, sid in enumerate(ee_site_ids):
            ee_body = model.site_bodyid[sid]
            b = ee_body
            while b > 0:
                if b == robot_body:
                    contacts[j] = 1.0
                    break
                b = model.body_parentid[b]
    return contacts

def find_spawn_height(model, data, home_qpos):
    """
    Find pelvis height where lowest EE is at z=0.03.
    Prints all EE heights so you can see if any don't reach.
    """
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    pelvis_h = data.qpos[2]
    ee_z = {}
    print("EE heights (pelvis at 1.0m):")
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        z = data.site_xpos[sid][2]
        ee_z[name] = z
        print(f"  {name:20s}: z={z:.4f}m")

    lowest_ee = min(ee_z.values())
    highest_ee = max(ee_z.values())

    # Set height so lowest EE is at 0.03m
    spawn_h = pelvis_h - lowest_ee + 0.03
    print(f"  Lowest EE at {lowest_ee:.4f}m → spawn pelvis at {spawn_h:.3f}m")

    # Check if other EEs will reach ground
    spread = highest_ee - lowest_ee
    if spread > 0.03:
        print(f"  ⚠ EE spread: {spread:.3f}m — some EEs won't touch ground")
        for name, z in ee_z.items():
            ground_z = z - lowest_ee + 0.03
            status = "✅ on ground" if ground_z < 0.06 else f"❌ at {ground_z:.3f}m (too high)"
            print(f"    {name:20s}: {status}")

    return spawn_h

def gentle_gravity_settle(model, data, home_qpos, kp, kd, max_tau, duration=3.0):
    """
    Gradually turn on gravity while holding PD.
    Starts with zero gravity, ramps to full over `duration` seconds.
    This lets the robot find its equilibrium without impact.
    """
    real_gravity = model.opt.gravity.copy()
    n_steps = int(duration / model.opt.timestep)
    jnt_ids = model.actuator_trnid[:, 0]
    act_qa = model.jnt_qposadr[jnt_ids]

    for step in range(n_steps):
        # Ramp gravity from 0 to full
        blend = min(1.0, step / (n_steps * 0.7))  # reach full at 70%
        model.opt.gravity[:] = real_gravity * blend

        # PD targets spider pose
        q_target = home_qpos.copy()
        q_target[0:7] = data.qpos[0:7]
        for i, qa in enumerate(act_qa):
            q_target[qa] = home_qpos[qa]
        pd_control_vec(model, data, q_target, kp, kd, max_tau)
        mujoco.mj_step(model, data)

    # Restore full gravity
    model.opt.gravity[:] = real_gravity
    # Zero velocities
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

# ──────────────────────────────────────
# Hold mode
# ──────────────────────────────────────

def hold_pose(args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)
    kp, kd, max_tau = build_gains(model)

    jnt_ids = model.actuator_trnid[:, 0]
    act_qa = model.jnt_qposadr[jnt_ids]

    # Find correct spawn height
    spawn_h = find_spawn_height(model, data, home_qpos)
    print(f"Computed spawn height: {spawn_h:.3f}m")

    # Set pose at correct height
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, spawn_h]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    # Print EE positions before settle
    print("\nEE positions before settle:")
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid]
        print(f"  {name:20s}: z={pos[2]:.4f}m")

    # Gradually turn on gravity
    print("\nSettling with gradual gravity (3s)...")
    gentle_gravity_settle(model, data, home_qpos, kp, kd, max_tau, duration=3.0)

    pelvis_h = data.qpos[2]
    print(f"Settled at h={pelvis_h:.3f}m")

    # Print EE positions after settle
    print("\nEE positions after settle:")
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos = data.site_xpos[sid]
        print(f"  {name:20s}: z={pos[2]:.4f}m")

    contacts = get_ee_contacts(model, data)
    print(f"EE contacts: {contacts.astype(int).tolist()}")

    start_xy = data.qpos[0:2].copy()

    print(f"\nHolding pose — target is to stay at h={pelvis_h:.3f}m\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t0 = time.time()

                q_target = home_qpos.copy()
                q_target[0:7] = data.qpos[0:7]
                for i, qa in enumerate(act_qa):
                    q_target[qa] = home_qpos[qa]
                pd_control_vec(model, data, q_target, kp, kd, max_tau)
                mujoco.mj_step(model, data)
                viewer.sync()
                step += 1

                if step % 500 == 0:
                    h = data.qpos[2]
                    drift = np.linalg.norm(data.qpos[0:2] - start_xy)
                    contacts = get_ee_contacts(model, data)
                    vel = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)

                    # Check joint saturation
                    tau = kp * (home_qpos[act_qa] - data.qpos[act_qa]) + kd * (-data.qvel[model.jnt_dofadr[jnt_ids]])
                    n_sat = np.sum(np.abs(tau) >= max_tau * 0.95)

                    status = "✅ STABLE" if drift < 0.03 and vel < 0.05 else "❌ SLIDING"
                    print(f"  t={step*0.002:5.1f}s  h={h:.3f}m  "
                          f"drift={drift:.3f}m  vel={vel:.3f}  "
                          f"EE={int(np.sum(contacts))}/4  "
                          f"sat={int(n_sat)}/{model.nu}  {status}")

                time.sleep(max(0, 0.002 - (time.time() - t0)))

        except KeyboardInterrupt:
            print("\nDone.")

# ──────────────────────────────────────
# Hold debug
# ──────────────────────────────────────

def hold_debug(args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)
    kp, kd, max_tau = build_gains(model)
    act_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]

    jnt_ids = model.actuator_trnid[:, 0]
    act_qa = model.jnt_qposadr[jnt_ids]

    spawn_h = find_spawn_height(model, data, home_qpos)
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, spawn_h]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)

    print("Settling with gradual gravity...")
    gentle_gravity_settle(model, data, home_qpos, kp, kd, max_tau, duration=3.0)

    print(f"\nDebug run — 3s after settle")
    print(f"{'step':>6}  {'h':>6}  {'EE':>4}  {'drift':>6}  top saturating joints")
    print("-" * 80)

    max_drift = np.zeros(model.nu)
    sat_count = np.zeros(model.nu)
    start_xy = data.qpos[0:2].copy()
    N = int(3.0 / model.opt.timestep)

    for s in range(N):
        q_target = home_qpos.copy()
        q_target[0:7] = data.qpos[0:7]
        for i, qa in enumerate(act_qa):
            q_target[qa] = home_qpos[qa]
        tau = pd_control_vec(model, data, q_target, kp, kd, max_tau)
        mujoco.mj_step(model, data)

        drift_j = np.abs(data.qpos[act_qa] - home_qpos[act_qa])
        max_drift = np.maximum(max_drift, drift_j)
        sat_count += (np.abs(tau) >= max_tau * 0.95).astype(float)

        if s % 500 == 0:
            h = data.qpos[2]
            contacts = get_ee_contacts(model, data)
            xy_drift = np.linalg.norm(data.qpos[0:2] - start_xy)
            top3 = np.argsort(drift_j)[::-1][:3]
            top_str = "  ".join(f"{act_names[j]}={drift_j[j]:.3f}" for j in top3 if drift_j[j] > 0.01)
            print(f"  {s:5d}  {h:6.3f}  {int(np.sum(contacts))}/4  {xy_drift:6.3f}  {top_str}")

    print(f"\n── Joint Analysis ──")
    print(f"{'Joint':<35s}  {'max_drift':>10s}  {'sat%':>6s}  {'kp':>7s}  {'max_tau':>8s}")
    print("-" * 75)
    for idx in np.argsort(max_drift)[::-1]:
        name = act_names[idx]
        pct = 100 * sat_count[idx] / N
        print(f"  {name:<35s}  {max_drift[idx]:10.4f}  {pct:5.1f}%  {kp[idx]:7.1f}  {max_tau[idx]:8.1f}")

    bad = [(act_names[i], max_drift[i], 100*sat_count[i]/N)
           for i in range(model.nu) if max_drift[i] > 0.05 and sat_count[i]/N > 0.3]
    if bad:
        print(f"\n⚠ {len(bad)} joints struggling:")
        for name, d, s in bad:
            print(f"  {name}: drift={d:.3f}rad, saturated {s:.0f}% of time")
    else:
        print("\n✅ All joints holding well!")

# ──────────────────────────────────────
# RL Environment
# ──────────────────────────────────────

class G1StandEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    RESIDUAL_SCALE = 0.10
    N_SUBSTEPS = 10
    MAX_STEPS = 500

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

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
        self.jnt_ids = self.model.actuator_trnid[:, 0]

        self.spawn_h = find_spawn_height(self.model, self.data, self.home_qpos)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        # quat(4) + angvel(3) + linvel(3) + joint_off(nu) + joint_vel(nu) + ee_contact(4) + height(1)
        obs_dim = 4 + 3 + 3 + self.model.nu * 2 + 4 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu)
        self.start_xy = np.zeros(2)
        self.settled_height = 0.15
        self.viewer = None

    def _get_obs(self):
        d = self.data
        joint_off = d.qpos[self.act_qpos] - self.home_qpos[self.act_qpos]
        joint_vel = np.array([
            d.qvel[self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]]
            for i in range(self.model.nu)
        ])
        contacts = get_ee_contacts(self.model, self.data)
        return np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3],
            joint_off, joint_vel, contacts, [d.qpos[2]]
        ]).astype(np.float32)

    def _get_reward(self, action):
        d = self.data
        h = d.qpos[2]

        # 1. Height: reward being at settled height
        target_height = self.settled_height + 0.05  # encourage higher

        height_rew = np.exp(-20.0 * (h - target_height) ** 2)
        height_bonus = h
        # 3. EE contacts
        contacts = get_ee_contacts(self.model, self.data)
        contact_rew = np.sum(contacts) / 4.0

        # 4. Belly up
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = max(0.0, -pelvis_rot[2, 2])

        # 5. Hold pose (joint angles close to settled)
        joint_dev = np.sum(np.square(d.qpos[self.act_qpos] - self.home_qpos[self.act_qpos]))
        pose_rew = np.exp(-0.5 * joint_dev / self.model.nu)

        height_drop = max(0.0, self.settled_height - h)
        # 6. Penalties
        vel_xy = np.sum(np.square(d.qvel[0:2]))
        ang_vel = np.sum(np.square(d.qvel[3:6]))
        drift = np.sum(np.square(d.qpos[0:2] - self.start_xy))
        energy = np.sum(np.square(d.ctrl)) / self.model.nu
        act_rate = np.sum(np.square(action - self.prev_action))

        reward = (
            3.0 * height_rew
            + 3.0 * height_bonus
            + 3.0 * contact_rew
            + 1.0 * belly_up
            + 1.0 * pose_rew
            + 0.5  # alive
            - 0.2 * vel_xy
            - 0.1 * ang_vel
            - 0.5 * drift
            - 0.0005 * energy
            - 0.01 * act_rate
            - 2.0 * height_drop**2
        )
        return reward

    def _is_terminated(self):
        if np.any(np.isnan(self.data.qpos)):
            return True
        h = self.data.qpos[2]
        return h > 0.70 or h < -0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Set pose at correct height
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, self.spawn_h]
        self.data.qpos[3:7] = BELLY_UP_QUAT
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02,
                                                          size=self.data.qpos[7:].shape)
        self.data.qvel[:] = 0

        # Gentle gravity settle
        gentle_gravity_settle(self.model, self.data, self.home_qpos,
                              self.kp, self.kd, self.max_tau, duration=1.5)

        self.settled_height = self.data.qpos[2]
        self.start_xy = self.data.qpos[0:2].copy()
        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        q_target = self.home_qpos.copy()
        q_target[0:7] = self.data.qpos[0:7]
        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            q_target[qa] = self.home_qpos[qa] + action[i] * self.RESIDUAL_SCALE
            jid = self.model.actuator_trnid[i, 0]
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

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
# Train / Eval / Test
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = SubprocVecEnv([lambda: G1StandEnv() for _ in range(args.n_envs)])
    eval_env = G1StandEnv()

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        model = PPO.load(args.checkpoint, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
            max_grad_norm=0.5, verbose=1, device=args.device,
            tensorboard_log="logs/",
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-1.5,
            ),
        )

    print(f"\nSpider Stand Training")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs} | Device: {args.device}\n")

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

def evaluate(args):
    from stable_baselines3 import PPO
    if not args.checkpoint:
        print("Error: --checkpoint required")
        return
    model = PPO.load(args.checkpoint)
    env = G1StandEnv(render_mode="human")
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
            if steps % 50 == 0:
                h = env.data.qpos[2]
                contacts = get_ee_contacts(env.model, env.data)
                drift = np.linalg.norm(env.data.qpos[0:2] - env.start_xy)
                print(f"  step={steps}  h={h:.3f}  EE={int(np.sum(contacts))}/4  "
                      f"drift={drift:.3f}  rew={total_reward:.1f}")
            if terminated or truncated:
                episodes += 1
                print(f"\nEp {episodes} — steps={steps} reward={total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

def test_random(args):
    env = G1StandEnv(render_mode="human")
    obs, _ = env.reset()
    print(f"Settled height: {env.settled_height:.3f}m\n")
    total_reward, steps = 0, 0
    try:
        while True:
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample() * 0.1)
            total_reward += reward
            steps += 1
            env.render()
            if steps % 50 == 0:
                h = env.data.qpos[2]
                contacts = get_ee_contacts(env.model, env.data)
                drift = np.linalg.norm(env.data.qpos[0:2] - env.start_xy)
                print(f"  step={steps:4d}  h={h:.3f}  EE={int(np.sum(contacts))}/4  "
                      f"drift={drift:.3f}  rew/step={total_reward/steps:+.2f}")
            if terminated or truncated:
                print(f"Done — steps={steps} total={total_reward:.1f} "
                      f"per_step={total_reward/max(steps,1):+.2f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Stand")
    parser.add_argument("--mode", required=True,
                        choices=["hold", "hold_debug", "train", "eval", "test"])
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    {"hold": hold_pose, "hold_debug": hold_debug,
     "train": train, "eval": evaluate, "test": test_random}[args.mode](args)

if __name__ == "__main__":
    main()
