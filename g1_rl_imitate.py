#!/usr/bin/env python3
"""
G1 Spider Crawl — Imitation Learning (Reference Tracking)

RL learns to reproduce the reference crawling motion under full physics.
The agent outputs small CORRECTIONS to the reference trajectory (residual policy).

Pipeline:
    1. Reference provides target joint angles for current gait phase
    2. RL agent adds small corrections (±0.1 rad) for balance/adaptation
    3. PD controller tracks the corrected targets
    4. Reward: track reference + move in target direction + stay alive

Requirements:
    First generate reference data:
        python g1_crawl_ref.py --save-all

Usage:
    python g1_rl_imitate.py --mode test
    python g1_rl_imitate.py --mode train --steps 5000000 --n_envs 8
    python g1_rl_imitate.py --mode eval --checkpoint models/best/best_model
"""

import argparse, os, time, glob
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

# ──────────────────────────────────────
# Core (same as other files)
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
    # for i in range(model.ngeom):
    #     model.geom_solref[i] = [0.02, 1.0]
    #     model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]
    for i in range(6, model.nv):
        model.dof_damping[i] = 1.0
    # floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    # model.geom_friction[floor_id] = [1.5, 0.005, 0.001]
    # for i in range(model.ngeom):
    #     if i != floor_id:
    #         model.geom_friction[i] = [1.5, 0.005, 0.001]

def build_gains(model):
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 1.0         # stronger tracking (was 0.5)
        kd[i] = kp[i] * 0.3       # moderate damping
    return kp, kd, max_tau

def pd_control_vec(model, data, q_target, kp, kd, max_tau):
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

# ──────────────────────────────────────
# Reference trajectory manager
# ──────────────────────────────────────

class ReferenceLibrary:
    """
    Loads and manages reference trajectories for multiple directions.
    Provides joint targets for any gait phase via interpolation.
    """

    def __init__(self, ref_dir="."):
        self.refs = {}
        self.directions = {}

        # Load all reference files
        patterns = glob.glob(os.path.join(ref_dir, "g1_ref_*.npz"))
        if not patterns:
            raise FileNotFoundError(
                f"No reference files found in '{ref_dir}'. "
                f"Run: python g1_crawl_ref.py --save-all"
            )

        for path in sorted(patterns):
            name = os.path.basename(path).replace("g1_ref_", "").replace(".npz", "")
            ref = np.load(path)
            self.refs[name] = {
                "joint_qpos": ref["joint_qpos"],     # [N, n_joints]
                "ee_positions": ref["ee_positions"],  # [N, 4, 3]
                "times": ref["times"],                # [N]
                "phase": ref["phase"],                # [N]
                "move_dir": ref["move_dir"],           # [2]
                "speed": float(ref["speed"]),
            }
            self.directions[name] = ref["move_dir"].copy()
            print(f"  Loaded {name:15s}: {len(ref['times'])} frames, "
                  f"dir=[{ref['move_dir'][0]:+.2f},{ref['move_dir'][1]:+.2f}]")

        self.dir_names = list(self.refs.keys())
        self.dir_vectors = np.array([self.directions[n] for n in self.dir_names])

        print(f"  Total: {len(self.refs)} references loaded\n")

    def get_closest_ref(self, target_dir):
        """Find the reference whose direction is closest to target_dir."""
        target = np.array(target_dir[:2])
        norm = np.linalg.norm(target)
        if norm < 1e-6:
            # No direction → use stationary if available
            if "stationary" in self.refs:
                return self.refs["stationary"], "stationary"
            return self.refs[self.dir_names[0]], self.dir_names[0]

        target /= norm
        # Dot product with all directions
        dots = self.dir_vectors @ target
        best = np.argmax(dots)
        name = self.dir_names[best]
        return self.refs[name], name

    def get_frame(self, ref, time_in_cycle):
        """
        Get reference joint angles and EE positions at a given time.
        Interpolates between frames. Loops the trajectory.
        """
        times = ref["times"]
        duration = times[-1] + (times[1] - times[0])  # total cycle duration

        # Wrap time to loop
        t = time_in_cycle % duration

        # Find surrounding frames
        idx = np.searchsorted(times, t, side="right") - 1
        idx = max(0, min(idx, len(times) - 2))

        # Linear interpolation
        t0, t1 = times[idx], times[idx + 1]
        dt = t1 - t0
        if dt < 1e-8:
            alpha = 0.0
        else:
            alpha = (t - t0) / dt

        joint_qpos = (1 - alpha) * ref["joint_qpos"][idx] + alpha * ref["joint_qpos"][idx + 1]
        ee_pos = (1 - alpha) * ref["ee_positions"][idx] + alpha * ref["ee_positions"][idx + 1]

        return joint_qpos, ee_pos

# ──────────────────────────────────────
# Environment
# ──────────────────────────────────────

class G1ImitateEnv(gym.Env):
    """
    Imitation learning environment.

    The agent receives the reference joint angles as part of the observation
    and outputs small corrections. The combined target is tracked by PD control.

    Action:  [nu] corrections in [-1, 1], scaled to ±RESIDUAL_SCALE rad
    Obs:     robot state + reference joint angles + gait phase + target direction
    Reward:  reference tracking + velocity + alive - penalties
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    RESIDUAL_SCALE = 0.1      # ±0.1 rad corrections to reference
    CONTROL_DT = 0.02         # 50 Hz
    N_SUBSTEPS = 10           # 10 × 2ms = 20ms
    MAX_STEPS = 1000          # 20s episodes
    SETTLE_STEPS = 50         # 1 second of settling before crawl starts
    BLEND_STEPS = 100         # 2 seconds to ramp from pose → full crawl

    # Reward weights
    W_JOINT_TRACK = 3.0       # joint angle tracking
    W_EE_TRACK = 2.0          # end-effector position tracking
    W_VELOCITY = 2.0          # forward velocity
    W_ALIVE = 0.5             # alive bonus
    W_ORIENTATION = 1.0       # belly-up reward
    W_ENERGY = 0.0005         # energy penalty
    W_SMOOTH = 0.02           # action smoothness

    def __init__(self, render_mode=None, target_dir=None, ref_dir="."):
        super().__init__()
        self.render_mode = render_mode

        td = np.array(target_dir or [1.0, 0.0], dtype=np.float64)
        self.target_dir = td / (np.linalg.norm(td) + 1e-8)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        self.joint_map = get_joint_map(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)
        self.kp, self.kd, self.max_tau = build_gains(self.model)

        # Actuator → qpos mapping
        self.act_qpos = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        # EE site IDs
        self.ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in EE_SITES
        ]
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Load references
        self.ref_lib = ReferenceLibrary(ref_dir)

        # Spaces
        # nu joints for corrections
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # Obs: state + reference + phase + direction
        # quat(4) + angvel(3) + linvel(3)
        # + joint_pos(nu) + joint_vel(nu)
        # + ref_joint_pos(nu) + joint_error(nu)
        # + ee_pos_rel(12) + ref_ee_rel(12)
        # + phase_clock(2) + target_dir(2) + height(1)
        obs_dim = (4 + 3 + 3
                   + self.model.nu * 4
                   + 12 + 12
                   + 2 + 2 + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        # State
        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.model.nu)
        self.current_ref = None
        self.current_ref_name = None
        self.ref_joint_target = np.zeros(self.model.nu)
        self.ref_ee_target = np.zeros((4, 3))
        self.settled_joints = np.zeros(self.model.nu)
        self.viewer = None

    def _get_obs(self):
        d = self.data

        pelvis_quat = d.qpos[3:7]
        pelvis_angvel = d.qvel[3:6]
        pelvis_linvel = d.qvel[0:3]
        pelvis_pos = d.qpos[0:3]

        # Current joint positions and velocities
        joint_pos = d.qpos[self.act_qpos]
        joint_vel = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            va = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            joint_vel[i] = d.qvel[va]

        # Reference joint targets (what we should be tracking)
        ref_joint = self.ref_joint_target

        # Error between current and reference
        joint_error = ref_joint - joint_pos

        # EE positions relative to pelvis
        ee_rel = np.zeros(12)
        for i, sid in enumerate(self.ee_site_ids):
            ee_rel[i*3:(i+1)*3] = d.site_xpos[sid] - pelvis_pos

        # Reference EE positions relative to pelvis
        ref_ee_rel = np.zeros(12)
        for i in range(4):
            ref_ee_rel[i*3:(i+1)*3] = self.ref_ee_target[i] - pelvis_pos

        # Gait phase clock
        speed = self.current_ref["speed"] if self.current_ref else 0.8
        phase = 2 * np.pi * speed * self.sim_time
        clock = np.array([np.sin(phase), np.cos(phase)])

        height = np.array([d.qpos[2]])

        return np.concatenate([
            pelvis_quat, pelvis_angvel, pelvis_linvel,
            joint_pos, joint_vel,
            ref_joint, joint_error,
            ee_rel, ref_ee_rel,
            clock, self.target_dir, height
        ]).astype(np.float32)

    def _update_reference(self):
        """Look up reference joint angles and EE positions for current time."""
        if self.current_ref is None:
            return
        joint_qpos, ee_pos = self.ref_lib.get_frame(self.current_ref, self.sim_time)
        self.ref_joint_target = joint_qpos
        self.ref_ee_target = ee_pos

    def _get_reward(self, action):
        d = self.data

        # ── 1. Joint tracking reward ──
        joint_pos = d.qpos[self.act_qpos]
        joint_error_sq = np.sum(np.square(self.ref_joint_target - joint_pos))
        # Exponential: 1.0 = perfect match, ~0.1 = 0.5 rad avg error
        joint_track = np.exp(-5.0 * joint_error_sq / self.model.nu)

        # ── 2. EE position tracking reward ──
        pelvis_pos = d.qpos[0:3]
        ee_error_sq = 0.0
        for i, sid in enumerate(self.ee_site_ids):
            ee_error_sq += np.sum(np.square(
                d.site_xpos[sid] - pelvis_pos - (self.ref_ee_target[i] - pelvis_pos)
            ))
        ee_track = np.exp(-50.0 * ee_error_sq / 4.0)

        # ── 3. Forward velocity ──
        vel_xy = d.qvel[0:2]
        forward_vel = np.dot(vel_xy, self.target_dir)

        # ── 4. Alive bonus ──
        alive = 1.0

        # ── 5. Orientation: belly up ──
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = max(0.0, -pelvis_rot[2, 2])

        # ── 6. Penalties ──
        energy = np.sum(np.square(d.ctrl)) / self.model.nu
        action_rate = np.sum(np.square(action - self.prev_action))

        reward = (
            self.W_JOINT_TRACK * joint_track
            + self.W_EE_TRACK * ee_track
            + self.W_VELOCITY * forward_vel
            + self.W_ALIVE * alive
            + self.W_ORIENTATION * belly_up
            - self.W_ENERGY * energy
            - self.W_SMOOTH * action_rate
        )

        return reward, {
            "joint_track": joint_track,
            "ee_track": ee_track,
            "forward_vel": forward_vel,
            "belly_up": belly_up,
        }

    def _is_terminated(self):
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True
        h = self.data.qpos[2]
        return h > 0.8 or h < -0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Spider pose, belly up
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]

        # Small random perturbation on joints
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02,
                                                          size=self.data.qpos[7:].shape)
            # Randomize target direction
            if self.np_random.random() < 0.5:
                angle = self.np_random.uniform(-np.pi, np.pi)
                self.target_dir = np.array([np.cos(angle), np.sin(angle)])

        self.data.qvel[:] = 0

        # ── Settle: hold spider pose under gravity for 1 second ──
        q_settle = self.home_qpos.copy()
        for _ in range(self.SETTLE_STEPS * self.N_SUBSTEPS):
            q_settle[0:7] = self.data.qpos[0:7]
            pd_control_vec(self.model, self.data, q_settle,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = 0  # kill any residual velocity from settling
        mujoco.mj_forward(self.model, self.data)

        # Record the settled home joint angles (what the pose looks like under gravity)
        self.settled_joints = self.data.qpos[self.act_qpos].copy()

        # Select reference for this direction
        self.current_ref, self.current_ref_name = self.ref_lib.get_closest_ref(self.target_dir)

        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.model.nu)

        # Initialize reference targets
        self._update_reference()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Look up reference for current time
        self._update_reference()

        # Blend factor: 0 = pure settled pose, 1 = full reference
        blend = min(1.0, self.step_count / max(self.BLEND_STEPS, 1))

        # Target = blend between settled pose and reference, + RL correction
        q_target = self.home_qpos.copy()
        q_target[0:7] = self.data.qpos[0:7]  # live base

        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            # Blend: settled_joints → reference as blend goes 0→1
            blended = (1.0 - blend) * self.settled_joints[i] + blend * self.ref_joint_target[i]
            q_target[qa] = blended + action[i] * self.RESIDUAL_SCALE

            # Clamp to joint limits
            jid = self.model.actuator_trnid[i, 0]
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

        # Run physics with PD
        for _ in range(self.N_SUBSTEPS):
            pd_control_vec(self.model, self.data, q_target,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        # Only advance reference time after blend is complete
        if blend >= 1.0:
            self.sim_time += self.CONTROL_DT
        self.step_count += 1

        reward, info = self._get_reward(action)
        info["blend"] = blend
        terminated = self._is_terminated()
        truncated = self.step_count >= self.MAX_STEPS

        self.prev_action = action.copy()
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

# ──────────────────────────────────────
# Train
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    target = [float(x) for x in args.target]
    ref_dir = args.ref_dir

    # Verify references exist
    if not glob.glob(os.path.join(ref_dir, "g1_ref_*.npz")):
        print("ERROR: No reference files found!")
        print("Run first:  python g1_crawl_ref.py --save-all")
        return

    def make_env():
        def _init():
            return G1ImitateEnv(target_dir=target, ref_dir=ref_dir)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    eval_env = G1ImitateEnv(target_dir=target, ref_dir=ref_dir)

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        print(f"Resuming from {args.checkpoint}")
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
            ent_coef=0.005,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=args.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,
            ),
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(200_000 // args.n_envs, 1),
            save_path="models/",
            name_prefix="g1_imitate",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path="models/best/",
            log_path="logs/eval/",
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    print(f"\nImitation Learning Training")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs} | Device: {args.device}")
    print(f"  Reference dir: {ref_dir}")
    print(f"  Action: residual corrections ±{G1ImitateEnv.RESIDUAL_SCALE} rad")
    print(f"  Reward: track reference + velocity + alive\n")

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save("models/g1_imitate_final")
    print("\nSaved to models/g1_imitate_final")
    env.close()

# ──────────────────────────────────────
# Eval
# ──────────────────────────────────────

def evaluate(args):
    from stable_baselines3 import PPO

    target = [float(x) for x in args.target]
    if not args.checkpoint:
        print("Error: --checkpoint required")
        return

    model = PPO.load(args.checkpoint)
    env = G1ImitateEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Evaluating — target: {target}, ref: {env.current_ref_name}")
    total_reward, steps, episodes = 0, 0, 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.02)

            if steps % 50 == 0:
                jt = info.get("joint_track", 0)
                et = info.get("ee_track", 0)
                fv = info.get("forward_vel", 0)
                print(f"  step={steps:4d}  rew={total_reward:7.1f}  "
                      f"j_track={jt:.3f}  ee_track={et:.3f}  "
                      f"fwd_vel={fv:+.3f}  ref={env.current_ref_name}")

            if terminated or truncated:
                episodes += 1
                print(f"\nEp {episodes} — steps: {steps}, reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print(f"\nDone. {episodes} episodes.")

# ──────────────────────────────────────
# Test (random agent)
# ──────────────────────────────────────

def test_random(args):
    target = [float(x) for x in args.target]
    env = G1ImitateEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Random agent test — ref: {env.current_ref_name}")
    print("Even with random corrections, the reference provides base motion\n")

    total_reward, steps = 0, 0

    try:
        while True:
            # Very small random corrections — mostly just follow reference
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if steps % 100 == 0:
                jt = info.get("joint_track", 0)
                et = info.get("ee_track", 0)
                print(f"  step={steps:4d}  rew={total_reward:7.1f}  "
                      f"j_track={jt:.3f}  ee_track={et:.3f}  "
                      f"h={env.data.qpos[2]:.3f}  "
                      f"vel=[{env.data.qvel[0]:+.3f},{env.data.qvel[1]:+.3f}]")

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nDone.")

# ──────────────────────────────────────
# Playback reference (no RL, just replay the reference under physics)
# ──────────────────────────────────────

def playback(args):
    """Replay reference trajectory under full physics (no RL corrections)."""
    target = [float(x) for x in args.target]
    env = G1ImitateEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Playback reference: {env.current_ref_name}")
    print(f"Settled, now blending in crawl motion over {env.BLEND_STEPS} steps...")
    print("Pure reference tracking (zero RL corrections)\n")

    total_reward, steps = 0, 0

    try:
        while True:
            action = np.zeros(env.model.nu, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.02)

            if steps % 25 == 0:
                jt = info.get("joint_track", 0)
                fv = info.get("forward_vel", 0)
                bl = info.get("blend", 0)
                phase = "BLEND" if bl < 1.0 else "CRAWL"
                print(f"  step={steps:4d}  [{phase:5s} {bl:.0%}]  "
                      f"j_track={jt:.3f}  fwd_vel={fv:+.3f}  "
                      f"h={env.data.qpos[2]:.3f}  "
                      f"vel=[{env.data.qvel[0]:+.3f},{env.data.qvel[1]:+.3f}]")

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nDone.")

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider — Imitation Learning")
    parser.add_argument("--mode", choices=["train", "eval", "test", "playback"], required=True,
                        help="train: RL | eval: run model | test: random | playback: pure reference")
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--target", nargs=2, default=["1.0", "0.0"])
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ref_dir", type=str, default=".",
                        help="directory containing g1_ref_*.npz files")

    args = parser.parse_args()

    {"train": train, "eval": evaluate, "test": test_random, "playback": playback}[args.mode](args)

if __name__ == "__main__":
    main()
