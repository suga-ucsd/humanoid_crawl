#!/usr/bin/env python3
"""
G1 Spider Crawl — Imitation Learning v3

Robust environment for learning spider crawl via reference tracking.
Comprehensive reward, healthy termination, diagnostic modes.

Usage:
    python g1_rl_imitate.py --mode playback    # pure reference, watch + diagnose
    python g1_rl_imitate.py --mode test         # random agent, must be positive
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
EE_LABELS = ["LF", "RF", "LH", "RH"]

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

# Joints that must stay "healthy" — if they collapse, terminate
CRITICAL_JOINTS = [
    "left_knee_joint", "right_knee_joint",
    "left_elbow_joint", "right_elbow_joint",
]

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
        pass
    return q

def configure_physics(model):
    model.opt.timestep = 0.002
    model.opt.iterations = 20
    model.opt.ls_iterations = 10
    model.opt.solver = 2
    for i in range(6, model.nv):
        model.dof_damping[i] = 0.5
    for i in range(model.ngeom):
        if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
            model.geom_friction[i] = [3.0, 0.5, 0.5]

def build_gains(model):
    max_tau = np.zeros(model.nu)
    kp = np.zeros(model.nu)
    kd = np.zeros(model.nu)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        frc = TORQUE_LIMITS.get(name, 25.0)
        max_tau[i] = frc
        kp[i] = frc * 1.0
        kd[i] = kp[i] * 0.3
    return kp, kd, max_tau

def pd_control_vec(model, data, q_target, kp, kd, max_tau):
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

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

def get_torso_floor_contact(model, data):
    """Check if torso or pelvis is touching the floor (bad)."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    torso_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    bad_bodies = {pelvis_body, torso_body}
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = c.geom1, c.geom2
        if g1 != floor_id and g2 != floor_id:
            continue
        robot_geom = g2 if g1 == floor_id else g1
        body = model.geom_bodyid[robot_geom]
        if body in bad_bodies:
            return True
    return False

# ──────────────────────────────────────
# Reference Library
# ──────────────────────────────────────

class ReferenceLibrary:
    def __init__(self, ref_dir="."):
        self.refs = {}
        self.directions = {}
        patterns = sorted(glob.glob(os.path.join(ref_dir, "g1_ref_*.npz")))
        if not patterns:
            raise FileNotFoundError(
                f"No reference files in '{ref_dir}'. Run: python g1_crawl_ref.py --save-all")
        for path in patterns:
            name = os.path.basename(path).replace("g1_ref_", "").replace(".npz", "")
            ref = np.load(path)
            self.refs[name] = {
                "joint_qpos": ref["joint_qpos"],
                "ee_positions": ref["ee_positions"],
                "times": ref["times"],
                "phase": ref["phase"],
                "move_dir": ref["move_dir"],
                "speed": float(ref["speed"]),
            }
            self.directions[name] = ref["move_dir"].copy()
        self.dir_names = list(self.refs.keys())
        self.dir_vectors = np.array([self.directions[n] for n in self.dir_names])
        print(f"  Loaded {len(self.refs)} references")

    def get_closest_ref(self, target_dir):
        target = np.array(target_dir[:2], dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-6:
            if "stationary" in self.refs:
                return self.refs["stationary"], "stationary"
            return self.refs[self.dir_names[0]], self.dir_names[0]
        target /= norm
        best = int(np.argmax(self.dir_vectors @ target))
        name = self.dir_names[best]
        return self.refs[name], name

    def get_frame(self, ref, t):
        times = ref["times"]
        dt_f = times[1] - times[0] if len(times) > 1 else 0.02
        dur = times[-1] + dt_f
        t = t % dur
        idx = max(0, min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2))
        t0, t1 = times[idx], times[idx + 1]
        a = 0.0 if (t1 - t0) < 1e-8 else (t - t0) / (t1 - t0)
        jq = (1 - a) * ref["joint_qpos"][idx] + a * ref["joint_qpos"][idx + 1]
        ee = (1 - a) * ref["ee_positions"][idx] + a * ref["ee_positions"][idx + 1]
        return jq, ee

# ──────────────────────────────────────
# Environment
# ──────────────────────────────────────

class G1UnifiedEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    RESIDUAL_SCALE = 0.10
    CONTROL_DT = 0.02
    N_SUBSTEPS = 10
    MAX_STEPS = 1500
    SETTLE_STEPS = 50
    BLEND_STEPS = 100

    # ── Height thresholds ──
    MIN_HEIGHT = 0.04     # below this = collapsed
    MAX_HEIGHT = 0.60     # above this = launched
    TARGET_HEIGHT = 0.10  # desired pelvis height

    def __init__(self, render_mode=None, target_dir=None, ref_dir="."):
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
        self.act_dof = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])
        self.act_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        self.ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n) for n in EE_SITES
        ]

        # Critical joint indices for health check
        self.critical_act_ids = []
        for cj in CRITICAL_JOINTS:
            for i, name in enumerate(self.act_names):
                if name == cj:
                    self.critical_act_ids.append(i)

        self.ref_lib = ReferenceLibrary(ref_dir)

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # Obs: quat(4) + angvel(3) + linvel(3) + joint_pos(nu) + joint_vel(nu)
        #      + ref_joint(nu) + joint_error(nu) + ee_rel(12) + ref_ee_rel(12)
        #      + ee_contacts(4) + clock(2) + target_dir(2) + height(1)
        obs_dim = (4 + 3 + 3 + self.model.nu * 4 + 12 + 12 + 4 + 2 + 2 + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.model.nu)
        self.current_ref = None
        self.current_ref_name = None
        self.ref_joint_target = np.zeros(self.model.nu)
        self.ref_ee_target = np.zeros((4, 3))
        self.settled_joints = np.zeros(self.model.nu)
        self.start_xy = np.zeros(2)
        self.viewer = None

    # ── Observation ──

    def _get_obs(self):
        d = self.data
        pp = d.qpos[0:3]

        joint_pos = d.qpos[self.act_qpos]
        joint_vel = d.qvel[self.act_dof]
        joint_error = self.ref_joint_target - joint_pos

        ee_rel = np.zeros(12)
        ref_ee_rel = np.zeros(12)
        for i, sid in enumerate(self.ee_site_ids):
            ee_rel[i*3:(i+1)*3] = d.site_xpos[sid] - pp
            ref_ee_rel[i*3:(i+1)*3] = self.ref_ee_target[i] - pp

        contacts = get_ee_contacts(self.model, self.data)

        speed = self.current_ref["speed"] if self.current_ref else 0.8
        phase = 2 * np.pi * speed * self.sim_time
        clock = np.array([np.sin(phase), np.cos(phase)])

        return np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3],
            joint_pos, joint_vel,
            self.ref_joint_target, joint_error,
            ee_rel, ref_ee_rel,
            contacts, clock, self.target_dir, [d.qpos[2]]
        ]).astype(np.float32)

    # ── Reference update ──

    def _update_reference(self):
        if self.current_ref is None:
            self.ref_joint_target = self.home_qpos[self.act_qpos]
            return
        jq, ee = self.ref_lib.get_frame(self.current_ref, self.sim_time)
        self.ref_joint_target = jq
        self.ref_ee_target = ee

    # ── Reward ──

    def _get_reward(self, action):
        d = self.data

        # 1. JOINT TRACKING — how well do joints match the reference
        joint_pos = d.qpos[self.act_qpos]
        j_err_sq = np.sum(np.square(self.ref_joint_target - joint_pos))
        joint_track = np.exp(-5.0 * j_err_sq / self.model.nu)

        # 2. EE TRACKING — end effector position matching
        ee_err_sq = 0.0
        for i, sid in enumerate(self.ee_site_ids):
            ee_err_sq += np.sum(np.square(d.site_xpos[sid] - self.ref_ee_target[i]))
        ee_track = np.exp(-40.0 * ee_err_sq / 4.0)

        # 3. FORWARD VELOCITY — along target direction
        forward_vel = float(np.dot(d.qvel[0:2], self.target_dir))
        # Clip to avoid rewarding excessive speed
        forward_rew = np.clip(forward_vel, -0.5, 1.0)

        # 4. HEIGHT — pelvis should be above ground but not too high
        h = d.qpos[2]
        height_rew = np.exp(-30.0 * (h - self.TARGET_HEIGHT) ** 2)

        # 5. BELLY UP — orientation
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = max(0.0, -pelvis_rot[2, 2])

        # 6. EE CONTACTS — at least 3 on ground
        contacts = get_ee_contacts(self.model, self.data)
        n_contacts = np.sum(contacts)
        contact_rew = min(n_contacts, 3) / 3.0

        # 7. TORSO NOT ON GROUND — penalty if body drags
        torso_on_ground = get_torso_floor_contact(self.model, self.data)
        torso_penalty = 1.0 if torso_on_ground else 0.0

        # 8. JOINT HEALTH — critical joints near their targets
        crit_err = 0.0
        for ci in self.critical_act_ids:
            qa = self.act_qpos[ci]
            crit_err += abs(d.qpos[qa] - self.ref_joint_target[ci])
        joint_health = np.exp(-3.0 * crit_err / max(len(self.critical_act_ids), 1))

        # 9. ENERGY — small penalty
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        # 10. SMOOTHNESS — penalize jerky actions
        action_rate = np.sum(np.square(action - self.prev_action))

        # 11. DRIFT — don't slide sideways
        lateral_vel = abs(d.qvel[0] * self.target_dir[1] - d.qvel[1] * self.target_dir[0])

        # 12. ANGULAR VELOCITY — don't spin
        ang_vel = np.sum(np.square(d.qvel[3:6]))

        # 13. ALIVE — constant positive baseline
        alive = 1.0

        reward = (
            3.0 * joint_track       # track reference joints
            + 2.0 * ee_track        # track reference EE positions
            + 3.0 * forward_rew     # move in target direction
            + 1.0 * height_rew      # maintain height
            + 1.0 * belly_up        # stay belly up
            + 1.5 * contact_rew     # EEs on ground
            + 1.0 * joint_health    # knees/elbows alive
            + 0.5 * alive           # exist
            - 3.0 * torso_penalty   # torso must NOT touch ground
            - 0.5 * lateral_vel     # don't slide sideways
            - 0.1 * ang_vel         # don't spin
            - 0.0005 * energy       # efficiency
            - 0.02 * action_rate    # smooth actions
        )

        info = {
            "joint_track": float(joint_track),
            "ee_track": float(ee_track),
            "forward_vel": float(forward_vel),
            "height": float(h),
            "belly_up": float(belly_up),
            "n_contacts": int(n_contacts),
            "torso_ground": torso_on_ground,
            "joint_health": float(joint_health),
            "reward": float(reward),
        }
        return reward, info

    # ── Termination ──

    def _is_terminated(self):
        d = self.data

        # NaN check
        if np.any(np.isnan(d.qpos)) or np.any(np.isnan(d.qvel)):
            return True, "NaN"

        h = d.qpos[2]

        # Height bounds
        if h < self.MIN_HEIGHT:
            return True, f"collapsed (h={h:.3f})"
        if h > self.MAX_HEIGHT:
            return True, f"launched (h={h:.3f})"

        # Orientation: belly must face up
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = -pelvis_rot[2, 2]
        if belly_up < 0.0:
            return True, f"flipped (belly_up={belly_up:.2f})"

        # Critical joints: check if knees/elbows have collapsed to limits
        for ci in self.critical_act_ids:
            qa = self.act_qpos[ci]
            jid = self.model.actuator_trnid[ci, 0]
            lo, hi = self.model.jnt_range[jid]
            pos = d.qpos[qa]
            # If joint is jammed at lower or upper limit
            if lo < hi:
                margin = 0.05 * (hi - lo)  # 5% of range
                if pos < lo + margin or pos > hi - margin:
                    name = self.act_names[ci]
                    return True, f"joint_limit ({name}={pos:.2f}, range=[{lo:.2f},{hi:.2f}])"

        return False, ""

    # ── Reset ──

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]

        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02,
                                                          size=self.data.qpos[7:].shape)
            if self.np_random.random() < 0.3:
                angle = self.np_random.uniform(-np.pi, np.pi)
                self.target_dir = np.array([np.cos(angle), np.sin(angle)])

        self.data.qvel[:] = 0

        # Settle: hold spider pose under gravity
        q_settle = self.home_qpos.copy()
        for _ in range(self.SETTLE_STEPS * self.N_SUBSTEPS):
            q_settle[0:7] = self.data.qpos[0:7]
            pd_control_vec(self.model, self.data, q_settle,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.settled_joints = self.data.qpos[self.act_qpos].copy()
        self.start_xy = self.data.qpos[0:2].copy()

        self.current_ref, self.current_ref_name = self.ref_lib.get_closest_ref(self.target_dir)

        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.model.nu)

        self._update_reference()
        return self._get_obs(), {}

    # ── Step ──

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._update_reference()

        blend = min(1.0, self.step_count / max(self.BLEND_STEPS, 1))

        q_target = self.home_qpos.copy()
        q_target[0:7] = self.data.qpos[0:7]

        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            blended = (1.0 - blend) * self.settled_joints[i] + blend * self.ref_joint_target[i]
            q_target[qa] = blended + action[i] * self.RESIDUAL_SCALE
            jid = self.model.actuator_trnid[i, 0]
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

        for _ in range(self.N_SUBSTEPS):
            pd_control_vec(self.model, self.data, q_target,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        if blend >= 1.0:
            self.sim_time += self.CONTROL_DT
        self.step_count += 1

        reward, info = self._get_reward(action)
        info["blend"] = float(blend)

        terminated, term_reason = self._is_terminated()
        if terminated:
            info["term_reason"] = term_reason
        truncated = self.step_count >= self.MAX_STEPS

        self.prev_action = action.copy()
        return self._get_obs(), reward, terminated, truncated, info

    # ── Render ──

    def render(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

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

    if not glob.glob(os.path.join(ref_dir, "g1_ref_*.npz")):
        print("ERROR: No reference files! Run: python g1_crawl_ref.py --save-all")
        return

    def make_env():
        def _init():
            return G1UnifiedEnv(target_dir=target, ref_dir=ref_dir)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    eval_env = G1UnifiedEnv(target_dir=target, ref_dir=ref_dir)

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        model = PPO.load(args.checkpoint, env=env, device=args.device)
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4,
            n_steps=2048,
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
            save_path="models/", name_prefix="g1_crawl"),
        EvalCallback(
            eval_env, best_model_save_path="models/best/",
            log_path="logs/eval/",
            eval_freq=max(50_000 // args.n_envs, 1),
            n_eval_episodes=5, deterministic=True),
    ]

    print(f"\nCrawl Imitation Training")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs} | Device: {args.device}")
    print(f"  Ref dir: {ref_dir}\n")

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save("models/g1_crawl_final")
    env.close()
    eval_env.close()

# ──────────────────────────────────────
# Evaluate
# ──────────────────────────────────────

def evaluate(args):
    from stable_baselines3 import PPO

    target = [float(x) for x in args.target]
    if not args.checkpoint:
        print("Error: --checkpoint required")
        return

    model = PPO.load(args.checkpoint)
    env = G1UnifiedEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Evaluating — ref: {env.current_ref_name}")
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
                _print_status(steps, total_reward, info, env)

            if terminated or truncated:
                episodes += 1
                reason = info.get("term_reason", "truncated")
                print(f"\n  Episode {episodes} [{reason}] — steps: {steps}, "
                      f"reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        env.close()

# ──────────────────────────────────────
# Test (random agent) — MUST be positive
# ──────────────────────────────────────

def test_random(args):
    target = [float(x) for x in args.target]
    env = G1UnifiedEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Random agent test — ref: {env.current_ref_name}")
    print(f"  Settled height: {env.data.qpos[2]:.3f}m")
    print(f"  Target dir: {env.target_dir}")
    print(f"\n  PASS criteria: rew/step > 0, episodes > 300 steps\n")

    total_reward, steps = 0, 0
    ep_count = 0
    ep_lengths = []
    ep_rewards = []

    try:
        while True:
            action = env.action_space.sample() * 0.05  # very small corrections
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if steps % 100 == 0:
                _print_status(steps, total_reward, info, env)

            if terminated or truncated:
                ep_count += 1
                ep_lengths.append(steps)
                ep_rewards.append(total_reward)
                per_step = total_reward / max(steps, 1)
                reason = info.get("term_reason", "truncated")

                verdict = "✅ PASS" if steps >= 300 and per_step > 0 else "❌ FAIL"
                print(f"\n  Episode {ep_count} [{reason}] — steps: {steps}, "
                      f"rew: {total_reward:.1f}, rew/step: {per_step:+.2f}  {verdict}")

                if ep_count >= 3:
                    avg_len = np.mean(ep_lengths)
                    avg_rew = np.mean(ep_rewards)
                    avg_ps = np.mean([r/max(l,1) for r, l in zip(ep_rewards, ep_lengths)])
                    print(f"\n  ── Summary ({ep_count} episodes) ──")
                    print(f"    Avg length: {avg_len:.0f} steps")
                    print(f"    Avg reward: {avg_rew:.1f}")
                    print(f"    Avg rew/step: {avg_ps:+.2f}")
                    overall = "✅ READY TO TRAIN" if avg_ps > 0 and avg_len > 200 else "❌ NOT READY"
                    print(f"    Verdict: {overall}\n")

                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        env.close()

# ──────────────────────────────────────
# Playback — pure reference, zero RL
# ──────────────────────────────────────

def playback(args):
    target = [float(x) for x in args.target]
    env = G1UnifiedEnv(render_mode="human", target_dir=target, ref_dir=args.ref_dir)
    obs, _ = env.reset()

    print(f"Playback — ref: {env.current_ref_name}")
    print(f"  Settled height: {env.data.qpos[2]:.3f}m")
    print(f"  Blend: {env.BLEND_STEPS} steps ({env.BLEND_STEPS * env.CONTROL_DT:.1f}s)")
    print(f"  Zero RL corrections — pure reference tracking\n")
    print(f"  {'step':>5}  {'phase':>6}  {'j_trk':>5}  {'ee_trk':>6}  "
          f"{'fwd_v':>6}  {'h':>5}  {'EE':>4}  {'torso':>5}  {'j_hlth':>6}  {'rew':>6}")
    print("  " + "-" * 75)

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
                bl = info.get("blend", 0)
                phase = "BLEND" if bl < 1.0 else "CRAWL"
                jt = info.get("joint_track", 0)
                et = info.get("ee_track", 0)
                fv = info.get("forward_vel", 0)
                h = info.get("height", 0)
                nc = info.get("n_contacts", 0)
                tg = "YES" if info.get("torso_ground", False) else "no"
                jh = info.get("joint_health", 0)
                r = info.get("reward", 0)

                print(f"  {steps:5d}  {phase:>6}  {jt:5.3f}  {et:6.3f}  "
                      f"{fv:+6.3f}  {h:5.3f}  {nc:2d}/4  {tg:>5}  {jh:6.3f}  {r:+6.2f}")

            if terminated or truncated:
                reason = info.get("term_reason", "truncated")
                per_step = total_reward / max(steps, 1)
                print(f"\n  Episode done [{reason}] — steps: {steps}, "
                      f"rew: {total_reward:.1f}, rew/step: {per_step:+.2f}")

                if reason == "truncated":
                    print(f"  ✅ Survived full episode!")
                else:
                    print(f"  ⚠ Terminated early: {reason}")

                contacts = get_ee_contacts(env.model, env.data)
                print(f"  Final: h={env.data.qpos[2]:.3f}m, "
                      f"EE={contacts.astype(int).tolist()}, "
                      f"vel=[{env.data.qvel[0]:+.3f},{env.data.qvel[1]:+.3f}]")

                # Critical joint check
                print(f"  Critical joints:")
                for ci in env.critical_act_ids:
                    qa = env.act_qpos[ci]
                    name = env.act_names[ci]
                    pos = env.data.qpos[qa]
                    ref = env.ref_joint_target[ci]
                    jid = env.model.actuator_trnid[ci, 0]
                    lo, hi = env.model.jnt_range[jid]
                    print(f"    {name:30s}: pos={pos:+.3f}  ref={ref:+.3f}  "
                          f"range=[{lo:.2f},{hi:.2f}]")
                print()

                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        env.close()

# ──────────────────────────────────────
# Shared status printer
# ──────────────────────────────────────

def _print_status(steps, total_reward, info, env):
    jt = info.get("joint_track", 0)
    et = info.get("ee_track", 0)
    fv = info.get("forward_vel", 0)
    h = info.get("height", 0)
    nc = info.get("n_contacts", 0)
    jh = info.get("joint_health", 0)
    tg = "⚠TRS" if info.get("torso_ground", False) else ""
    bl = info.get("blend", 1)
    ps = total_reward / max(steps, 1)

    print(f"  step={steps:4d}  rew/s={ps:+.2f}  j={jt:.2f}  ee={et:.2f}  "
          f"fwd={fv:+.3f}  h={h:.3f}  EE={nc}/4  jh={jh:.2f}  "
          f"bl={bl:.0%} {tg}")

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl — Imitation Learning")
    parser.add_argument("--mode", choices=["train", "eval", "test", "playback"], required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--target", nargs=2, default=["1.0", "0.0"])
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ref_dir", type=str, default=".")
    args = parser.parse_args()

    {"train": train, "eval": evaluate, "test": test_random, "playback": playback}[args.mode](args)

if __name__ == "__main__":
    main()
