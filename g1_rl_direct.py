#!/usr/bin/env python3
"""
G1 Spider Crawl — Option A: Direct Joint Control

RL directly outputs joint angle offsets from spider pose.
No IK. PD controller tracks the target angles.

Usage:
    python g1_rl_direct.py --mode test
    python g1_rl_direct.py --mode train --steps 10000000 --n_envs 8
    python g1_rl_direct.py --mode eval --checkpoint models/g1_direct_final
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

# Diagonal pairs for trot gait reward
# Pair 0: left_foot + right_hand (should move together)
# Pair 1: right_foot + left_hand (should move together)
DIAG_PAIRS = [(0, 3), (1, 2)]  # indices into EE_SITES

# ──────────────────────────────────────
# Core functions
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
    model.opt.timestep = 0.001
    model.opt.iterations = 30
    model.opt.ls_iterations = 30
    model.opt.solver = 2
    for i in range(model.ngeom):
        model.geom_solref[i] = [0.02, 1.0]
        model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]
    for i in range(6, model.nv):
        model.dof_damping[i] = 0.5
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
    """Vectorized PD — no Python loop."""
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

# ──────────────────────────────────────
# Environment
# ──────────────────────────────────────

class G1DirectEnv(gym.Env):
    """
    Direct joint control for spider crawling.

    Action:  nu-dimensional, scaled to ±ACTION_SCALE radians offset from spider pose
    Obs:     pelvis orientation + angular/linear vel + joint pos/vel + gait clock + target dir
    Reward:  forward vel + alive + orientation + contact cycling + energy penalty
    """

    ACTION_SCALE = 0.3        # ±0.3 rad max offset from spider pose
    CONTROL_DT = 0.02         # 50 Hz control
    PHYSICS_DT = 0.001        # 1000 Hz physics
    N_SUBSTEPS = 20
    MAX_STEPS = 500           # 10s episodes
    GAIT_FREQ = 1.5           # Hz — target stepping frequency

    def __init__(self, render_mode=None, target_dir=None):
        super().__init__()
        self.render_mode = render_mode

        # Target crawl direction (default: +X)
        td = np.array(target_dir or [1.0, 0.0], dtype=np.float64)
        self.target_dir = td / (np.linalg.norm(td) + 1e-8)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        self.joint_map = get_joint_map(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)
        self.kp, self.kd, self.max_tau = build_gains(self.model)

        # Build actuator -> qpos mapping for fast indexing
        self.act_qpos = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        # EE site IDs
        self.ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in EE_SITES
        ]

        # Floor geom ID for contact detection
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        # Spaces
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        obs_size = self._build_obs_dummy()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        # State
        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(self.model.nu)
        self.prev_torques = np.zeros(self.model.nu)
        self.viewer = None

    def _build_obs_dummy(self):
        """Count observation dimensions."""
        # pelvis_quat(4) + pelvis_angvel(3) + pelvis_linvel(3)
        # + joint_pos_offset(nu) + joint_vel(nu)
        # + ee_heights(4) + ee_contact(4)
        # + gait_clock(2) + target_dir(2) + base_height(1)
        return 4 + 3 + 3 + self.model.nu + self.model.nu + 4 + 4 + 2 + 2 + 1

    def _get_ee_heights(self):
        heights = np.zeros(4)
        for i, sid in enumerate(self.ee_site_ids):
            heights[i] = self.data.site_xpos[sid][2]
        return heights

    def _get_ee_contacts(self):
        """Binary: is each EE's parent body in contact with ground?"""
        contacts = np.zeros(4)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 != self.floor_id and g2 != self.floor_id:
                continue
            # Which robot geom touched the floor?
            robot_geom = g2 if g1 == self.floor_id else g1
            robot_body = self.model.geom_bodyid[robot_geom]
            # Check if this body is in any EE's kinematic chain
            for j, sid in enumerate(self.ee_site_ids):
                ee_body = self.model.site_bodyid[sid]
                # Walk up the body tree to see if robot_body is an ancestor
                b = ee_body
                while b > 0:
                    if b == robot_body:
                        contacts[j] = 1.0
                        break
                    b = self.model.body_parentid[b]
        return contacts

    def _get_obs(self):
        d = self.data

        pelvis_quat = d.qpos[3:7].copy()
        pelvis_angvel = d.qvel[3:6].copy()
        pelvis_linvel = d.qvel[0:3].copy()

        # Joint positions as offset from home pose
        joint_offset = np.zeros(self.model.nu)
        joint_vel = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            va = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            joint_offset[i] = d.qpos[qa] - self.home_qpos[qa]
            joint_vel[i] = d.qvel[va]

        ee_heights = self._get_ee_heights()
        ee_contacts = self._get_ee_contacts()

        # Gait phase clock — gives the agent a sense of rhythm
        gait_clock = np.array([np.sin(self.phase), np.cos(self.phase)])

        base_height = np.array([d.qpos[2]])

        obs = np.concatenate([
            pelvis_quat, pelvis_angvel, pelvis_linvel,
            joint_offset, joint_vel,
            ee_heights, ee_contacts,
            gait_clock, self.target_dir, base_height
        ])
        return obs.astype(np.float32)

    def _get_reward(self, action):
        d = self.data

        # ── 1. Forward velocity (main reward) ──
        vel_xy = d.qvel[0:2]
        forward_vel = np.dot(vel_xy, self.target_dir)

        # ── 2. Alive bonus ──
        # Pelvis z-axis in world: want it pointing down (belly up)
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = -pelvis_rot[2, 2]  # +1 when belly fully up
        alive = float(belly_up > 0.0)

        # ── 3. Height reward ──
        height = d.qpos[2]
        height_rew = -5.0 * (height - 0.15) ** 2  # quadratic penalty around 15cm

        # ── 4. Contact cycling reward ──
        # Encourage alternating diagonal contacts (trot gait)
        contacts = self._get_ee_contacts()
        phase_signal = np.sin(self.phase)
        # Pair 0 (LF+RH) should be down when phase_signal > 0
        # Pair 1 (RF+LH) should be down when phase_signal < 0
        pair0_down = contacts[DIAG_PAIRS[0][0]] + contacts[DIAG_PAIRS[0][1]]
        pair1_down = contacts[DIAG_PAIRS[1][0]] + contacts[DIAG_PAIRS[1][1]]

        if phase_signal > 0:
            contact_rew = 0.5 * pair0_down - 0.5 * pair1_down
        else:
            contact_rew = 0.5 * pair1_down - 0.5 * pair0_down

        # ── 5. Orientation stability ──
        # Penalize roll and pitch deviation from flat
        angular_vel_penalty = np.sum(np.square(d.qvel[3:6]))

        # ── 6. Energy penalty ──
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        # ── 7. Smoothness penalty ──
        action_rate = np.sum(np.square(action - self.prev_action))

        # ── 8. Pose regularization ──
        # Don't deviate too far from spider pose
        joint_dev = 0.0
        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            joint_dev += (d.qpos[qa] - self.home_qpos[qa]) ** 2
        joint_dev /= self.model.nu

        reward = (
            3.0 * forward_vel          # main objective
            + 0.5 * alive              # stay belly up
            + 0.3 * height_rew         # maintain height
            + 0.5 * contact_rew        # gait coordination
            - 0.01 * angular_vel_penalty  # don't spin wildly
            - 0.001 * energy           # efficiency
            - 0.05 * action_rate       # smooth actions
            - 0.1 * joint_dev          # stay near spider pose
        )

        return reward

    def _is_terminated(self):
        h = self.data.qpos[2]
        if h > 0.5 or h < 0.02:
            return True
        if np.any(np.isnan(self.data.qpos)):
            return True
        # Flipped over (belly no longer up)
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = self.data.xmat[pelvis_id].reshape(3, 3)
        if pelvis_rot[2, 2] > 0.3:  # belly facing down = bad
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Spider pose, belly up
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]

        # Small random perturbation
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.03, 0.03,
                                                          size=self.data.qpos[7:].shape)
            # Random target direction for curriculum
            if self.np_random.random() < 0.3:
                angle = self.np_random.uniform(-np.pi, np.pi)
                self.target_dir = np.array([np.cos(angle), np.sin(angle)])

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0
        self.prev_action = np.zeros(self.model.nu)
        self.prev_torques = np.zeros(self.model.nu)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Target = spider pose + scaled action offset
        q_target = self.home_qpos.copy()
        for i in range(self.model.nu):
            qa = self.act_qpos[i]
            q_target[qa] += action[i] * self.ACTION_SCALE

            # Clamp to joint limits
            jid = self.model.actuator_trnid[i, 0]
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

        # Keep base from q_target as current sim state
        q_target[0:7] = self.data.qpos[0:7]

        # Run physics with PD
        for _ in range(self.N_SUBSTEPS):
            pd_control_vec(self.model, self.data, q_target,
                           self.kp, self.kd, self.max_tau)
            mujoco.mj_step(self.model, self.data)

        # Advance gait clock
        self.phase += 2 * np.pi * self.GAIT_FREQ * self.CONTROL_DT
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi

        self.step_count += 1
        reward = self._get_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= self.MAX_STEPS

        self.prev_action = action.copy()
        self.prev_torques = self.data.ctrl.copy()

        return self._get_obs(), reward, terminated, truncated, {}

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

    def make_env():
        def _init():
            return G1DirectEnv(target_dir=target)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])

    # Separate eval env
    eval_env = G1DirectEnv(target_dir=target)

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
                log_std_init=-1.0,  # start with smaller action noise
            ),
        )

    save_cb = CheckpointCallback(
        save_freq=max(100_000 // args.n_envs, 1),
        save_path="models/",
        name_prefix="g1_direct",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best/",
        log_path="logs/eval/",
        eval_freq=max(50_000 // args.n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )

    print(f"\nTraining for {args.steps:,} steps | {args.n_envs} envs | device: {args.device}")
    print(f"Target direction: {target}\n")

    model.learn(
        total_timesteps=args.steps,
        callback=[save_cb, eval_cb],
        progress_bar=True,
    )
    model.save("models/g1_direct_final")
    print("\nSaved to models/g1_direct_final")
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
    env = G1DirectEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()

    print(f"Evaluating — target: {target}")

    total_reward = 0
    steps = 0
    episodes = 0

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
                print(f"Episode {episodes} — steps: {steps}, reward: {total_reward:.2f}")
                total_reward = 0
                steps = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print(f"\nDone. {episodes} episodes.")

# ──────────────────────────────────────
# Test (random agent)
# ──────────────────────────────────────

def test_random(args):
    target = [float(x) for x in args.target]
    env = G1DirectEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()

    print("Random agent test — Ctrl+C to stop\n")

    total_reward = 0
    steps = 0

    try:
        while True:
            action = env.action_space.sample() * 0.3
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if steps % 50 == 0:
                contacts = env._get_ee_contacts()
                print(f"  step={steps:4d}  rew={total_reward:7.2f}  "
                      f"h={env.data.qpos[2]:.3f}  "
                      f"vel=[{env.data.qvel[0]:.2f},{env.data.qvel[1]:.2f}]  "
                      f"contacts={contacts}")

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, reward: {total_reward:.2f}\n")
                total_reward = 0
                steps = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nDone.")

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl — Direct Joint Control")
    parser.add_argument("--mode", choices=["train", "eval", "test"], required=True)
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--target", nargs=2, default=["1.0", "0.0"],
                        help="crawl direction X Y")
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "test":
        test_random(args)

if __name__ == "__main__":
    main()
