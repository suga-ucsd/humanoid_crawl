#!/usr/bin/env python3
"""
G1 Spider Crawl — RL + IK + PD Pipeline

Usage:
    python g1_rl.py --mode train --steps 2000000
    python g1_rl.py --mode train --steps 500000 --checkpoint models/g1_spider
    python g1_rl.py --mode eval --checkpoint models/g1_spider
    python g1_rl.py --mode eval --checkpoint models/g1_spider --target 1.0 0.5

Requires: pip install stable-baselines3 gymnasium
"""

import argparse, os, time
import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
from gymnasium import spaces

MODEL_PATH = "humanoids/g1_29dof_lock_waist_rev_1_0.xml"

EE_SITES = ["left_foot_site", "right_foot_site", "left_hand_site", "right_hand_site"]

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
# Shared core functions
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

def configure_physics(model):
    model.opt.timestep = 0.001
    model.opt.iterations = 30
    model.opt.ls_iterations = 30
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

def pd_control(model, data, q_target, kp, kd, max_tau):
    for i in range(model.nu):
        jnt_id = model.actuator_trnid[i][0]
        qa = model.jnt_qposadr[jnt_id]
        va = model.jnt_dofadr[jnt_id]
        tau = kp[i] * (q_target[qa] - data.qpos[qa]) + kd[i] * (-data.qvel[va])
        data.ctrl[i] = np.clip(tau, -max_tau[i], max_tau[i])

def ik_solve_limb(model, data, site_name, target_pos, limb_dofs,
                  max_iters=30, step_size=0.5, tol=0.005):
    """Fast IK for training — fewer iters than interactive version."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    dof_ids = limb_dofs[site_name]
    joint_names = LIMB_JOINTS[site_name]
    nv = model.nv
    jacp = np.zeros((3, nv))

    # Work on a copy so we don't disturb running sim
    qpos_save = data.qpos.copy()

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

    # Extract solved joint angles, restore original qpos
    solved_qpos = data.qpos.copy()
    data.qpos[:] = qpos_save
    mujoco.mj_forward(model, data)
    return solved_qpos

def get_ee_positions(model, data):
    pos = {}
    for name in EE_SITES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        pos[name] = data.site_xpos[sid].copy()
    return pos

# ──────────────────────────────────────
# Gymnasium Environment
# ──────────────────────────────────────

class G1SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    # RL runs at 50Hz, physics at 1000Hz
    CONTROL_DT = 0.02
    PHYSICS_DT = 0.001
    N_SUBSTEPS = 20
    MAX_STEPS = 1000          # 20s episodes
    EE_OFFSET_SCALE = 0.10    # actions scale to ±10cm

    def __init__(self, render_mode=None, target_dir=None):
        super().__init__()
        self.render_mode = render_mode
        self.target_dir = np.array(target_dir or [1.0, 0.0], dtype=np.float64)
        self.target_dir /= (np.linalg.norm(self.target_dir) + 1e-8)

        # Load model
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        self.joint_map = get_joint_map(self.model)
        self.limb_dofs = build_limb_dofs(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)
        self.kp, self.kd, self.max_tau = build_gains(self.model)

        # Spaces: 12D action (4 EEs × 3 XYZ offsets), ~66D obs
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self._obs_size(),), dtype=np.float32)

        # Internal
        self.step_count = 0
        self.prev_action = np.zeros(12)
        self.q_target = None
        self.ee_home = None
        self.viewer = None

    def _obs_size(self):
        # pelvis_quat(4) + pelvis_angvel(3) + pelvis_linvel(3)
        # + joint_pos(21) + joint_vel(21) + ee_relative(12) + target_dir(2)
        return 4 + 3 + 3 + self.model.nu + self.model.nu + 12 + 2

    def _get_obs(self):
        d = self.data
        pelvis_quat = d.qpos[3:7].copy()
        pelvis_angvel = d.qvel[3:6].copy()
        pelvis_linvel = d.qvel[0:3].copy()

        # Joint positions and velocities (actuated only)
        joint_pos = np.zeros(self.model.nu)
        joint_vel = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            jid = self.model.actuator_trnid[i][0]
            qa = self.model.jnt_qposadr[jid]
            va = self.model.jnt_dofadr[jid]
            joint_pos[i] = d.qpos[qa]
            joint_vel[i] = d.qvel[va]

        # EE positions relative to pelvis
        pelvis_pos = d.qpos[0:3]
        ee_rel = np.zeros(12)
        for i, name in enumerate(EE_SITES):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            ee_rel[i*3:(i+1)*3] = d.site_xpos[sid] - pelvis_pos

        obs = np.concatenate([
            pelvis_quat, pelvis_angvel, pelvis_linvel,
            joint_pos, joint_vel,
            ee_rel, self.target_dir
        ])
        return obs.astype(np.float32)

    def _get_reward(self, action):
        d = self.data

        # 1) Forward velocity toward target direction
        vel_xy = d.qvel[0:2]
        forward_vel = np.dot(vel_xy, self.target_dir)

        # 2) Alive bonus — belly facing up (z-axis of pelvis should point down in world)
        # Pelvis body z-axis in world frame
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        belly_up = -pelvis_rot[2, 2]  # want local-Z pointing down → world -Z
        alive = 1.0 if belly_up > 0.3 else 0.0

        # 3) Height maintenance — pelvis should stay at reasonable height
        height = d.qpos[2]
        height_rew = -2.0 * abs(height - 0.15)  # target ~15cm off ground

        # 4) Energy penalty
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        # 5) Action smoothness
        action_delta = np.sum(np.square(action - self.prev_action))

        # 6) Contact reward — reward having EEs touching ground
        n_contacts = 0
        for i in range(d.ncon):
            geom1 = d.contact[i].geom1
            geom2 = d.contact[i].geom2
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            if geom1 == floor_id or geom2 == floor_id:
                n_contacts += 1
        contact_rew = min(n_contacts, 4) / 4.0

        reward = (
            2.0 * forward_vel
            + 0.5 * alive
            + 0.3 * height_rew
            + 0.3 * contact_rew
            - 0.001 * energy
            - 0.01 * action_delta
        )
        return reward

    def _is_terminated(self):
        height = self.data.qpos[2]
        if height > 0.6 or height < 0.02:
            return True
        # Check for NaN
        if np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)):
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Spider pose, belly up with small random perturbation
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]

        # Small random noise on joints
        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02, size=self.data.qpos[7:].shape)

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        # Record EE home positions after forward kinematics
        self.ee_home = get_ee_positions(self.model, self.data)
        self.q_target = self.data.qpos.copy()
        self.step_count = 0
        self.prev_action = np.zeros(12)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Convert action to EE targets
        ee_targets = {}
        for i, name in enumerate(EE_SITES):
            offset = action[i*3:(i+1)*3] * self.EE_OFFSET_SCALE
            ee_targets[name] = self.ee_home[name] + offset

        # Solve IK for each limb (using a temp copy for IK)
        ik_data = mujoco.MjData(self.model)
        ik_data.qpos[:] = self.data.qpos.copy()
        for name, target in ee_targets.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            dof_ids = self.limb_dofs[name]
            joint_names = LIMB_JOINTS[name]
            nv = self.model.nv
            jacp = np.zeros((3, nv))

            for _ in range(30):
                mujoco.mj_forward(self.model, ik_data)
                err = target - ik_data.site_xpos[sid]
                if np.linalg.norm(err) < 0.005:
                    break
                jacp[:] = 0
                mujoco.mj_jacSite(self.model, ik_data, jacp, None, sid)
                J = jacp[:, dof_ids]
                JJT = J @ J.T + 0.01 * np.eye(3)
                dq = J.T @ np.linalg.solve(JJT, err)
                for k, jname in enumerate(joint_names):
                    jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                    qa = self.model.jnt_qposadr[jid]
                    ik_data.qpos[qa] += 0.5 * dq[k]
                    lo, hi = self.model.jnt_range[jid]
                    if lo < hi:
                        ik_data.qpos[qa] = np.clip(ik_data.qpos[qa], lo, hi)

        # Set PD target from IK solution
        self.q_target[7:] = ik_data.qpos[7:]

        # Run physics with PD control
        for _ in range(self.N_SUBSTEPS):
            pd_control(self.model, self.data, self.q_target, self.kp, self.kd, self.max_tau)
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
# Training
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    target = [float(x) for x in args.target]

    def make_env():
        def _init():
            return G1SpiderEnv(target_dir=target)
        return _init

    n_envs = args.n_envs
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log="logs/",
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        print(f"Resuming from {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env)
    else:
        model = PPO(**model_kwargs)

    save_cb = CheckpointCallback(
        save_freq=max(50000 // n_envs, 1),
        save_path="models/",
        name_prefix="g1_spider",
    )

    print(f"\nTraining for {args.steps} steps with {n_envs} envs...")
    print(f"Target direction: {target}\n")

    model.learn(total_timesteps=args.steps, callback=save_cb, progress_bar=True)
    model.save("models/g1_spider_final")
    print("\nSaved final model to models/g1_spider_final")
    env.close()

# ──────────────────────────────────────
# Evaluation
# ──────────────────────────────────────

def evaluate(args):
    from stable_baselines3 import PPO

    target = [float(x) for x in args.target]

    if not args.checkpoint:
        print("Error: --checkpoint required for eval mode")
        return

    print(f"Loading {args.checkpoint}...")
    model = PPO.load(args.checkpoint)

    env = G1SpiderEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()

    print(f"\nEvaluating — target direction: {target}")
    print("Press Ctrl+C to stop\n")

    total_reward = 0
    steps = 0

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            time.sleep(0.02)

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, reward: {total_reward:.2f}")
                total_reward = 0
                steps = 0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print(f"\nStopped — final steps: {steps}, reward: {total_reward:.2f}")

# ──────────────────────────────────────
# Random agent test (no SB3 needed)
# ──────────────────────────────────────

def test_random(args):
    target = [float(x) for x in args.target]
    env = G1SpiderEnv(render_mode="human", target_dir=target)
    obs, _ = env.reset()

    print("Running random agent to test environment...")
    print("Press Ctrl+C to stop\n")

    total_reward = 0
    steps = 0

    try:
        while True:
            action = env.action_space.sample() * 0.3  # small random actions
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if steps % 50 == 0:
                print(f"  step={steps}  reward={total_reward:.2f}  "
                      f"height={env.data.qpos[2]:.3f}  "
                      f"vel=[{env.data.qvel[0]:.3f}, {env.data.qvel[1]:.3f}]")

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, total reward: {total_reward:.2f}")
                total_reward = 0
                steps = 0
                obs, _ = env.reset()

    except KeyboardInterrupt:
        print(f"\nStopped.")

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl RL")
    parser.add_argument("--mode", choices=["train", "eval", "test"], required=True,
                        help="train: RL training | eval: run trained model | test: random agent")
    parser.add_argument("--steps", type=int, default=2_000_000, help="training steps")
    parser.add_argument("--checkpoint", type=str, default=None, help="model path (no .zip)")
    parser.add_argument("--target", nargs=2, default=["1.0", "0.0"],
                        help="crawl direction as X Y (default: 1 0 = forward)")
    parser.add_argument("--n_envs", type=int, default=4, help="parallel envs for training")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        evaluate(args)
    elif args.mode == "test":
        test_random(args)

if __name__ == "__main__":
    main()
