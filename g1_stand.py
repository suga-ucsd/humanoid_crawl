#!/usr/bin/env python3
"""
G1 Spider Stand — Lift body off ground, supported by 4 limbs.

Goal: pelvis elevated, 4 end-effectors on ground, torso NOT touching ground.
Like a spider doing a push-up.

Usage:
    python g1_stand.py --mode hold
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
    "left_shoulder_pitch_joint": 25, "left_shoulder_roll_joint": 25,
    "left_shoulder_yaw_joint": 25, "left_elbow_joint": 25,
    "left_wrist_pitch_joint": 5,
    "right_shoulder_pitch_joint": 25, "right_shoulder_roll_joint": 25,
    "right_shoulder_yaw_joint": 25, "right_elbow_joint": 25,
    "right_wrist_pitch_joint": 5,
}

TARGET_HEIGHT = 0.20  # pelvis 20cm off ground

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
        kp[i] = frc * 0.8
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

# ──────────────────────────────────────
# Hold mode — PD only
# ──────────────────────────────────────

def hold_pose(args):
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    configure_physics(model)

    joint_map = get_joint_map(model)
    home_qpos = load_pose(model, joint_map)
    kp, kd, max_tau = build_gains(model)

    # Start at spider pose, belly up, slightly elevated
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = [0.707, 0, -0.707, 0]
    data.qvel[:] = 0

    q_target = home_qpos.copy()
    start_xy = data.qpos[0:2].copy()

    print(f"Hold mode — PD holds spider pose, target height={TARGET_HEIGHT}m")
    print("Robot should settle into spider stance with body elevated\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        try:
            while viewer.is_running():
                t = time.time()

                q_target[0:7] = data.qpos[0:7]
                pd_control_vec(model, data, q_target, kp, kd, max_tau)
                mujoco.mj_step(model, data)
                viewer.sync()
                step += 1

                if step % 500 == 0:
                    h = data.qpos[2]
                    drift = np.linalg.norm(data.qpos[0:2] - start_xy)
                    contacts = get_ee_contacts(model, data)
                    n_ee = int(np.sum(contacts))
                    vel = np.sqrt(data.qvel[0]**2 + data.qvel[1]**2)

                    print(f"  t={step*0.002:5.1f}s  h={h:.3f}m  "
                          f"target={TARGET_HEIGHT:.2f}  "
                          f"drift={drift:.3f}m  vel={vel:.3f}  "
                          f"EE_on_ground={n_ee}/4  "
                          f"contacts={contacts.astype(int).tolist()}")

                time.sleep(max(0, 0.002 - (time.time() - t)))

        except KeyboardInterrupt:
            print("\nDone.")

# ──────────────────────────────────────
# RL Environment
# ──────────────────────────────────────

class G1StandEnv(gym.Env):
    """
    Goal: lift pelvis to TARGET_HEIGHT, supported by end-effectors.

    Reward:
        + height (closer to target = better)
        + EE contacts (more feet on ground = better)
        + belly up orientation
        + alive bonus
        - velocity (don't move)
        - energy
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}
    RESIDUAL_SCALE = 0.15
    CONTROL_DT = 0.02
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

        self.ee_site_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in EE_SITES
        ]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        # Obs: quat(4) + angvel(3) + linvel(3) + joint_off(nu) + joint_vel(nu)
        #      + ee_contacts(4) + height(1)
        obs_dim = 4 + 3 + 3 + self.model.nu * 2 + 4 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu)
        self.start_xy = np.zeros(2)
        self.viewer = None

    def _get_obs(self):
        d = self.data
        joint_off = d.qpos[self.act_qpos] - self.home_qpos[self.act_qpos]
        joint_vel = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            va = self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            joint_vel[i] = d.qvel[va]

        contacts = get_ee_contacts(self.model, self.data)

        return np.concatenate([
            d.qpos[3:7], d.qvel[3:6], d.qvel[0:3],
            joint_off, joint_vel,
            contacts, [d.qpos[2]]
        ]).astype(np.float32)

    def _get_reward(self, action):
        d = self.data

        # 1. HEIGHT — the main reward. Closer to target = better.
        h = d.qpos[2]
        height_rew = np.exp(-20.0 * (h - TARGET_HEIGHT) ** 2)
        # Also raw bonus for being higher (encourages pushing up)
        height_bonus = min(h / TARGET_HEIGHT, 1.5)

        # 2. EE CONTACTS — reward feet on ground (supporting body)
        contacts = get_ee_contacts(self.model, self.data)
        n_contacts = np.sum(contacts)
        contact_rew = n_contacts / 4.0  # 0 to 1

        # 3. BELLY UP
        # pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        # pelvis_rot = d.xmat[pelvis_id].reshape(3, 3)
        # belly_up = max(0.0, -pelvis_rot[2, 2])

        # 4. ALIVE
        alive = 1.0

        # 5. STAY STILL (small penalty)
        vel_xy = np.sum(np.square(d.qvel[0:2]))
        ang_vel = np.sum(np.square(d.qvel[3:6]))
        drift = np.sum(np.square(d.qpos[0:2] - self.start_xy))

        # 6. ENERGY (small)
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        # 7. SMOOTHNESS
        action_rate = np.sum(np.square(action - self.prev_action))

        reward = (
            3.0 * height_rew        # match target height
            + 4.0 * height_bonus    # encourage pushing up
            + 1.5 * contact_rew     # feet on ground
            # + 1.0 * belly_up        # stay oriented
            + 0.5 * alive           # exist
            - 0.2 * vel_xy          # don't slide
            - 0.1 * ang_vel         # don't spin
            - 0.5 * drift           # don't drift
            - 0.0005 * energy       # efficiency
            - 0.01 * action_rate    # smooth
        )

        return reward

    def _is_terminated(self):
        if np.any(np.isnan(self.data.qpos)):
            return True
        h = self.data.qpos[2]
        return h > 0.6 or h < -0.05

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # Start at spider pose, belly up
        self.data.qpos[:] = self.home_qpos
        self.data.qpos[0:3] = [0, 0, 0.25]
        self.data.qpos[3:7] = [0.707, 0, -0.707, 0]

        if self.np_random is not None:
            self.data.qpos[7:] += self.np_random.uniform(-0.02, 0.02,
                                                          size=self.data.qpos[7:].shape)

        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

        self.start_xy = self.data.qpos[0:2].copy()
        self.step_count = 0
        self.prev_action = np.zeros(self.model.nu)

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # Target = spider pose + RL correction
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
# Train
# ──────────────────────────────────────

def train(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    def make_env():
        def _init():
            return G1StandEnv()
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    eval_env = G1StandEnv()

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
                net_arch=dict(pi=[128, 128], vf=[128, 128]),
                log_std_init=-1.5,
            ),
        )

    callbacks = [
        CheckpointCallback(
            save_freq=max(100_000 // args.n_envs, 1),
            save_path="models/",
            name_prefix="g1_stand",
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

    print(f"\nSpider Stand Training")
    print(f"  Goal: lift pelvis to {TARGET_HEIGHT}m, supported by 4 limbs")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs} | Device: {args.device}\n")

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=True)
    model.save("models/g1_stand_final")
    env.close()

# ──────────────────────────────────────
# Eval
# ──────────────────────────────────────

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
                print(f"  step={steps:4d}  rew={total_reward:.1f}  "
                      f"h={h:.3f}/{TARGET_HEIGHT:.2f}  "
                      f"EE={int(np.sum(contacts))}/4  drift={drift:.3f}")

            if terminated or truncated:
                episodes += 1
                print(f"\nEp {episodes} — steps: {steps}, reward: {total_reward:.1f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# Test
# ──────────────────────────────────────

def test_random(args):
    env = G1StandEnv(render_mode="human")
    obs, _ = env.reset()

    print(f"Target height: {TARGET_HEIGHT}m")
    print("Random agent test\n")
    total_reward, steps = 0, 0

    try:
        while True:
            action = env.action_space.sample() * 0.1
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            env.render()

            if steps % 50 == 0:
                h = env.data.qpos[2]
                contacts = get_ee_contacts(env.model, env.data)
                drift = np.linalg.norm(env.data.qpos[0:2] - env.start_xy)
                per_step = total_reward / max(steps, 1)
                print(f"  step={steps:4d}  rew/step={per_step:+.2f}  "
                      f"h={h:.3f}/{TARGET_HEIGHT:.2f}  "
                      f"EE={int(np.sum(contacts))}/4  drift={drift:.3f}")

            if terminated or truncated:
                print(f"Episode done — steps: {steps}, total: {total_reward:.1f}, "
                      f"per_step: {total_reward/max(steps,1):+.2f}\n")
                total_reward, steps = 0, 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Stand")
    parser.add_argument("--mode", choices=["hold", "train", "eval", "test"], required=True)
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    {"hold": hold_pose, "train": train, "eval": evaluate, "test": test_random}[args.mode](args)

if __name__ == "__main__":
    main()
