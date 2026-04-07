#!/usr/bin/env python3
"""
G1 Spider Crawl — Forward Only

Action space: only legs + arms (no waist, no wrists).
Same physics/PD/settle as working stand code.

Usage:
    python g1_crawl.py --mode playback
    python g1_crawl.py --mode test
    python g1_crawl.py --mode train --steps 5000000 --n_envs 16
    python g1_crawl.py --mode train --steps 5000000 --n_envs 16 --from_stand models/g1_stand_final
    python g1_crawl.py --mode eval --checkpoint models/best/best_model
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

# RL controls ONLY these joints. Waist + wrists are locked at home pose.
ACTIVE_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_hip_yaw_joint", "left_knee_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
]

BELLY_UP_QUAT = np.array([0.707, 0.0, -0.707, 0.0])

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
    model.opt.iterations = 50
    model.opt.ls_iterations = 20
    model.opt.solver = 2
    for i in range(6, model.nv):
        model.dof_damping[i] = 3.0
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
        kp[i] = frc * 3.0
        kd[i] = kp[i] * 0.3
    return kp, kd, max_tau

def pd_control_all(model, data, q_target, kp, kd, max_tau):
    """PD on ALL actuators — used for settling."""
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]
    tau = kp * (q_target[qa] - data.qpos[qa]) + kd * (-data.qvel[va])
    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

def pd_control_active(model, data, q_target, kp, kd, max_tau, active_ids, home_qpos):
    """PD on active joints only. Locked joints track home pose."""
    jnt_ids = model.actuator_trnid[:, 0]
    qa = model.jnt_qposadr[jnt_ids]
    va = model.jnt_dofadr[jnt_ids]

    # All joints track home pose by default
    tau = kp * (home_qpos[qa] - data.qpos[qa]) + kd * (-data.qvel[va])

    # Active joints track q_target instead
    for ai in active_ids:
        jid = jnt_ids[ai]
        q = model.jnt_qposadr[jid]
        v = model.jnt_dofadr[jid]
        tau[ai] = kp[ai] * (q_target[q] - data.qpos[q]) + kd[ai] * (-data.qvel[v])

    data.ctrl[:] = np.clip(tau, -max_tau, max_tau)

def find_spawn_height(model, data, home_qpos):
    data.qpos[:] = home_qpos
    data.qpos[0:3] = [0, 0, 1.0]
    data.qpos[3:7] = BELLY_UP_QUAT
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    lowest = min(data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)][2]
                 for n in EE_SITES)
    return data.qpos[2] - lowest + 0.03

def gentle_gravity_settle(model, data, home_qpos, kp, kd, max_tau, duration=2.0):
    """Settle using ALL actuators (not just active)."""
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

def get_ee_contacts(model, data):
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

# ──────────────────────────────────────
# Reference — forward only
# ──────────────────────────────────────

class ForwardReference:
    def __init__(self, ref_dir="."):
        path = os.path.join(ref_dir, "g1_ref_forward.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run: python g1_crawl_ref.py --save")
        ref = np.load(path)
        self.joint_qpos = ref["joint_qpos"]
        self.ee_positions = ref["ee_positions"]
        self.times = ref["times"]
        self.speed = float(ref["speed"])
        dt = self.times[1] - self.times[0] if len(self.times) > 1 else 0.02
        self.duration = self.times[-1] + dt
        print(f"  Forward ref: {len(self.times)} frames, {self.duration:.1f}s, {self.speed:.1f}Hz")

    def get_frame(self, t):
        t = t % self.duration
        idx = max(0, min(int(np.searchsorted(self.times, t, side="right")) - 1,
                         len(self.times) - 2))
        t0, t1 = self.times[idx], self.times[idx + 1]
        a = 0.0 if (t1 - t0) < 1e-8 else (t - t0) / (t1 - t0)
        jq = (1 - a) * self.joint_qpos[idx] + a * self.joint_qpos[idx + 1]
        ee = (1 - a) * self.ee_positions[idx] + a * self.ee_positions[idx + 1]
        return jq, ee

# ──────────────────────────────────────
# Environment
# ──────────────────────────────────────

class G1CrawlEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    RESIDUAL_SCALE = 0.10
    N_SUBSTEPS = 10
    MAX_STEPS = 1500
    BLEND_STEPS = 150

    def __init__(self, render_mode=None, ref_dir="."):
        super().__init__()
        self.render_mode = render_mode

        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)
        configure_physics(self.model)

        self.joint_map = get_joint_map(self.model)
        self.home_qpos = load_pose(self.model, self.joint_map)
        self.kp, self.kd, self.max_tau = build_gains(self.model)

        # Build active joint indices
        self.active_ids = []     # indices into model.nu (actuator indices)
        self.active_qpos = []    # qpos addresses for active joints
        self.active_dof = []     # dof addresses for active joints
        self.active_names = []

        all_act_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                         for i in range(self.model.nu)]

        for i, name in enumerate(all_act_names):
            if name in ACTIVE_JOINTS:
                self.active_ids.append(i)
                jid = self.model.actuator_trnid[i, 0]
                self.active_qpos.append(self.model.jnt_qposadr[jid])
                self.active_dof.append(self.model.jnt_dofadr[jid])
                self.active_names.append(name)

        self.active_ids = np.array(self.active_ids)
        self.active_qpos = np.array(self.active_qpos)
        self.active_dof = np.array(self.active_dof)
        self.n_active = len(self.active_ids)

        # Also need all actuator qpos/dof for full obs
        self.all_act_qpos = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])
        self.all_act_dof = np.array([
            self.model.jnt_dofadr[self.model.actuator_trnid[i, 0]]
            for i in range(self.model.nu)
        ])

        print(f"  Active joints: {self.n_active}/{self.model.nu}")
        print(f"  Locked: {[n for n in all_act_names if n not in ACTIVE_JOINTS]}")

        self.spawn_h = find_spawn_height(self.model, self.data, self.home_qpos)
        self.ref = ForwardReference(ref_dir)

        # Action: only active joints
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.n_active,), dtype=np.float32)

        # Obs: quat(4) + angvel(3) + linvel(3)
        #    + active_joint_pos(n_active) + active_joint_vel(n_active)
        #    + active_ref_joint(n_active)
        #    + ee_contacts(4) + clock(2) + height(1)
        obs_dim = 4 + 3 + 3 + self.n_active * 3 + 4 + 2 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.n_active)
        self.settled_joints = np.zeros(self.n_active)  # only active joints
        self.settled_height = 0.1
        self.start_x = 0.0
        self.ref_jq_full = np.zeros(self.model.nu)  # full reference (all actuators)
        self.ref_jq_active = np.zeros(self.n_active) # reference for active joints only
        self.ref_ee = np.zeros((4, 3))
        self.viewer = None

    def _get_obs(self):
        d = self.data
        contacts = get_ee_contacts(self.model, self.data)
        phase = 2 * np.pi * self.ref.speed * self.sim_time
        return np.concatenate([
            d.qpos[3:7],                        # pelvis quaternion (4)
            d.qvel[3:6],                         # angular velocity (3)
            d.qvel[0:3],                         # linear velocity (3)
            d.qpos[self.active_qpos],            # active joint positions (n_active)
            d.qvel[self.active_dof],             # active joint velocities (n_active)
            self.ref_jq_active,                  # reference for active joints (n_active)
            contacts,                            # EE contacts (4)
            [np.sin(phase), np.cos(phase)],      # clock (2)
            [d.qpos[2]],                         # height (1)
        ]).astype(np.float32)

    def _update_ref(self):
        jq_full, ee = self.ref.get_frame(self.sim_time)
        self.ref_jq_full = jq_full
        self.ref_ee = ee
        # Extract reference for active joints only
        # ref_jq_full is indexed by actuator order, active_ids selects the active ones
        self.ref_jq_active = jq_full[self.active_ids] if len(jq_full) == self.model.nu else jq_full[:self.n_active]

    def _get_reward(self, action):
        d = self.data

        # ----------------------------
        # Forward motion (PRIMARY TASK)
        # ----------------------------
        fwd_vel = d.qvel[0]
        fwd_rew = np.clip(fwd_vel, 0.0, 0.6)   # no reward for going backward

        # ----------------------------
        # Joint tracking (imitation)
        # ----------------------------
        j_err = np.sum(np.square(self.ref_jq_active - d.qpos[self.active_qpos]))
        j_track = np.exp(-2.0 * j_err / self.n_active)   # softer than before

        # ----------------------------
        # Height stabilization
        # ----------------------------
        h = d.qpos[2]
        h_rew = np.exp(-20.0 * (h - self.settled_height) ** 2)

        # ----------------------------
        # Orientation (belly-up crawl)
        # ----------------------------
        pid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        R = d.xmat[pid].reshape(3, 3)
        belly_up = max(0.0, -R[2, 2])

        # ----------------------------
        # Contacts (encourage crawling support)
        # ----------------------------
        contacts = get_ee_contacts(self.model, self.data)
        contact_rew = np.sum(contacts) / 4.0   # smoother than min(...,3)

        # ----------------------------
        # Progress (IMPORTANT)
        # ----------------------------
        progress = d.qpos[0] - self.start_x

        # ----------------------------
        # Direction alignment (soft constraint)
        # ----------------------------
        forward_vec = R[:, 0]
        alignment = forward_vec[0]  # dot with world x

        # ----------------------------
        # SMALL penalties (keep them tiny)
        # ----------------------------
        lat_vel = abs(d.qvel[1])
        yaw_rate = abs(d.qvel[5])
        act_rate = np.sum(np.square(action - self.prev_action))
        energy = np.sum(np.square(d.ctrl)) / self.model.nu

        torso_bad = 1.0 if get_torso_ground(self.model, self.data) else 0.0
            
        if self.step_count < 100000:
            weight_forward = 6.0
            weight_track = 1.0
        else:
            weight_forward = 3.0
            weight_track = 3.0
        # ----------------------------
        # FINAL REWARD
        # ----------------------------
        reward = (
            5.0 * fwd_rew          # strong incentive to move forward
            + weight_forward * progress       # long-term movement
            + weight_track * j_track        # imitation
            + 1.0 * h_rew          # stability
            + 1.0 * belly_up       # correct orientation
            + 0.5 * contact_rew    # ground interaction
            + 1.0 * alignment      # keep facing forward
            + 0.2                  # survival bonus

            # penalties (REDUCED)
            - 1.0 * torso_bad
            - 0.3 * lat_vel
            - 0.3 * yaw_rate
            - 0.001 * energy
            - 0.01 * act_rate
        )

        return reward, {
            "fwd_vel": float(fwd_vel),
            "j_track": float(j_track),
            "height": float(h),
            "alignment": float(alignment),
            "yaw_rate": float(yaw_rate),
            "n_contacts": int(np.sum(contacts)),
            "progress": float(progress),
            "torso_ground": bool(torso_bad),
        }


    def _is_terminated(self):
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
                              self.kp, self.kd, self.max_tau, duration=2.0)

        # Record settled positions for active joints only
        self.settled_joints = self.data.qpos[self.active_qpos].copy()
        self.settled_height = self.data.qpos[2]
        self.start_x = self.data.qpos[0]
        self.step_count = 0
        self.sim_time = 0.0
        self.prev_action = np.zeros(self.n_active)
        self._update_ref()

        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._update_ref()

        blend = min(1.0, self.step_count / max(self.BLEND_STEPS, 1))

        # Build full q_target: locked joints at home, active joints get action
        q_target = self.home_qpos.copy()
        q_target[0:7] = self.data.qpos[0:7]

        # Active joints: blend settled → reference + RL residual
        for idx, ai in enumerate(self.active_ids):
            qa = self.active_qpos[idx]
            base = (1 - blend) * self.settled_joints[idx] + blend * self.ref_jq_active[idx]
            q_target[qa] = base + action[idx] * self.RESIDUAL_SCALE
            jid = self.model.actuator_trnid[ai, 0]
            lo, hi = self.model.jnt_range[jid]
            if lo < hi:
                q_target[qa] = np.clip(q_target[qa], lo, hi)

        # Locked joints stay at home_qpos (already set above)

        for _ in range(self.N_SUBSTEPS):
            pd_control_active(self.model, self.data, q_target,
                              self.kp, self.kd, self.max_tau,
                              self.active_ids, self.home_qpos)
            mujoco.mj_step(self.model, self.data)

        if blend >= 1.0:
            self.sim_time += 0.02
        self.step_count += 1

        reward, info = self._get_reward(action)
        info["blend"] = float(blend)
        term, reason = self._is_terminated()
        if term:
            info["term_reason"] = reason
        trunc = self.step_count >= self.MAX_STEPS

        self.prev_action = action.copy()
        return self._get_obs(), reward, term, trunc, info

    def render(self):
        if self.viewer is None and self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

# ──────────────────────────────────────
# Playback
# ──────────────────────────────────────

def playback(args):
    env = G1CrawlEnv(render_mode="human", ref_dir=args.ref_dir)
    obs, _ = env.reset()
    print(f"Playback — settled h={env.settled_height:.3f}m, blend={env.BLEND_STEPS} steps")
    print(f"  Active joints: {env.n_active}, action dim: {env.action_space.shape[0]}")
    print(f"  {'step':>5}  {'phase':>6}  {'j_trk':>5}  {'fwd_v':>6}  "
          f"{'h':>5}  {'EE':>4}  {'torso':>5}  {'prog':>6}  {'rew':>6}")
    print("  " + "-" * 65)
    total, steps = 0, 0
    try:
        while True:
            action = np.zeros(env.n_active, dtype=np.float32)  # zero corrections
            obs, rew, term, trunc, info = env.step(action)
            total += rew; steps += 1
            env.render(); time.sleep(0.02)
            if steps % 25 == 0:
                bl = info["blend"]
                ph = "BLEND" if bl < 1 else "CRAWL"
                tg = "⚠YES" if info["torso_ground"] else "  no"
                print(f"  {steps:5d}  {ph:>6}  {info['j_track']:5.3f}  "
                      f"{info['fwd_vel']:+6.3f}  {info['height']:5.3f}  "
                      f"{info['n_contacts']:2d}/4  {tg}  "
                      f"{info['progress']:+6.3f}  {rew:+6.2f}")
            if term or trunc:
                r = info.get("term_reason", "truncated")
                v = "✅ survived" if not term else f"❌ {r}"
                print(f"\n  [{v}] steps={steps} rew={total:.1f} "
                      f"rew/s={total/max(steps,1):+.2f} prog={info['progress']:+.3f}m\n")
                total, steps = 0, 0; obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# Test
# ──────────────────────────────────────

def test_random(args):
    env = G1CrawlEnv(render_mode="human", ref_dir=args.ref_dir)
    obs, _ = env.reset()
    print(f"Random test — settled h={env.settled_height:.3f}m")
    print(f"  Active joints: {env.n_active}, action dim: {env.action_space.shape[0]}")
    print(f"  PASS: rew/step > 0, survives > 500 steps\n")
    total, steps, ep = 0, 0, 0
    ep_lens, ep_rews = [], []
    try:
        while True:
            action = env.action_space.sample() * 0.05  # tiny corrections
            obs, rew, term, trunc, info = env.step(action)
            total += rew; steps += 1; env.render()
            if steps % 100 == 0:
                ps = total / max(steps, 1)
                tg = "⚠TRS" if info["torso_ground"] else ""
                print(f"  step={steps:4d}  rew/s={ps:+.2f}  j={info['j_track']:.2f}  "
                      f"fwd={info['fwd_vel']:+.3f}  h={info['height']:.3f}  "
                      f"EE={info['n_contacts']}/4  prog={info['progress']:+.3f} {tg}")
            if term or trunc:
                ep += 1; ep_lens.append(steps); ep_rews.append(total)
                r = info.get("term_reason", "truncated")
                v = "✅" if steps >= 500 and total/max(steps,1) > 0 else "❌"
                print(f"  Ep {ep} [{r}] steps={steps} rew/s={total/max(steps,1):+.2f} {v}")
                if ep >= 3:
                    al = np.mean(ep_lens)
                    ar = np.mean([r/max(l,1) for r, l in zip(ep_rews, ep_lens)])
                    verdict = "✅ READY" if ar > 0 and al > 300 else "❌ NOT READY"
                    print(f"\n  Summary: avg_len={al:.0f} avg_rew/s={ar:+.2f} → {verdict}\n")
                total, steps = 0, 0; obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

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
            return G1CrawlEnv(ref_dir=args.ref_dir)
        return _init

    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    eval_env = G1CrawlEnv(ref_dir=args.ref_dir)

    if args.checkpoint and os.path.exists(args.checkpoint + ".zip"):
        print(f"Resuming from {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env, device=args.device)

    elif args.from_stand and os.path.exists(args.from_stand + ".zip"):
        print(f"Transfer learning from stand: {args.from_stand}")
        stand_model = PPO.load(args.from_stand)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=1e-4, n_steps=2048, batch_size=256,
            n_epochs=5, gamma=0.99, gae_lambda=0.95,
            clip_range=0.15, ent_coef=0.01, vf_coef=0.5,
            max_grad_norm=0.5, verbose=1, device=args.device,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,
            ),
        )
        try:
            sp = stand_model.policy.state_dict()
            cp = model.policy.state_dict()
            n = 0
            for k in cp:
                if k in sp and cp[k].shape == sp[k].shape:
                    cp[k] = sp[k]; n += 1
            model.policy.load_state_dict(cp)
            print(f"  Transferred {n}/{len(cp)} tensors")
        except Exception as e:
            print(f"  Transfer failed ({e}), training from scratch")
    else:
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=2048, batch_size=256,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
            max_grad_norm=0.5, verbose=1, device=args.device,
            tensorboard_log="logs/",
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
                log_std_init=-1.0,
            ),
        )

    xfer = f" (from {args.from_stand})" if args.from_stand else ""
    print(f"\nForward Crawl Training{xfer}")
    print(f"  Steps: {args.steps:,} | Envs: {args.n_envs} | Device: {args.device}")
    print(f"  Action dim: {env.action_space.shape[0]} (active joints only)\n")

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

# ──────────────────────────────────────
# Evaluate
# ──────────────────────────────────────

def evaluate(args):
    from stable_baselines3 import PPO
    if not args.checkpoint:
        print("Error: --checkpoint required"); return
    model = PPO.load(args.checkpoint)
    env = G1CrawlEnv(render_mode="human", ref_dir=args.ref_dir)
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
                      f"j={info['j_track']:.2f}  h={info['height']:.3f}  "
                      f"EE={info['n_contacts']}/4  prog={info['progress']:+.3f}m")
            if term or trunc:
                episodes += 1
                r = info.get("term_reason", "truncated")
                print(f"\n  Ep {episodes} [{r}] steps={steps} rew={total:.1f} "
                      f"prog={info['progress']:+.3f}m\n")
                total, steps = 0, 0; obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

# ──────────────────────────────────────
# CLI
# ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="G1 Spider Crawl — Forward Only")
    parser.add_argument("--mode", choices=["train", "eval", "test", "playback"], required=True)
    parser.add_argument("--steps", type=int, default=5_000_000)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--from_stand", type=str, default=None)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ref_dir", type=str, default=".")
    args = parser.parse_args()
    {"train": train, "eval": evaluate, "test": test_random, "playback": playback}[args.mode](args)

if __name__ == "__main__":
    main()
