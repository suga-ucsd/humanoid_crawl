# G1 Spider Crawl — Project Summary

---

## 1. Project Goal

The goal of this project is to make a Unitree G1 humanoid robot crawl like a spider — lying belly-up on the ground with all four limbs acting as legs. The system uses a layered control architecture where Inverse Kinematics (IK) handles the geometry of limb movement and a PD controller tracks the desired joint angles under full physics simulation.

<!-- [IMAGE: Final spider pose — robot belly-up with limbs extended] -->

---

## 2. System Architecture

The control pipeline follows a strict separation of responsibilities:

```
Target EE Positions → IK Solver → Joint Angles → PD Controller → Torques → MuJoCo Physics
```

Each layer has a single job:

- **IK Solver** — takes desired end-effector (hand/foot) positions in 3D space and computes the joint angles needed to reach them. It knows nothing about forces, gravity, or balance.
- **PD Controller** — takes the joint angles from IK and produces motor torques to track them. It fights gravity, friction, and contact forces to hold the robot in the desired configuration.
- **MuJoCo Physics** — simulates the actual dynamics: rigid body motion, contact, friction, gravity. The PD torques are applied as actuator commands.

This separation means the IK can be tested in pure kinematics mode (no physics) for debugging, and the PD controller can be tuned independently.

<!-- [IMAGE: Architecture diagram showing the pipeline] -->

---

## 3. The Robot — Unitree G1 (29-DOF)

The robot is defined in `g1_29dof_lock_waist_rev_1_0.xml`. It is a full humanoid with a free-floating base (6 DOF) and 21 actuated hinge joints. Several joints (ankle roll, wrist roll, wrist yaw) are locked/commented out in the XML, reducing the original DOF count.

### 3.1 Joint Layout

The actuated joints and their torque limits (from `actuatorfrcrange` in the XML):

| Group | Joints | Torque Limit |
|---|---|---|
| **Left Leg** | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch | 88, 139, 88, 139, 50 Nm |
| **Right Leg** | hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch | 88, 139, 88, 139, 50 Nm |
| **Waist** | waist_yaw | 88 Nm |
| **Left Arm** | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch | 25, 25, 25, 25, 5 Nm |
| **Right Arm** | shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_pitch | 25, 25, 25, 25, 5 Nm |

Key observation: the legs are significantly stronger than the arms (139 Nm vs 25 Nm). This directly affects PD gain tuning — uniform gains cause the legs to be too weak and the arms to jitter.

### 3.2 End-Effector Sites

Four sites are defined in the XML for IK targeting:

- `left_foot_site` — on the left ankle, pointing downward
- `right_foot_site` — on the right ankle, pointing downward
- `left_hand_site` — on the left wrist, pointing outward
- `right_hand_site` — on the right wrist, pointing outward

These sites are what the IK solver targets. Each site is associated with a kinematic chain of 5 joints.

<!-- [IMAGE: Robot with end-effector sites highlighted] -->

### 3.3 Kinematic Chains

Each end-effector is controlled by exactly 5 joints in its kinematic chain. This is critical for the per-limb IK solver — it only modifies joints in the relevant chain, preventing cross-talk between limbs.

```
Left Foot:  left_hip_pitch → left_hip_roll → left_hip_yaw → left_knee → left_ankle_pitch
Right Foot: right_hip_pitch → right_hip_roll → right_hip_yaw → right_knee → right_ankle_pitch
Left Hand:  left_shoulder_pitch → left_shoulder_roll → left_shoulder_yaw → left_elbow → left_wrist_pitch
Right Hand: right_shoulder_pitch → right_shoulder_roll → right_shoulder_yaw → right_elbow → right_wrist_pitch
```

---

## 4. The Spider Pose

The spider pose is the resting configuration where the robot lies belly-up with limbs bent and splayed outward, resembling an upturned spider. This pose is defined in `g1_pose.txt` and was designed using a separate joint manipulation GUI.

### 4.1 Pose Values

```
left_hip_pitch_joint    -0.46
left_hip_roll_joint      0.60
left_hip_yaw_joint      -0.16
left_knee_joint          1.47
left_ankle_pitch_joint   0.00
right_hip_pitch_joint   -0.60
right_hip_roll_joint    -0.52
right_hip_yaw_joint      0.16
right_knee_joint         1.77
right_ankle_pitch_joint  0.00
waist_yaw_joint          0.00
left_shoulder_pitch_joint  -2.91
left_shoulder_roll_joint    0.84
left_shoulder_yaw_joint     0.00
left_elbow_joint            0.03
left_wrist_pitch_joint      0.00
right_shoulder_pitch_joint -3.02
right_shoulder_roll_joint  -0.90
right_shoulder_yaw_joint    0.00
right_elbow_joint           0.00
right_wrist_pitch_joint     0.00
```

### 4.2 Pose Design Tool

The pose was created using a tkinter GUI (`g1_viewer.py`) with:

- Per-joint sliders (range ±5 rad)
- Scrollable interface for all 21+ joints
- Save/load to `g1_pose.txt`
- Pure kinematics rendering (no physics, `mj_forward` only)
- Sliders initialize to saved pose values on startup

<!-- [IMAGE: Tkinter pose editor alongside MuJoCo viewer] -->

### 4.3 Belly-Up Orientation

The robot is placed belly-up using a quaternion rotation. The key values:

```python
data.qpos[0:3] = [0, 0, 0.25]              # position: centered, 25cm above ground
data.qpos[3:7] = [0.707, 0, -0.707, 0]     # quaternion: -90° around Y axis
```

This quaternion rotates the robot so its spine is horizontal and its belly faces upward. The height of 0.25m places it just above the ground so limbs make contact without clipping.

<!-- [IMAGE: Side view showing belly-up orientation] -->

---

## 5. Physics Configuration

Getting stable physics was one of the hardest parts. The default MuJoCo settings cause jitter, explosions, or sliding. Every parameter below was tuned to solve a specific problem.

### 5.1 Solver Settings

```python
model.opt.timestep = 0.001      # 1ms — half the default 2ms
model.opt.iterations = 30       # solver iterations (default ~20)
model.opt.ls_iterations = 30    # line search iterations
model.opt.solver = 2            # Newton solver (most stable)
```

**Why 1ms timestep?** At the default 2ms, the PD controller can't react fast enough to contact forces. The robot's limbs hit the ground, bounce, and the controller overcorrects, creating a feedback loop that amplifies into jitter or NaN explosions. At 1ms, forces are caught before they grow.

**Why Newton solver?** MuJoCo offers three solvers (PGS=0, CG=1, Newton=2). Newton is the most accurate and stable for articulated contact, at the cost of being slower per iteration. For a 21-DOF robot with ground contact, this tradeoff is worth it.

### 5.2 Contact Softening

```python
for i in range(model.ngeom):
    model.geom_solref[i] = [0.02, 1.0]
    model.geom_solimp[i] = [0.9, 0.95, 0.001, 0.5, 2.0]
```

**`solref = [0.02, 1.0]`** — the contact spring settles over 20ms (first value) with a damping ratio of 1.0 (critically damped). Default MuJoCo contacts are very stiff, which causes high-frequency bouncing when limbs touch the ground.

**`solimp`** — controls the impedance of the contact constraint. The values `[0.9, 0.95, 0.001, 0.5, 2.0]` create soft but well-damped contacts that don't allow deep penetration.

### 5.3 Joint Damping

```python
for i in range(6, model.nv):    # skip free joint DOFs 0-5
    model.dof_damping[i] = 1.0
```

This adds passive viscous damping to every joint. Without it, joints oscillate freely when not under active PD control. The damping is applied only to hinge joint DOFs (indices 6+), not the free-floating base (indices 0-5), because damping the base would resist the robot's overall translation/rotation.

### 5.4 Ground Friction

```python
floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
model.geom_friction[floor_id] = [1.5, 0.005, 0.001]

for i in range(model.ngeom):
    if i != floor_id:
        model.geom_friction[i] = [1.5, 0.005, 0.001]
```

The three friction values are:

- **1.5** — sliding (Coulomb) friction. Higher means grippier. 1.5 gives enough traction for the limbs to push without slipping.
- **0.005** — torsional friction. Resists rotation of the contact patch.
- **0.001** — rolling friction. Very low since the robot doesn't have wheels.

Both the floor and all robot geoms get the same friction values. MuJoCo computes effective contact friction as the geometric mean of the two contacting geoms.

---

## 6. PD Controller

The PD (Proportional-Derivative) controller is the lowest layer. It takes desired joint angles from the IK solver and produces motor torques to track them.

### 6.1 Control Law

For each actuator:

```
torque = Kp × (q_target - q_current) - Kd × q_velocity
```

Where:
- `q_target` — desired joint angle (from IK)
- `q_current` — actual joint angle (from simulation)
- `q_velocity` — actual joint velocity (from simulation)
- `Kp` — proportional gain (stiffness)
- `Kd` — derivative gain (damping)

The torque is then clamped to the joint's maximum:

```python
data.ctrl[i] = np.clip(tau, -max_tau[i], max_tau[i])
```

### 6.2 Per-Joint Gain Tuning

This was a critical lesson: **uniform gains don't work.** The robot has joints ranging from 5 Nm (wrist) to 139 Nm (knee). Using the same Kp for all joints means either:

- Kp is high enough for the knees → wrists jitter violently (overshoot on tiny inertia)
- Kp is low enough for wrists → knees can't fight gravity (too weak)

The solution is to scale gains proportionally to each joint's torque limit:

```python
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    frc = TORQUE_LIMITS[name]     # from XML actuatorfrcrange
    max_tau[i] = frc
    kp[i] = frc * 0.5            # proportional to joint strength
    kd[i] = kp[i] * 0.5          # half of Kp for damping
```

This gives:

| Joint | Max Torque | Kp | Kd |
|---|---|---|---|
| Knee | 139 Nm | 69.5 | 34.75 |
| Hip pitch | 88 Nm | 44.0 | 22.0 |
| Ankle | 50 Nm | 25.0 | 12.5 |
| Shoulder | 25 Nm | 12.5 | 6.25 |
| Wrist | 5 Nm | 2.5 | 1.25 |

### 6.3 Why Not Read Gains from XML?

The XML defines `actuatorfrcrange` on the `<joint>` elements, not `forcerange` on the `<motor>` elements. MuJoCo's `model.actuator_forcerange` only populates when `forcerange` is set on the motor itself. Since it wasn't, all values read as 0. The solution was to hardcode the limits from the XML into a Python dictionary.

<!-- [IMAGE: Robot holding spider pose under gravity with PD control] -->

---

## 7. Inverse Kinematics

The IK solver computes joint angles that place end-effectors at desired 3D positions. It uses the Jacobian-based damped least squares method.

### 7.1 Algorithm

For each IK iteration:

1. Compute the position error: `err = target_pos - current_pos`
2. Compute the Jacobian `J` of the end-effector position with respect to joint angles
3. Solve for joint angle changes using damped least squares:
   ```
   Δq = J^T (J J^T + λI)^{-1} × err
   ```
4. Apply `Δq` to joint angles
5. Clamp to joint limits
6. Repeat until error < tolerance

### 7.2 Per-Limb Solving

A critical design choice: each end-effector's IK only modifies the 5 joints in its kinematic chain. This is implemented by extracting only the relevant columns from the full Jacobian:

```python
# Full Jacobian is 3 × nv (all DOFs)
mujoco.mj_jacSite(model, data, jacp, None, site_id)

# Extract only this limb's DOF columns
J = jacp[:, dof_ids]    # 3 × 5 for a 5-joint chain
```

This prevents moving the left foot from accidentally rotating the right shoulder. Without per-limb solving, all 21 joints would shift to satisfy one end-effector, disturbing the entire pose.

### 7.3 Damping Parameter (λ)

The damping parameter `λ = 0.01` in the least squares prevents singularity issues. Near joint limits or stretched-out configurations, the Jacobian becomes rank-deficient (some directions become unreachable). Without damping, `(J J^T)^{-1}` explodes, producing huge joint velocity commands. The damping term `λI` regularizes the inversion at the cost of slightly slower convergence.

### 7.4 Joint Limit Clamping

After each IK iteration, every joint is clamped to its range defined in the XML:

```python
for i, jname in enumerate(joint_names):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    qa = model.jnt_qposadr[jid]
    data.qpos[qa] += step_size * dq[i]
    lo, hi = model.jnt_range[jid]
    if lo < hi:
        data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)
```

This prevents the IK from commanding impossible configurations.

### 7.5 IK Parameters

| Parameter | Interactive (Viewer) | Training (RL) |
|---|---|---|
| Max iterations | 100 | 30 |
| Step size | 0.5 | 0.5 |
| Tolerance | 0.002m | 0.005m |
| Damping (λ) | 0.01 | 0.01 |

Interactive mode uses more iterations for precision. Training uses fewer for speed since the PD controller smooths out small IK errors anyway.

<!-- [IMAGE: IK visualization — end-effector moving with limb following] -->

---

## 8. Interactive IK Viewer

Two interactive viewers were built for testing and debugging.

### 8.1 Tkinter Slider Interface

The first version uses a tkinter panel alongside the MuJoCo passive viewer:

- 4 end-effector sections (color-coded)
- XYZ offset sliders per end-effector (±20cm range)
- Reset button to return to home pose
- Shared state via `EEController` class between GUI thread and sim loop
- Pure kinematics (`mj_forward`) — no physics, perfectly stable
- Base position locked every frame

<!-- [IMAGE: Tkinter control panel with sliders] -->

### 8.2 Custom GLFW Viewer with Mouse Picking

The second version replaces tkinter with direct mouse interaction in the 3D viewport:

**Controls:**
- **Right-click + drag** — grab and move the nearest end-effector
- **Left-click + drag** — orbit camera
- **Middle-click / Shift+left** — pan camera
- **Scroll** — zoom

**How clicking works:**

1. On right-click, MuJoCo's `mjv_select` raycasts from the mouse into the 3D scene and returns a hit point
2. The hit point is compared against all 4 end-effector site positions
3. If within 20cm of a site, that end-effector is selected
4. During drag, mouse pixel delta is converted to 3D world-space movement using the camera's right and up vectors:

```python
right, up = get_camera_vectors(cam)
scale = 0.002 * cam.distance
target_pos += right * dx * scale
target_pos -= up * dy * scale
```

5. IK solves per-limb for the grabbed end-effector every frame
6. A HUD overlay shows which end-effector is being dragged

<!-- [IMAGE: GLFW viewer with end-effector being dragged] -->

### 8.3 Camera Vector Extraction

The 3D mouse movement requires knowing the camera's right and up directions in world space. These are computed from the azimuth and elevation angles:

```python
def get_camera_vectors(cam):
    az = np.radians(cam.azimuth)
    el = np.radians(cam.elevation)
    fwd = np.array([cos(el)*cos(az), cos(el)*sin(az), sin(el)])
    right = cross(fwd, [0, 0, 1])     # world up
    right /= norm(right)
    up = cross(right, fwd)
    up /= norm(up)
    return right, up
```

This ensures dragging the mouse left always moves the end-effector left from the camera's perspective, regardless of camera angle.

---

## 9. Key Lessons & Debugging

### 9.1 The Robot Flies Away

**Cause:** PD gains too high + high initial drop height. A 1-radian position error with `Kp=200` produces 200 Nm of torque, which on a 0.074 kg ankle link produces enormous acceleration.

**Fix:** Clamp torques to XML limits, reduce gains, start closer to the ground.

### 9.2 Joints Don't Move

**Cause:** The `joint_map` was storing joint IDs instead of `qpos` addresses. Joint ID 3 might map to `qpos[10]` due to the 7-element free joint at the start. Writing to `qpos[3]` modifies the base quaternion, not the intended joint.

**Fix:** Always use `model.jnt_qposadr[joint_id]` to get the correct qpos index.

### 9.3 Knees Don't Bend, Ankles Jitter

**Cause:** Uniform PD gains. Knees need ~80+ Nm to hold against gravity but were clamped at 20 Nm. Ankles (0.074 kg) were getting the same Kp as knees, massively overshooting.

**Fix:** Per-joint gains scaled to torque limits.

### 9.4 NaN Explosions

**Cause:** Contacts at coarse timestep. When a limb hits the ground at 2ms timestep, the contact force is computed as a huge impulse concentrated in one step. This exceeds float64 range.

**Fix:** Reduce timestep to 1ms (or 0.5ms), increase solver iterations, use Newton solver.

### 9.5 Robot Slides Around

**Cause:** Running full physics (`mj_step`) when only IK testing is needed. Gravity + contacts + PD create a dynamic system that's inherently unstable for a belly-up configuration.

**Fix:** For IK testing, use `mj_forward` (kinematics only). Lock the base position. No PD needed.

### 9.6 `actuator_forcerange` Returns All Zeros

**Cause:** The XML puts `actuatorfrcrange` on `<joint>` elements, not `forcerange` on `<motor>` elements. MuJoCo only populates `model.actuator_forcerange` from the motor's own `forcerange` attribute.

**Fix:** Hardcode the limits in a Python dictionary, read from the XML manually.

<!-- [IMAGE: Before/after comparison — unstable vs stable pose] -->

---

## 10. RL Attempt #1 — IK-in-the-Loop (Failed)

The first RL approach followed the original architecture plan: RL outputs end-effector positions, IK solves joint angles, PD tracks them.

### 10.1 Design

The action space was 12D — XYZ offset for each of 4 end-effectors (±10cm from home position). The pipeline per step at 50Hz was: RL outputs 12D action → IK solves per-limb joint angles (30 iterations) → PD tracks those angles for 20 physics substeps.

### 10.2 Why It Failed

After 2M training steps, reward oscillated in the negatives with no improvement. Several fundamental problems combined to make this approach unworkable.

**IK failure corrupts the learning signal.** The IK solver frequently fails or produces weird joint configurations near singularities or joint limits. When the RL agent outputs a reasonable EE target but IK produces a bad solution, the agent receives negative reward for something that wasn't its fault. The agent cannot distinguish "I chose a bad target" from "IK failed on a good target." This makes the reward signal extremely noisy and learning impossible.

**Static home positions become meaningless.** The `ee_home` positions were recorded once at reset in world coordinates. As the robot moves under physics, those world-space targets drift relative to the body. After a few seconds, the targets point to positions the robot left behind.

**12D EE space is still too large for exploration.** Random XYZ offsets for 4 end-effectors produce random flailing, not coordinated gaits. The probability of randomly discovering a useful crawling pattern (specific phase relationships between limbs, correct stride timing) is near zero.

**2M steps is insufficient.** Complex locomotion tasks with high-dimensional action spaces typically require 10-50M steps even with good reward shaping.

### 10.3 Lesson

IK-in-the-loop adds a noisy, sometimes-failing layer between the RL agent and the physics. The agent cannot learn to control something it cannot reliably predict. Removing IK from the training loop and giving RL direct joint control eliminates this uncertainty.

<!-- [IMAGE: Training curves showing flat/oscillating reward] -->

---

## 11. RL Attempt #2 — Direct Joint Control (Failed)

Inspired by successful locomotion papers (ANYmal, Cassie), the second attempt removed IK entirely. RL directly outputs target joint angles. This is how most successful legged locomotion works.

### 11.1 Design

The action space was 21D — one offset per actuated joint, scaled to ±0.3 rad from the spider pose. A gait phase clock (`sin(phase)`, `cos(phase)`) was added to the observation to encourage periodic motion. The reward included forward velocity, alive bonus, orientation, contact cycling (rewarding diagonal pair alternation), pose regularization, energy penalty, and action smoothness.

### 11.2 Curriculum Learning Addition

After the initial version showed no learning at 5M steps, a curriculum was added with three phases. Phase 1 (0-3M steps) rewarded only stability — big alive bonus, belly-up orientation, height maintenance, pose holding. Phase 2 (3M-7M steps) gradually ramped up velocity reward while maintaining stability reward. Phase 3 (7M+ steps) used the full velocity reward.

Additional changes: smaller action scale (±0.15 rad), smaller initial policy noise (`log_std_init=-1.5`), very lenient termination (only NaN or extreme height), and all reward components designed to give positive baseline reward.

### 11.3 Why It Failed

Even with curriculum learning and 5M+ steps, the agent failed to discover crawling motion.

**The fundamental problem is the search space.** The agent must output 21 joint values every single timestep (50Hz) and somehow discover that they need to be periodic, coordinated across 4 limbs, and phase-offset in a specific pattern. This is like asking someone to independently control 21 knobs 50 times per second and produce a walking pattern. The search space is enormous — the chance of random exploration stumbling upon anything resembling coordinated locomotion is effectively zero.

**The agent learned to not die, but not to move.** The curriculum successfully taught the agent to hold the spider pose (Phase 1 reward was positive). But transitioning to forward motion (Phase 2) required discovering a coordinated gait from scratch, which random exploration never achieved. The agent found a local optimum: hold still and collect stability reward.

**The gait clock didn't help enough.** While the phase clock provided timing information, the agent still needed to learn which joints to oscillate, with what amplitude, and with what phase relationship to the clock. This is still a huge combinatorial search.

### 11.4 Lesson

Direct joint control works in the literature because those papers use either massive compute (billions of steps on GPU-accelerated simulators like Isaac Gym), sophisticated reward shaping with domain randomization, or reference motions. With SB3 on CPU, the sample efficiency is too low for a 21D action space to find coordinated periodic motion from scratch.

---

## 12. RL Attempt #3 — Central Pattern Generator (Failed)

To reduce the search space, a CPG was introduced. The RL agent controls only gait parameters, and the CPG converts them into coordinated periodic joint trajectories.

### 12.1 Design

The action space was reduced to 10D: 4 per-limb amplitudes (how much each limb swings), 4 per-limb phase offsets (fine-tune the trot timing), 1 global frequency (stepping speed), and 1 stride height (how high to lift limbs). The CPG generated sinusoidal joint trajectories: `pitch = amplitude × sin(phase)` for forward/back swing, `roll = amplitude × 0.3 × cos(phase)` for lateral lift, and `knee = amplitude × 0.5 × max(0, sin(phase))` for ground clearance.

Default phases encoded a trot gait (diagonal pairs in sync). The RL agent only needed to fine-tune amplitudes and timing — periodicity was guaranteed.

### 12.2 Manual Testing Revealed Fundamental CPG Problems

A `--mode manual` option with tkinter sliders was added to test CPG parameters without RL. This revealed that no combination of parameters produced forward motion — the robot slid in circles instead of crawling.

**Problem 1: Symmetric sine waves produce zero net force.** The CPG used pure sinusoids for pitch — the limb pushes forward and backward equally, resulting in no net displacement. For crawling, the stance phase (pushing against ground) and swing phase (repositioning in air) need to be asymmetric.

**Problem 2: Per-limb push directions were wrong.** The `get_push_directions` function computed outward directions from the pelvis for each limb. This meant front limbs pushed forward, back limbs pushed backward, left limbs pushed left, right limbs pushed right. These forces canceled out, producing no net motion (or circular sliding from slight asymmetries).

**Problem 3: Trot gait (2 limbs up, 2 down) was unstable.** A belly-up robot with only 2 limbs on the ground is inherently unstable. The robot needs 3 limbs on the ground at all times (wave gait).

### 12.3 Attempted Fixes

The CPG was redesigned with distinct stance/swing phases using piecewise waveforms instead of pure sinusoids. Per-limb sign conventions were added to flip push directions for arms vs legs. The manual slider interface allowed real-time parameter tuning.

Despite these fixes, the robot still slid rather than crawled. The core issue was that the CPG was designed heuristically without understanding the precise kinematics of the belly-up configuration. Getting the sign conventions, phase relationships, and joint coupling correct for all 4 limbs by hand proved extremely difficult.

### 12.4 Lesson

CPGs work well when the gait pattern is well-understood (e.g., quadruped walking has decades of research). For an unusual configuration like belly-up spider crawling, the correct CPG parameters are not intuitive. The manual tuning approach exposed the problem: if a human engineer can't find working parameters with live sliders, an RL agent won't either — the CPG structure itself was wrong, not just the parameters.

---

## 13. The Reference Motion Approach (Current)

After three failed RL attempts, the approach shifted to a fundamentally different strategy: first create a working crawling motion using IK (no physics), then train RL to reproduce it under physics. This is imitation learning, specifically reference motion tracking (similar to DeepMimic).

### 13.1 Why This Should Work

All three previous approaches failed because the RL agent had to **discover** the crawling motion from scratch. With imitation learning, the motion is **given** — the agent only needs to learn the corrections required to execute it under real physics (gravity compensation, contact force management, balance).

This is analogous to teaching someone to swim: instead of throwing them in the pool and rewarding forward motion (Options A/B/C), you show them the arm strokes and kick pattern on land first, then have them practice it in water (imitation learning). The second approach is how every swim instructor actually teaches.

The technical advantages are: dense reward signal at every timestep (distance to reference pose, not just "did you move forward"), small action space (corrections are ±0.1 rad, not ±0.3 rad), the observation includes both current and target state so the agent knows exactly what to correct, and even with zero RL (pure playback), the motion approximates crawling.

### 13.2 Reference Motion Generator (`g1_crawl_ref.py`)

The reference generator creates crawling trajectories using pure kinematics (no physics, no gravity, base locked).

**End-effector trajectories** trace D-shaped hook/digging paths. Each foot follows an elliptical trajectory: during swing, the foot lifts from the rear, arcs forward and up in a bell curve, and plants at the front. During stance, the foot stays on the ground and drags from front to rear, pushing the body forward.

The D-shape was critical — earlier versions used pure sinusoids where the foot went straight up and came straight back down to the same position, producing no forward travel. The fix was switching from `sin(phase)` to `-cos(phase)` for the horizontal component, which creates the necessary offset between the lift point and the plant point.

<!-- [IMAGE: D-shaped trajectory diagram — swing arc above, stance drag below] -->

**Wave gait (3+1 pattern)** ensures stability. Each limb is phase-offset by 90° (π/2): left foot at 0°, left hand at 90°, right foot at 180°, right hand at 270°. The swing duty cycle is 25% — each limb is airborne for only one quarter of the cycle. With 4 limbs staggered by 25%, exactly 1 limb is in the air and 3 are on the ground at any moment.

```
Time →
LF:  ██SWING██░░░░░░░░░░░░░░░░░░░░░░░░  (25% swing, quick scoop)
LH:  ░░░░░░░░██SWING██░░░░░░░░░░░░░░░░
RF:  ░░░░░░░░░░░░░░░░██SWING██░░░░░░░░
RH:  ░░░░░░░░░░░░░░░░░░░░░░░░██SWING██

Ground: 3 limbs   3 limbs   3 limbs   3 limbs  ← always stable
```

**Unified push direction** was another critical fix. All 4 limbs use the same movement direction vector. During stance, every foot drags opposite to the movement direction — this is what creates net forward force. Earlier versions computed per-limb push directions (outward from pelvis), which caused front and back limbs to cancel each other out, resulting in spinning instead of translation.

```
WRONG (per-limb outward):           CORRECT (unified direction):
  LF→↗   RF→↘                        LF→→   RF→→
      [body]         net = 0              [body]       net = →→→
  LH→↙   RH→↖                        LH→→   RH→→
```

**IK solves per-limb.** For each end-effector target position, the Jacobian-based IK solver adjusts only the 5 joints in that limb's kinematic chain. This prevents cross-limb interference.

**Multi-direction reference data.** The generator can save trajectories for 8 compass directions plus stationary, creating a library of references for the RL agent. The `--save-all` flag produces `g1_ref_forward.npz`, `g1_ref_backward.npz`, `g1_ref_left.npz`, etc. Each file contains timestamped joint angle trajectories, EE positions, gait phase, and movement direction.

### 13.3 Reference Data Format

Each `.npz` file contains:

| Field | Shape | Description |
|---|---|---|
| `times` | [N] | Timestamps at 50Hz |
| `joint_qpos` | [N, 21] | Actuated joint angles per frame |
| `ee_positions` | [N, 4, 3] | End-effector XYZ positions per frame |
| `phase` | [N] | Gait phase angle (0 to 2π) |
| `move_dir` | [2] | Movement direction vector |
| `speed` | scalar | Gait frequency in Hz |
| `stride` | scalar | Stride length in meters |
| `lift` | scalar | Foot lift height in meters |

The reference loops seamlessly — the last frame connects back to the first. During RL training, the `ReferenceLibrary` class loads all reference files and provides frame-interpolated joint targets for any gait phase.

<!-- [IMAGE: Reference motion visualization — robot with limb trajectories traced] -->

---

## 14. Imitation Learning Environment (`g1_rl_imitate.py`)

The imitation learning RL environment uses a **residual policy** approach (similar to DeepMimic). The reference motion provides the base trajectory, and the RL agent learns small corrections to make it work under real physics.

### 14.1 Residual Policy Architecture

At each control step (50Hz):

```
reference_angles = lookup(reference_trajectory, current_phase)
correction = RL_agent(observation) × 0.1 rad
target_angles = reference_angles + correction
torques = PD_controller(target_angles)
```

The agent does NOT learn the crawling motion — that comes from the reference. The agent learns the 10% that physics demands: gravity compensation, contact force management, balance recovery. The correction scale (±0.1 rad) is deliberately small to keep the motion close to the reference.

### 14.2 Settling and Blending

Early versions of the environment dropped the robot from z=0.25m and immediately started playing the reference. Under gravity, the robot fell while trying to crawl, producing chaos. Two mechanisms were added to fix this.

**Settling phase:** During `reset()`, the robot holds the spider pose under PD control for 50 control steps (1 second) with full physics. Gravity pulls it to the ground, the PD controller holds the pose, and velocities are zeroed afterward. The robot starts the episode already resting stably on the ground.

**Motion blending:** For the first 100 steps (2 seconds) of each episode, the joint targets are linearly blended from the settled pose to the full reference trajectory. This prevents the sudden onset of crawling motion from destabilizing the settled robot. The blend factor `b` goes from 0 to 1:

```
target = (1 - b) × settled_pose + b × reference_pose + RL_correction
```

The reference clock only advances once blending is complete, so the gait doesn't get ahead of the body.

### 14.3 Observation Space

The observation is designed so the agent knows both where it is and where it should be:

| Component | Dimensions | Purpose |
|---|---|---|
| Pelvis quaternion | 4 | Body orientation |
| Pelvis angular velocity | 3 | Rotational dynamics |
| Pelvis linear velocity | 3 | Translational dynamics |
| Joint positions | 21 | Current joint angles |
| Joint velocities | 21 | Current joint speeds |
| Reference joint positions | 21 | What the joints SHOULD be |
| Joint tracking error | 21 | Difference (ref - current) |
| EE positions (relative) | 12 | Where the feet/hands are |
| Reference EE positions | 12 | Where they SHOULD be |
| Gait phase clock | 2 | sin/cos of current phase |
| Target direction | 2 | Where to crawl |
| Base height | 1 | How high off ground |
| **Total** | **123** | |

The reference joint positions and tracking error are critical — they tell the agent exactly what needs correcting at each moment.

### 14.4 Reward Function

The reward combines tracking fidelity with locomotion objectives:

**Joint tracking (weight 3.0):** `exp(-5 × ||q - q_ref||² / n_joints)`. Exponential form bounded between 0 (terrible tracking) and 1 (perfect match). The scaling by `n_joints` normalizes across different robot sizes. A typical well-tracked pose scores 0.7-0.9.

**EE position tracking (weight 2.0):** `exp(-50 × Σ||ee - ee_ref||² / 4)`. Same exponential form but in Cartesian space. More robust than joint tracking because different joint configurations can produce the same EE position.

**Forward velocity (weight 2.0):** `dot(vel_xy, target_dir)`. Positive when moving in the target direction. This is the secondary objective — the agent should crawl forward while tracking the reference.

**Alive bonus (weight 0.5):** Constant +1.0 per step for existing. Ensures the baseline reward is positive so longer episodes are always better than early termination.

**Orientation (weight 1.0):** `max(0, -pelvis_rot[2,2])`. Rewards keeping the belly facing up. Ranges from 0 (belly down) to 1 (belly perfectly up).

**Energy penalty (weight 0.0005):** `Σ(ctrl²) / n_actuators`. Small penalty for large torques.

**Action smoothness (weight 0.02):** `Σ(action - prev_action)²`. Penalizes jerky corrections.

### 14.5 Reference Library

The `ReferenceLibrary` class loads all `g1_ref_*.npz` files at initialization. At each episode reset, it selects the reference whose movement direction is closest (by dot product) to the current target direction. Frame lookup uses linear interpolation between recorded timesteps, and the trajectory loops seamlessly.

### 14.6 Stronger PD Gains

The PD gains were increased for imitation learning: `Kp = 1.0 × torque_limit` (was `0.5×`), `Kd = 0.3 × Kp`. Tracking a moving reference under gravity requires stronger control than holding a static pose. The previous gains were too weak — the robot would sag away from the reference, and the tracking reward would be perpetually low, giving the RL agent no useful gradient.

### 14.7 Playback Mode

The `--mode playback` command runs the reference under full physics with zero RL corrections. This serves as a diagnostic: if playback produces recognizable crawling motion (even imperfect), the RL agent only needs to learn small corrections. If playback fails completely, the reference or physics configuration needs fixing before training.

---

## 15. Key Lessons & Debugging (RL Phase)

### 15.1 The Agent Learns to Die

**Problem:** In early RL versions, the reward was net-negative for surviving. The agent discovered that dying quickly (short episode) produced less total negative reward than living longer. It learned to flip itself over on purpose.

**Fix:** Always design rewards with a positive alive baseline. If the agent scores higher by dying, your reward is broken. The simplest test: a random agent should accumulate positive reward. If it doesn't, fix the reward before training.

### 15.2 IK-in-the-Loop Corrupts Learning

**Problem:** The IK solver sometimes fails, producing bad joint configurations. The RL agent receives negative reward for IK failures, not its own mistakes.

**Fix:** Remove IK from the training loop entirely. For training, RL should control joints directly (or through a reference + corrections). IK is useful for visualization and reference generation, not for online control during training.

### 15.3 The Random Agent Test

**Problem:** Training for millions of steps with no improvement, unable to diagnose why.

**Fix:** Always test with `--mode test` (random agent) before training. Check: (1) Are episodes surviving the full length? If not, termination is too aggressive. (2) Is the random reward positive? If not, the reward baseline is wrong. (3) What do the contacts look like? If always `[0,0,0,0]`, the limbs never touch ground.

### 15.4 The Manual CPG Test

**Problem:** The CPG-based RL wasn't learning, but it was unclear whether the CPG structure was wrong or just the parameters.

**Fix:** Adding a `--mode manual` with live tkinter sliders instantly revealed that no parameter combination produced forward motion — the CPG structure itself was broken. Always provide a manual testing mode for any controller you plan to train with RL.

### 15.5 Push Direction Cancellation

**Problem:** The robot slid in circles instead of moving forward, despite all limbs moving periodically.

**Fix:** All 4 limbs must push in the SAME direction (the movement direction). Per-limb outward directions cause front and back limbs to cancel each other's forces. This is counterintuitive — it feels like each limb should push "outward" — but for translation, all forces must be aligned.

### 15.6 Reference Generated Without Physics ≠ Playable Under Physics

**Problem:** The reference motion looked perfect in kinematics mode (no gravity, fixed base) but produced sliding chaos when played under full physics.

**Fix:** Three mechanisms: (1) Settling phase — let the robot stabilize under gravity before starting the reference. (2) Gradual blending — ramp from settled pose to full reference over 2 seconds. (3) Stronger PD gains — tracking a moving reference under gravity requires higher stiffness than holding a static pose.

### 15.7 Gym vs Gymnasium

**Problem:** `pip install gym` installs the deprecated, unmaintained package that doesn't support NumPy 2.0. Produces walls of deprecation warnings.

**Fix:** `pip uninstall gym && pip install gymnasium`. The maintained fork, required by Stable-Baselines3.

### 15.8 Tensorboard Not Installed

**Problem:** SB3 crashes with `ImportError: Trying to log data to tensorboard` when `tensorboard_log` is specified.

**Fix:** Either `pip install tensorboard` or remove the `tensorboard_log="logs/"` line from the PPO constructor.

---

## 16. File Structure (Updated)

```
project/
├── humanoids/
│   ├── g1_29dof_lock_waist_rev_1_0.xml    # Robot model (XML)
│   └── meshes/                             # STL mesh files
├── g1_pose.txt                             # Spider pose joint values
│
├── ── Pose Design ──
├── g1_viewer.py                            # Pose editor (tkinter sliders + MuJoCo viewer)
│
├── ── Interactive IK ──
├── g1_crawl.py                             # GLFW viewer with click-and-drag IK
│
├── ── Reference Motion ──
├── g1_crawl_ref.py                         # IK crawling reference generator
├── g1_ref_forward.npz                      # Reference: forward crawl
├── g1_ref_backward.npz                     # Reference: backward crawl
├── g1_ref_left.npz                         # Reference: left crawl
├── g1_ref_right.npz                        # Reference: right crawl
├── g1_ref_fwd_left.npz                     # Reference: diagonal
├── g1_ref_fwd_right.npz                    # Reference: diagonal
├── g1_ref_bwd_left.npz                     # Reference: diagonal
├── g1_ref_bwd_right.npz                    # Reference: diagonal
├── g1_ref_stationary.npz                   # Reference: idle pose
│
├── ── RL Training (Failed Attempts) ──
├── g1_rl.py                                # Attempt 1: IK-in-the-loop (failed)
├── g1_rl_direct.py                         # Attempt 2: direct joint control (failed)
├── g1_rl_cpg.py                            # Attempt 3: CPG + RL (failed)
│
├── ── RL Training (Current) ──
├── g1_rl_imitate.py                        # Imitation learning (reference tracking)
│
├── models/                                 # Saved RL checkpoints
│   ├── best/                               # Best model from eval callback
│   └── g1_imitate_*.zip                    # Periodic checkpoints
└── logs/                                   # Training logs
```

---

## 17. Dependencies

```bash
pip install mujoco gymnasium stable-baselines3 numpy
```

- **MuJoCo** (≥3.0) — physics simulation and rendering
- **GLFW** — bundled with MuJoCo, used for the custom interactive viewer
- **Gymnasium** — RL environment interface (NOT `gym`, which is deprecated)
- **Stable-Baselines3** — PPO implementation for training
- **NumPy** — array operations
- **Tensorboard** (optional) — `pip install tensorboard` for training visualization

---

## 18. Command Reference

### Pose Design
```bash
python g1_viewer.py                           # Edit spider pose with sliders
```

### Interactive IK
```bash
python g1_crawl.py                            # Click-and-drag end-effectors
```

### Reference Motion
```bash
python g1_crawl_ref.py                        # Visualize forward crawl
python g1_crawl_ref.py --dir 0 1              # Visualize leftward crawl
python g1_crawl_ref.py --stride 0.06 --lift 0.05 --speed 0.5
python g1_crawl_ref.py --save                 # Save forward reference
python g1_crawl_ref.py --save-all             # Save all 8 directions + idle
```

### Imitation Learning
```bash
python g1_rl_imitate.py --mode playback       # Test reference under physics
python g1_rl_imitate.py --mode test           # Random agent diagnostic
python g1_rl_imitate.py --mode train --steps 5000000 --n_envs 8
python g1_rl_imitate.py --mode eval --checkpoint models/best/best_model
python g1_rl_imitate.py --mode eval --checkpoint models/best/best_model --target 0 1
```

---

## 19. Architecture Evolution

The project went through four distinct architectural phases. Each failure informed the next design.

```
Attempt 1: RL → IK → PD → Physics
  Problem: IK failures corrupt RL signal
  Result:  No learning after 2M steps

Attempt 2: RL → PD → Physics  (direct joint control)
  Problem: 21D action space too large for random exploration
  Result:  Agent learns to hold still, never discovers gait

Attempt 3: RL → CPG → PD → Physics
  Problem: CPG structure wrong for belly-up config
  Result:  No parameter combo produces forward motion

Attempt 4: Reference + RL corrections → PD → Physics  (current)
  Approach: IK generates reference offline, RL learns small corrections online
  Status:  In progress
```

The key insight from this evolution: **separate motion design from motion execution.** Design the crawling motion offline using tools you can see and control (IK, sliders, visualization). Then train RL to execute that motion under physics constraints. Don't ask RL to simultaneously discover AND execute a complex motion.

<!-- [IMAGE: Architecture evolution diagram showing the four attempts] -->

---

## 20. What's Next

The imitation learning pipeline is set up. The remaining work is:

1. **Verify playback** — confirm the reference motion produces recognizable crawling under physics (`--mode playback`). If tracking scores (`j_track`) are above 0.5, the setup is correct.

2. **Train the residual policy** — run imitation learning with 84 parallel environments for 5-10M steps. The agent should learn gravity compensation and balance corrections quickly since the reference provides 90% of the solution.

3. **Evaluate directional control** — test with different `--target` directions to verify the agent selects and tracks the correct reference.

4. **Fine-tune for real crawling** — once tracking works, gradually increase the velocity reward weight to encourage the agent to push beyond just matching the reference, producing actual forward locomotion.

5. **Optional: Adversarial Motion Priors (AMP)** — if reference tracking works but the resulting motion looks unnatural, AMP uses a discriminator network to learn a style reward from the reference data, producing more natural-looking gaits.

<!-- [IMAGE: Vision of the final crawling behavior] -->
