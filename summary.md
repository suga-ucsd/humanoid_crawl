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

## 10. File Structure

```
project/
├── humanoids/
│   ├── g1_29dof_lock_waist_rev_1_0.xml    # Robot model
│   └── meshes/                             # STL mesh files
├── g1_pose.txt                             # Spider pose joint values
├── g1_viewer.py                            # Pose editor (tkinter + slider GUI)
├── g1_crawl.py                             # Interactive IK viewer (GLFW + mouse)
├── g1_rl.py                                # RL training pipeline
└── models/                                 # Saved RL checkpoints
```

---

## 11. Dependencies

```bash
pip install mujoco gymnasium stable-baselines3 numpy
```

- **MuJoCo** — physics simulation and rendering
- **GLFW** — bundled with MuJoCo, used for the custom interactive viewer
- **Gymnasium** — RL environment interface
- **Stable-Baselines3** — PPO implementation for training
- **NumPy** — array operations

---

## 12. Next Steps

The IK and PD layers are complete and tested. The next phase is RL training, where a policy network learns to output end-effector target positions that produce coordinated spider crawling locomotion. The RL agent's job is coordination, gait timing, and balance — the IK and PD handle the mechanical execution.

<!-- [IMAGE: Vision of the final crawling behavior] -->
