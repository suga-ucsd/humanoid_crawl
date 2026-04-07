# G1 Spider Crawl — Running Development Log

> This is a raw development log, not a polished report. It captures every decision, failure, fix, and observation in chronological order. For the structured analysis, see `main.pdf`.

---

## Day 1 (Apr 1): Pose Viewer and First Contact with the Robot

### What was attempted
- Built an interactive GUI (tkinter) with sliders for all 22 joints to visualize the spider pose.
- Loaded the G1 model from `g1_29dof_lock_waist_rev_1_0.xml` and tried to display the spider pose from `g1_pose.txt`.

### What went wrong
- **GUI update loop conflict:** The slider update function ran every 50ms and overwrote loaded pose values with slider defaults (0). Pose loading was immediately undone.
- **Physics simulation ran simultaneously:** `mj_step` evolved the configuration even when trying to display a static pose. With gravity on, the robot collapsed within milliseconds.
- **Threading issues:** GUI needed main thread, simulation needed its own thread. Race conditions between updates.

### What was learned
- Can't run physics and GUI pose editing simultaneously without careful synchronization.
- For static pose visualization, disable gravity and collisions.

### Resolution
- Stripped the GUI. Built a minimal viewer that loads the pose once and displays it in MuJoCo's built-in viewer. ~60 lines, no threading, no conflicts. Worked immediately.

---

## Day 1 (continued): The Spider Pose

### Pose design
- Used the minimal viewer + slider GUI (separate script, `g1_viewer.py`) to design the spider pose.
- Robot belly-up, quaternion `[0.707, 0, -0.707, 0]` = -90° around Y axis.
- Legs: hip_pitch = -0.46, hip_roll = ±0.6, knee = 1.5
- Arms: shoulder_pitch = -2.91/-3.02, shoulder_roll = ±1.0, elbow = 0.72
- Waist, yaw joints, wrists: 0.0
- Saved to `g1_pose.txt` (22 joint values).

### Observation
- In zero gravity, the pose looks like a spider. All 4 EEs reach the ground plane.
- With gravity, the robot instantly collapses. PD control needed.

---

## Day 2 (Apr 2): IK + PD Control — First Active Control Attempt

### Approach
- Built a per-limb IK solver (damped least-squares Jacobian) that takes EE target positions and computes joint angles.
- PD controller tracks the IK output. Gains: Kp=100, Kd=10 (uniform).

### Failure 1: Robot flies away
- Kp=100 was way too aggressive. Initial pose error (default config → spider pose) = large. PD applied massive torques. Robot launched.
- Also: floating base (qpos[0:7]) has no actuator. Code was trying to control it through the actuator mapping → uncontrolled forces on the base.
- **Fix:** Reduced gains, fixed actuator mapping to use `model.actuator_trnid` correctly. Skip base DOFs.

### Failure 2: Knees stuck, ankles jitter
- After fixing gains: elbows worked perfectly, knees were rigid, ankles buzzed.
- Root cause: uniform gains across joints with very different dynamics. Knees have 139 Nm torque limit + heavy lower leg. Ankles have 50 Nm + tiny 0.074 kg link.
- Also: joint mapping used joint indices instead of `jnt_qposadr` — off-by-7 due to free joint taking qpos[0:6].
- **Fix:** Per-joint gains derived from `actuatorfrcrange`. Corrected qposadr indexing.

### Observation
- Even with correct gains, the robot could hold the pose for ~2 seconds before slowly sagging. PD alone can resist gravity briefly but not indefinitely at these gain levels.
- The IK solver sometimes failed near joint limits (singular Jacobian). Bad output = bad PD tracking = instability.

---

## Day 2 (continued): Interactive IK Viewer

### Built `g1_crawl.py` (the GLFW viewer)
- Mouse-click on an end-effector in the MuJoCo viewer → select it.
- Drag → IK solver computes new joint angles → PD tracks → limb follows mouse in real-time.
- Used `mjv_select` for picking (had to handle `flexid` attribute that was missing in older Python bindings).
- This tool was incredibly useful for understanding the robot's workspace and joint limits.

### Key observation from interactive IK
- The arms have very limited reach in the belly-up configuration. Shoulder pitch needs to be near -3.0 (limit is -3.09) just to get hands on the floor.
- The original arm torque limits (25 Nm) are far too weak to support body weight through the arm linkage.

---

## Day 2–3 (Apr 2–3): RL Attempt #1 — IK in the Loop

### Architecture
```
RL (12D EE deltas) → IK Solver → Joint angles → PD → MuJoCo Physics
```

### Training setup
- Gymnasium env, PPO via Stable-Baselines3.
- Observation: pelvis quat + velocities + joint positions + EE positions.
- Action: 12D continuous (XYZ offset per EE).
- Reward: forward velocity + alive + orientation.
- 2M timesteps, 8 parallel envs.

### Result: Complete failure
- The agent learned to output zero actions. Any non-zero action risked an IK failure (singular Jacobian, joint limit violation), which produced violent motion and negative reward.
- The learning signal was corrupted: the agent couldn't distinguish "my action was bad" from "IK failed on a valid action."
- After 2M steps: zero forward velocity, zero exploration, agent holds still.

### Lesson
- **Never put a fallible solver in the RL control loop.** IK is for offline trajectory generation.
- The action space (12D Cartesian) is too abstract. The agent has no direct relationship between its outputs and joint-level control.

---

## Day 3 (Apr 3): RL Attempt #2 — Direct Joint Control

### Architecture
```
RL (21D joint targets) → PD → MuJoCo Physics
```

### Reasoning
- Remove IK entirely. RL directly outputs target angles for all 21 actuated joints. Simpler pipeline.

### Training: 5M+ steps, 16 envs

### Result: Agent learns to hold still
- 21D continuous action space is enormous. To produce a coordinated gait, PPO must simultaneously discover: which joints, how much, what frequency, what phase offsets.
- The agent converged on holding perfectly still — the action that minimized negative reward.
- Worse: early versions had net-negative reward for surviving. The agent learned to **die quickly** because shorter episodes = less total negative reward.
- **Fix for dying:** Added positive alive baseline. Now surviving is always better than dying.
- Even with the fix: no locomotion discovered. 5M steps wasn't enough for 21D gait discovery.

### Lesson
- Action space must match the problem's true dimensionality. A gait has ~4-8 meaningful parameters. Embedding those in 21D is searching for a needle in a haystack.

---

## Day 3 (continued): RL Attempt #3 — CPG

### Architecture
```
RL (CPG parameters) → CPG (sinusoidal oscillations) → PD → Physics
```

### Reasoning
- CPGs produce rhythmic motion inherently. RL only needs to modulate frequency, amplitude, coupling.

### Result: Robot spins in circles
- The sign conventions for joint axes were wrong for belly-up orientation. When the robot is inverted, "forward" and "backward" for each joint axis flip.
- We guessed the signs. The guesses were wrong. Some limbs pushed forward, others backward. Net force: rotational.
- Manual sign flipping: tried ~10 combinations out of 256 possible. None worked.

### Lesson
- With 256+ possible sign combinations, **automate the search**. Don't guess. This lesson was applied later in `--mode search`.

---

## Day 3–4 (Apr 3–4): Reference Motion Generation

### Built `g1_crawl_ref.py`
- Uses IK to generate crawling trajectories offline (no physics).
- Wave gait: each limb traces a D-shaped path (lift → swing forward → place → push back).
- Phase offsets: 0°, 90°, 180°, 270° (standard spider wave gait).
- Multiple directions: forward, backward, left, right, diagonal.
- Saves as `.npz` files: `g1_ref_forward.npz` etc.
- Each file contains: joint_qpos[N, nu], ee_positions[N, 4, 3], times[N], phase[N], move_dir[2], speed.

### Observation
- The reference looks great in zero-gravity playback. Smooth coordinated crawling.
- Under full physics (gravity + contacts), the reference is NOT physically consistent. Joint angles that produce smooth motion in zero gravity don't account for gravity loading.

---

## Day 4 (Apr 4): RL Attempt #4 — Imitation Learning

### Architecture
```
Reference .npz → RL adds residual corrections (±0.1 rad) → PD → Physics
```

### The plan
- Reference provides 90% of the motion (the gait pattern).
- RL learns the 10% correction (balance, gravity compensation).
- Blend from settled pose → reference over 100 steps (2s ramp).

### Training: 5M steps, reward = joint tracking + EE tracking + forward velocity

### Result: Positive reward, zero crawling
- The reward was always positive and increasing.
- The robot oscillated its joints (high tracking score), kept all 4 feet planted, and slid forward using body momentum.
- The reward function could be satisfied without walking: gait tracking was high (oscillating joints), velocity was small but positive (sliding).
- **The robot learned to dance in place while sliding.**

### Lesson
- Reference motions must be physically consistent with execution physics. Zero-gravity references ≠ full-gravity execution.
- Reward functions that can be satisfied without the intended behavior **will** be exploited.

---

## Day 4 (continued): THE FRICTION DISCOVERY

### The diagnostic
- Wrote `diag_friction.py` to inspect every geom's friction values.
- **Every mesh geom had default friction: [1.0, 0.005, 0.0001].**
- MuJoCo computes contact friction as geometric mean of the two surfaces.
- Even with a high-friction floor: √(0.5 × 0.005) = 0.05 torsional friction.
- **The robot was on ice for the entire project so far.**

### The fix
```python
for i in range(model.ngeom):
    if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
        model.geom_friction[i] = [3.0, 0.5, 0.5]
```

### Impact
- Drift dropped from METERS to 2.4 MILLIMETERS over 3 seconds.
- **This single fix invalidated every previous experiment.** IK, CPG, imitation learning — all were fighting a broken foundation.
- 3 days of work were built on ice. A 10-minute diagnostic would have saved them.

### Also discovered
- Ankle joints vibrate (tiny 0.074 kg links). **Removed ankle joints from XML entirely.**
- Added contact box geoms at all 4 EE sites with high friction and condim="6".
- Increased arm torque limits: 25 → 60 Nm (shoulders/elbows), 5 → 10 Nm (wrists).

---

## Day 4 (continued): PyBullet Attempt

### Tried porting to PyBullet
- Wrote `g1_stand_bullet.py` using `loadMJCF`.
- PyBullet's MJCF loader failed: can't handle mesh geoms, can't find STL files (wrong path separator), "expected 'type' attribute for joint" errors, "multiple root links" warning.
- Only loaded 2 bodies (floor + partial robot). Robot had no joints.

### Decision
- **Abandoned PyBullet.** The MJCF loader is too limited for this XML. Converting to URDF would take days and distract from the actual problem.
- Committed to MuJoCo for the rest of the project.

---

## Day 4–5 (Apr 4–5): Standing — The Foundation

### Philosophy
- Refused to attempt crawling until standing was solved. If PD can't hold a static pose, it can't hold a dynamic gait.

### Problem 1: PD launches the robot
- Setting spider pose as PD target from frame 1 → massive torque (pose error × high Kp) → lateral reaction forces on single contact point → robot slides or launches.
- **Fix: `gentle_gravity_settle()`** — start with zero gravity and PD on. Ramp gravity 0→100% over 2–3 seconds. PD adapts incrementally.

### Problem 2: Wrong spawn height
- Guessing height (0.25m) either dropped too far or penetrated ground.
- **Fix: `find_spawn_height()`** — compute exact height where lowest EE is at z=3cm. Analytical, no guessing.

### Problem 3: Wrong PD targets
- Zero-gravity pose ≠ gravity-equilibrium pose. Tracking file values under gravity → constant error → lateral forces → sliding.
- **Fix:** After gravity settle, record actual joint positions as PD targets. Target IS equilibrium → error ≈ 0 → torques ≈ 0 → no sliding.

### Problem 4: Wrong quaternion
- Used `[0.707, 0.707, 0, 0]` (rotation around X) instead of `[0.707, 0, -0.707, 0]` (rotation around Y). Robot was on its side, not belly-up. Spawn height calculation produced garbage. Robot "vanished."
- **Fix:** Corrected to `[0.707, 0, -0.707, 0]`.

### Problem 5: Arms don't reach the ground
- With shoulder_pitch = -1.60, elbow = 1.20, hands were 10+ cm above ground.
- Iterated: shoulder_pitch → -2.30 → -2.80 (near limit -3.09), shoulder_roll ±0.40 → ±0.80, elbow 1.20 → 0.80 → 0.30.
- Final arm pose: shoulder_pitch = -2.80, shoulder_roll = ±0.80, elbow = 0.30.

### Standing RL
- Observation: quat + velocities + joint offsets + EE contacts + height.
- Action: residual corrections ±0.1 rad.
- Reward: height + contacts + belly_up + pose tracking - drift - energy.
- **Converged in ~2M steps.** Robot holds spider pose stably.

---

## Day 5 (Apr 5): Crawl Attempt #5 — Imitation Learning (with fixed physics)

### Setup
- Same imitation learning approach as Attempt #4, but now with:
  - Fixed friction (not ice anymore)
  - Gentle gravity settle
  - Correct spawn height
  - Correct quaternion
  - Removed ankles

### Reduced action space
- Went from 19 joints to 16 (removed waist + wrists from active set).
- Reasoning: waist and wrists contribute nothing to crawling.

### Result
- Reward positive, episodes survive full length.
- Robot follows the gait pattern visually.
- But: all 4 feet stay planted. Robot dances in place. No actual displacement.
- The imitation reward (joint tracking) is satisfied by oscillating, regardless of displacement.

### Decision
- **Abandoned imitation learning.** The reference/physics mismatch persists even with fixed friction. The robot satisfies the reward without walking.

---

## Day 5 (continued): Crawl Attempt #6 — 8-DOF Sine Gait + RL

### Key simplification
- Reduced to 8 active joints: 2 per limb (pitch + bend).
  - Legs: hip_pitch + knee
  - Arms: shoulder_pitch + elbow
- All other joints locked at home pose.
- No reference files. Gait generated live by sine wave function.

### `generate_sine_targets()`
- Each limb gets a sinusoidal offset from home pose.
- Phase offsets: 0°, 90°, 180°, 270° (wave gait).
- Swing = 25%, stance = 75%.
- Parameters: frequency, pitch_amp, bend_amp, per-limb signs.

### The sign problem (again)
- Default signs [-1, -1, 1, 1] didn't produce forward motion.
- Built `--mode search`: brute-force 324 combinations (3^4 pitch signs × 4 bend patterns). Each runs 5 seconds headlessly. Ranked by forward progress. Takes ~5 minutes.
- Found working signs automatically.

### Critical discovery: Stand gains kill crawling
- Stand code used Kp = 3×, dof_damping = 3.0. These are tuned for zero motion.
- The sine wave targets were being set correctly, but the overdamped PD snapped joints back instantly. Robot appeared frozen.
- **Fix: Two-gain architecture.** Stiff gains for settle phase, soft gains for crawl phase.
  - Settle: Kp = 3×, Kd = 0.3×Kp, damping = 3.0
  - Crawl: Kp = 1×, Kd = 0.1×Kp, damping = 0.5

### With soft gains + correct signs
- Limbs actually move! Visible oscillation.
- But forward velocity ≈ 0. The sine wave is too smooth — slow acceleration at extremes.

### Built `--mode tune`
- Tkinter GUI with sliders for frequency, amplitudes, per-limb signs.
- Runs alongside MuJoCo viewer. Adjust in real-time, see effect immediately.

---

## Day 5–6 (Apr 5–6): Reward Engineering Marathon

### Reward v1: Forward velocity + cumulative progress
```
r = 5 * fwd_vel + 3 * (x - x_start) + stability
```
**Exploit:** Cumulative progress grows forever. Robot moved 5cm once, then held still, collecting +0.15/step forever. Reward at step 1500: >2000. Actual crawling: zero.

**Lesson:** Never use cumulative quantities in per-step rewards.

### Reward v2: Per-step displacement
```
r = 200 * max(0, x_t - x_{t-1}) + stability
```
**Exploit:** Robot jerked back-and-forth, all 4 feet planted. Micro-positive displacements from friction asymmetry. Sliding, not walking.

### Reward v3: Require limb lifting
Added: lifting bonus (+2 if any EE airborne), all-feet-down penalty (-3), jerk penalty.

**Result:** Legs started crawling! Visible swing/stance cycling. First real locomotion-like motion.

**Exploit:** Only legs worked. Arms stayed planted. Reward said "lift A limb" not "lift ALL limbs."

### Reward v4: Fairness + lazy limb penalty
Added:
- Per-limb lift tracking: `limb_lift_count[j]` increments on ground→air transitions.
- Fairness: `n_active_limbs / 4`. Using only legs = 0.5 (+1.0). All 4 = 1.0 (+2.0).
- Lazy penalty: after step 200, -0.5 per limb that never lifted.
- Cycle bonus: +0.5 when a different limb lifts than previous.

**Result:** Arms beginning to participate. Lift counts increasing for all 4 limbs.

**Exploit:** Robot discovered jumping. Launch off ground → land forward → massive displacement reward. No crawling, pure hopping. The 200× displacement weight was too dominant with no constraint on HOW displacement was achieved.

### Reward v5: Anti-jumping
Added:
- Vertical velocity penalty: -5.0 * |v_z|
- Airborne penalty: -3.0 if n_contacts ≤ 1 (fully airborne)
- Height variation penalty: -2.0 * (h - h_settled)²
- Reduced displacement weight: 200 → 50

**Result:** Jumping eliminated. Robot stays grounded.

**Exploit:** Asymmetric coordination. Left hand pushed forward, three other limbs pushed backward. Net displacement was positive (barely). Agent satisfied "all limbs lift" and "forward displacement" simultaneously — but 3 limbs were fighting the direction of travel.

### Reward v6: Per-limb directional consistency
Added: compute per-EE velocity in world X using `mj_jacSite`. Penalize any EE moving backward during stance.

```python
for each EE:
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, sid)
    ee_vel_x = jacp[0] @ data.qvel
    if ee_vel_x < 0:
        backward_penalty += abs(ee_vel_x)
```

**Result:** All four limbs now push in the same direction. Symmetry-breaking resolved.

---

## Day 6 (Apr 6): Triangle Wave Replacement

### Why sine waves weren't enough
- Sine waves have smooth acceleration everywhere. At the swing-phase extremes (where the limb reverses direction), acceleration is maximum but velocity is zero. At the zero-crossing (where the limb should push hardest), the velocity is maximum but the acceleration is zero.
- For crawling, we need: fast swing (snappy repositioning), slow stance (sustained pushing). Sine gives equal time to both.

### Triangle/piecewise gait
```python
swing_ratio = 0.2  # 20% of cycle in swing (fast), 80% in stance (slow)
if t < swing_ratio:
    curve = 1.0 - abs(2*s - 1)  # triangle peak
else:
    curve = -(1-s) * 0.6  # slow linear pushback
```

### Combined with higher frequency
- Sine: 0.5 Hz (2 sec/cycle) → Triangle: 1.5–2.0 Hz (0.5–0.67 sec/cycle)
- 3–4× faster stepping
- Visually recognizable crawling for the first time

### Video recording
- Added `--mode eval --video` flag.
- Uses MuJoCo offscreen rendering + OpenCV to save MP4.
- Camera set to follow robot or fixed zoomed-out view.

---

## Current State

### What works
- Static spider stand: stable indefinitely with RL-trained policy.
- Triangle-wave CPG gait: all 4 limbs cycle through swing/stance.
- Forward displacement: positive, sustained, not from jumping or sliding.
- All 4 limbs participate: fairness reward + lazy penalty + directional consistency.
- No torso ground contact: body stays elevated.
- No jumping: vertical velocity + airborne penalties.

### What's still improving
- Forward velocity is low (~0.05–0.15 m/s). Needs more training or gait parameter optimization.
- Arm contribution is weaker than legs (expected given the torque asymmetry).

### Key metrics (latest eval)
| Metric | Value |
|--------|-------|
| Episode survival | 1500/1500 steps |
| Forward velocity | 0.05–0.15 m/s |
| Pelvis height | 0.15–0.18 m |
| Torso-ground contact | Never |
| Leg crawling | Yes — visible cycling |
| Arm participation | Yes — all 4 limbs |
| Jumping | Eliminated |
| Directional consistency | All limbs same direction |

---

## Architecture Evolution

```
Attempt 1: RL → IK → PD → Physics           ❌ IK failures corrupt RL signal
Attempt 2: RL → PD → Physics (21D)           ❌ Action space too large, agent holds still
Attempt 3: RL → CPG → PD → Physics           ❌ CPG signs wrong, robot spins
Attempt 4: Ref + RL → PD → Physics           ❌ Slides on ice, dances in place
         ↓ FRICTION FIX + ANKLE REMOVAL ↓
Attempt 5: RL for STANDING only               ✅ Converged. Foundation works.
Attempt 6: Sine CPG + RL → PD → Physics      ⚠️ Stand gains kill motion. Sine too smooth.
Attempt 7: Triangle CPG + RL → PD → Physics  ✅ Forward crawling with all 4 limbs.
```

---

## File Structure

```
project/
├── humanoids/
│   ├── g1_29dof_lock_waist_rev_1_0.xml   # Modified: no ankles, contact geoms, friction settings
│   └── meshes/                            # STL mesh files
├── g1_pose.txt                            # Spider pose joint values
│
├── ── Tools ──
├── g1_viewer.py                           # Tkinter pose editor with sliders
├── g1_crawl.py (IK viewer)               # GLFW interactive IK with mouse picking
├── g1_crawl_ref.py                        # IK reference trajectory generator
├── diag_friction.py                       # Friction diagnostic tool
│
├── ── RL Training ──
├── g1_stand.py                            # Spider stand (hold, hold_debug, train, eval, test)
├── g1_crawl.py / g1_crawl_cpg.py         # 8-DOF CPG crawl (sine, triangle, tune, search, train, eval, test, video)
│
├── ── Failed Approaches ──
├── g1_rl.py                               # Attempt 1: IK-in-the-loop RL
├── g1_rl_direct.py                        # Attempt 2: Direct 21D joint control
├── g1_rl_cpg.py                           # Attempt 3: CPG with wrong signs
├── g1_rl_imitate.py                       # Attempt 4: Imitation learning
├── g1_stand_bullet.py                     # PyBullet attempt (loader failed)
│
├── ── Reference Data ──
├── g1_ref_forward.npz                     # Forward crawl reference
├── g1_ref_*.npz                           # Other direction references
│
├── ── Models ──
├── models/
│   ├── best/                              # Best eval checkpoint
│   ├── g1_stand_final.zip                 # Trained stand policy
│   └── g1_crawl_final.zip                 # Trained crawl policy
│
├── logs/                                  # TensorBoard training logs
├── summary.md                             # This file
└── main.tex / main.pdf                    # Formal report
```

---

## Physics Configuration (Final)

```python
# Solver
model.opt.timestep      = 0.002
model.opt.iterations    = 50
model.opt.ls_iterations = 20
model.opt.solver        = 2       # Newton

# Friction (ALL colliding geoms — the fix that saved the project)
for i in range(model.ngeom):
    if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
        model.geom_friction[i] = [3.0, 0.5, 0.5]

# Two-gain architecture:
# SETTLE (stiff)          CRAWL (soft)
# Kp = frc * 3.0          Kp = frc * 1.0
# Kd = Kp * 0.3           Kd = Kp * 0.1
# dof_damping = 3.0       dof_damping = 0.5
```

---

## Command Reference

### Standing
```bash
python g1_stand.py --mode hold                                    # PD hold test
python g1_stand.py --mode hold_debug                              # Joint saturation analysis
python g1_stand.py --mode test                                    # Random agent (must be positive)
python g1_stand.py --mode train --steps 2000000 --n_envs 8        # Train
python g1_stand.py --mode eval --checkpoint models/best/best_model # Evaluate
```

### Crawling
```bash
python g1_crawl.py --mode sine                                    # Watch sine gait
python g1_crawl.py --mode sine --pitch_amp 0.4 --freq 2.0        # Adjust params
python g1_crawl.py --mode tune                                    # Slider GUI
python g1_crawl.py --mode search                                  # Brute-force 324 sign combos
python g1_crawl.py --mode test                                    # Random agent
python g1_crawl.py --mode train --steps 5000000 --n_envs 16       # Train
python g1_crawl.py --mode eval --checkpoint models/best/best_model # Evaluate
python g1_crawl.py --mode eval --checkpoint models/best/best_model --video  # Record MP4
```

### Diagnostics
```bash
python diag_friction.py                                           # Check/fix friction values
```

---

## Known Issues and Gotchas

- **84 parallel envs → segfault.** Not RAM (62GB available). Likely OS process limit or OpenGL context limit. Use 16–32 envs.
- **PyBullet MJCF loader can't handle this XML.** Mesh paths, free joints, complex hierarchy all fail. Don't bother.
- **GPU driver too old for CUDA PyTorch.** Training runs on CPU. Works fine, just slower.
- **`pip install gymnasium` not `gym`.** The old `gym` package is deprecated.
- **`site_xvelp` doesn't exist in MuJoCo Python bindings.** Use `mj_jacSite` to compute site velocities manually.
- **Eval print bug:** `prog={'progress':+.3f}` → should be `prog={info.get('progress', 0):+.3f}`. String vs float format specifier.

---

## Reward Function Evolution (Quick Reference)

| Version | Mechanism | Exploit | Fix Applied |
|---------|-----------|---------|-------------|
| v1 | Cumulative progress | Move once, collect forever | Per-step displacement |
| v2 | Per-step displacement | Jerk back-and-forth, slide | Lifting requirement |
| v3 | Require lifting | Only legs lift, arms idle | Fairness + lazy penalty |
| v4 | Fairness enforcement | Jumping for displacement | Anti-jump penalties |
| v5 | Anti-jumping | 1 limb forward, 3 backward | Per-limb directional consistency |
| v6 | Directional consistency | — | **Current working version** |

---

## Key Lessons (Ordered by Impact)

1. **Fix the physics first.** Friction, contacts, spawn, gains — validate all of these before writing any RL code.
2. **The random agent test.** Before every training run: does a random agent survive with positive reward? If not, fix the environment.
3. **Two-gain architecture.** Stiff for settling, soft for motion. Neither alone works for both.
4. **Cumulative rewards are traps.** Always use per-step deltas.
5. **Automate what you can't intuit.** Sign search, parameter sweeps — compute is cheap, human time isn't.
6. **Reward exploits are inevitable.** For every reward term, ask "what's the laziest way to maximize this?" Then penalize that.
7. **Triangle > sine for locomotion.** Fast swing, slow stance, constant-velocity segments.
8. **Simplify aggressively.** 8 DOF with triangle waves works better than 21 DOF with sophisticated control.
9. **The simplest solution wins.** We got there after IK chains, 21D direct control, imitation learning, and sinusoidal CPGs. The final system is simpler than all of them.

---

*Last updated: April 6, 2026*
