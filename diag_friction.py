"""Quick diagnostic: what friction does each geom actually have?"""
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("humanoids/g1_29dof_lock_waist_rev_1_0.xml")
data = mujoco.MjData(model)

print("=== GEOM FRICTION VALUES ===\n")
print(f"{'Name':40s} {'Type':6s} {'ConType':8s} {'Friction':30s} {'Collides?'}")
print("-" * 100)

for i in range(model.ngeom):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
    gtype = ["plane","hfield","sphere","capsule","ellipsoid","cylinder","box","mesh"][model.geom_type[i]]
    ct = model.geom_contype[i]
    ca = model.geom_conaffinity[i]
    fric = model.geom_friction[i]
    collides = "YES" if (ct > 0 or ca > 0) else "no"
    print(f"{name:40s} {gtype:6s} ct={ct:<3d} ca={ca:<3d} [{fric[0]:.3f}, {fric[1]:.4f}, {fric[2]:.5f}]  {collides}")

print(f"\n=== OPTIONS ===")
print(f"  cone: {model.opt.cone}")
print(f"  impratio: {model.opt.impratio}")
print(f"  solver: {model.opt.solver}")
print(f"  timestep: {model.opt.timestep}")

# Now fix friction on ALL colliding geoms and test
print("\n=== FIXING FRICTION ===\n")
for i in range(model.ngeom):
    if model.geom_contype[i] > 0 or model.geom_conaffinity[i] > 0:
        model.geom_friction[i] = [3.0, 0.5, 0.5]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or f"geom_{i}"
        print(f"  Fixed: {name}")

print("\n=== DROP TEST ===\n")

# Drop and check if it stays still
data.qpos[2] = 0.3
data.qpos[3:7] = [0.707, 0, -0.707, 0]
data.qvel[:] = 0

# Simulate 3 seconds
for step in range(1500):
    mujoco.mj_step(model, data)
    if step % 500 == 499:
        x, y, z = data.qpos[0], data.qpos[1], data.qpos[2]
        vx, vy = data.qvel[0], data.qvel[1]
        print(f"  t={step*0.002:.1f}s  pos=[{x:+.4f}, {y:+.4f}, {z:.4f}]  vel=[{vx:+.4f}, {vy:+.4f}]")

drift = np.sqrt(data.qpos[0]**2 + data.qpos[1]**2)
print(f"\n  Total XY drift after 3s: {drift:.4f}m")
if drift < 0.01:
    print("  ✅ STABLE — friction works!")
else:
    print("  ❌ STILL SLIDING — need more investigation")
    print("\n  Contacts active:")
    for i in range(data.ncon):
        c = data.contact[i]
        g1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"g{c.geom1}"
        g2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"g{c.geom2}"
        print(f"    {g1_name} <-> {g2_name}  friction={c.friction}")
