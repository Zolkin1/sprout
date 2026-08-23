# Upstream reference URDFs

Vendored, unmodified, purely as a comparison datum. Nothing loads these at runtime.

## `g1_29dof_mode_11.urdf`

- source: <https://github.com/unitreerobotics/unitree_ros>, `robots/g1_description/g1_29dof_mode_11.urdf`
- fetched: 2026-08-19, `sha256 05c15b116ba7abeea0ebe2cf307a81de383c3881f1986b303bb79bec126207de`
- licence: BSD-3-Clause (the upstream repository's)

**mode_11 is the variant this robot is** — high-torque hip (the 22.5 gear), revolute-35
waist, light wrists. Unitree ships modes 10 and 12–18 alongside it, and they differ in
exactly the fields that matter here, so the mode in the filename is load-bearing: reading
the hand or the motor limits out of `g1_29dof_rev_1_0.urdf` is a different robot's answer
that happens to agree in some fields and not others.

It is vendored rather than fetched on demand so `apply_mode11_deltas.py` derives its numbers
from a file under version control instead of from a transcription, and so the diff that
justified each delta can be re-run offline. Re-vendor (and re-run the patch) to take an
upstream revision; the patch prints everything it changes.

### What our own source deliberately does NOT take from it

`g1_21j_29dof.urdf` carries local customisations that must survive any sync:

| ours | why |
|---|---|
| `pelvis_link`, not `pelvis` | the whole stack keys off the name; the inertials are byte-identical |
| box foot collider `0.20311 x 0.06547 x 0.01851` | replaces Unitree's four 5 mm contact spheres |
| `*_heel` / `*_toe` / `*_foot_middle` frames (14) | ours, massless |
| `pelvis_contour_link`, `waist_support_link` | 1 g visual tokens upstream dropped |
| no `d435_link` / `mid360_link` / `imu_in_*` | we mount our own cameras; see `robot_assets/g1_perceptive/` |
