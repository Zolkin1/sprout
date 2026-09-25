#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Spawn SONIC's G1 and report its DOF order, body names and prim tree.

Run this once after fetching the GR00T checkout's LFS assets. It answers the three things the
SONIC baseline env has to know about the third-party robot, none of which should be assumed:

1. **DOF order.** SONIC's observation and action vectors are in IsaacSim's breadth-first ordering
   of its URDF. If our converted articulation resolves to a different order, every reference and
   every action is silently permuted. Checked against
   :data:`~robot_rl.assets.robots.g1_sonic.G1_SONIC_ISAAC_JOINT_ORDER`.
2. **Body names.** Our terminations and events name bodies (``torso_link``, ``.*_ankle_roll_link``).
3. **Prim tree.** Our contact sensors address bodies by full nested prim path, which encodes our
   own USD's layout; the converted URDF will not match it, so the SONIC env needs its own paths
   and this is where they come from.

Usage::

    PYTHONPATH=source/robot_rl:source/motion_gen \\
      python scripts/rsl_rl/inspect_sonic_asset.py [--depth 4]
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--depth", type=int, default=5, help="Prim-tree depth to print.")
parser.add_argument("--force_convert", action="store_true",
                    help="Re-run the URDF->USD conversion even if a cached USD exists.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils                                            # noqa: E402
from isaaclab.assets import Articulation                                    # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext                   # noqa: E402

from robot_rl.assets.robots.g1_sonic import (                               # noqa: E402
    G1_SONIC_ISAAC_JOINT_ORDER,
    g1_sonic_cfg,
)


def _print_tree(prim, depth: int, indent: int = 0) -> None:
    """Print a USD prim subtree down to ``depth``."""
    if depth < 0:
        return
    name = prim.GetName()
    kind = prim.GetTypeName()
    print(f"    {'  ' * indent}{name}  <{kind}>")
    for child in prim.GetChildren():
        _print_tree(child, depth - 1, indent + 1)


def main() -> int:
    """Spawn the articulation and print what the env cfg needs to know."""
    sim = SimulationContext(SimulationCfg(dt=0.005, device=args_cli.device or "cuda:0"))
    cfg = g1_sonic_cfg()
    if args_cli.force_convert:
        cfg.spawn.force_usd_conversion = True
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    robot = Articulation(cfg.replace(prim_path="/World/Robot"))
    sim.reset()

    print("\n" + "=" * 78)
    print(f"SONIC G1: {robot.num_joints} joints, {robot.num_bodies} bodies")
    print("=" * 78)

    got, want = list(robot.joint_names), list(G1_SONIC_ISAAC_JOINT_ORDER)
    print("\n--- DOF order ---")
    for i, n in enumerate(got):
        flag = "" if i < len(want) and want[i] == n else "   <-- MISMATCH"
        print(f"  {i:2d} {n}{flag}")
    if got == want:
        print("\n  OK: matches G1_SONIC_ISAAC_JOINT_ORDER exactly.")
    else:
        print("\n  MISMATCH vs G1_SONIC_ISAAC_JOINT_ORDER -- the SONIC baseline would feed the")
        print("  policy a permuted observation. Reconcile before running any rollout.")
        print(f"  expected: {want}")

    print("\n--- body names ---")
    for i, n in enumerate(robot.body_names):
        print(f"  {i:2d} {n}")

    print("\n--- default joint pos (DOF order) ---")
    import warp as wp

    def _t(x):
        """Read an IsaacLab data field as torch (Newton hands back warp arrays)."""
        return x if hasattr(x, "detach") else wp.to_torch(x)

    dq = _t(robot.data.default_joint_pos)[0].tolist()
    for n, v in zip(robot.joint_names, dq):
        if abs(v) > 1e-9:
            print(f"  {n:32s} {v:+.4f}")

    # Mass and inertia are the usual reason a converted URDF explodes on the first physics step:
    # a near-massless body driven by a real PD torque sees an enormous angular acceleration.
    print("\n--- body mass / inertia as SIMULATED ---")
    mass = _t(robot.data.default_mass)[0]
    inertia = _t(robot.data.default_inertia)[0].reshape(robot.num_bodies, 3, 3)
    diag = inertia[:, [0, 1, 2], [0, 1, 2]]
    print(f"  total mass {float(mass.sum()):.3f} kg")
    order = mass.argsort()
    for i in order[:8].tolist():
        print(f"  {robot.body_names[i]:32s} m={float(mass[i]):.6f} kg  "
              f"I_diag={[f'{float(v):.3e}' for v in diag[i]]}")
    tiny_m = [robot.body_names[i] for i in range(robot.num_bodies) if float(mass[i]) < 1e-4]
    tiny_i = [robot.body_names[i] for i in range(robot.num_bodies)
              if float(diag[i].min()) < 1e-9]
    print(f"  bodies with mass < 1e-4 kg : {tiny_m or 'none'}")
    print(f"  bodies with a zero inertia axis: {tiny_i or 'none'}")

    print("\n--- applied actuator gains (DOF order) ---")
    kp = _t(robot.data.joint_stiffness)[0]
    kd = _t(robot.data.joint_damping)[0]
    arm = _t(robot.data.joint_armature)[0]
    eff = _t(robot.data.joint_effort_limits)[0] if hasattr(robot.data, "joint_effort_limits") \
        else None
    for i, n in enumerate(robot.joint_names):
        e = f" eff={float(eff[i]):7.1f}" if eff is not None else ""
        print(f"  {n:32s} kp={float(kp[i]):9.3f} kd={float(kd[i]):8.4f} "
              f"armature={float(arm[i]):.6f}{e}")

    print(f"\n--- prim tree under /World/Robot (depth {args_cli.depth}) ---")
    import isaacsim.core.utils.stage as stage_utils
    stage = stage_utils.get_current_stage()
    _print_tree(stage.GetPrimAtPath("/World/Robot"), args_cli.depth)
    print()
    return 0


if __name__ == "__main__":
    main()
    simulation_app.close()
