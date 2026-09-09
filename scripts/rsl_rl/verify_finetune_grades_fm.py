#!/usr/bin/env python
# Copyright 2026 Zachary Olkin. All rights reserved.

"""Verify the fine-tune task GRADES the generator, and that the body links reach the CLF.

Replaces ``verify_finetune_serves_fm.py``, which checked the opposite invariant.  That
script belonged to the first fine-tune design, where the generator fed the observation alone
and the CLF still graded the clip -- so it asserted that the clip serve was *restored* after
the observation was taken, and failed with "the CLF is grading the generator" if it was not.
Under ``reference_source="fm"`` grading the generator IS the design, so that assertion is
now exactly backwards and there is nothing in it to keep.

What this checks instead, and each one fails silently otherwise:

1. **The generator is APPLIED and moving.**  A generator that quietly stopped re-planning
   would serve a frozen reference, which is a perfectly finite thing to track.
2. **Every channel the body CLF grades is finite.**  ``strict_fm_only`` NaNs each ``y_des``
   channel the generator does not write, and in body mode 91 + 78 of them are written by
   ``_write_fm_root_body_frames`` rather than by the generator.  Miss the registration and
   ``v`` goes NaN; miss the write ORDER and it does not -- see (4).
3. **``clf_reward`` is finite and non-zero.**  This is the whole reward, and the two
   interesting failure modes both land here: a NaN ``v`` from (2), and a reward pinned at 0
   from the pelvis-relative-read-as-world bug.
4. **The body links agree with a fresh FK of the served joints**, checked through the
   distances between them -- which the world lift preserves and a wrong FK, a wrong joint
   order or a wrong body order all break.  This is the live seam that
   ``test_fm_body_fk.py`` pins offline: that test drives the write on a stub, this one
   drives it inside the env.

Run (from the repo root, ``env_isaaclab`` active, and ``python -u`` because Kit discards
buffered stdout at shutdown)::

    PYTHONPATH=source/robot_rl:source/motion_gen python -u \\
        scripts/rsl_rl/verify_finetune_grades_fm.py \\
        --clip clips/varied60_train.h5 --fm_checkpoint logs/flow_matching/<run>/best.pt

Exits non-zero, naming the failure, on any of the four.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="G1-box2-finetune",
                    help="the fine-tune task; must run reference_source='fm'")
parser.add_argument("--clip", required=True, help="clip library .h5 (no default, by design)")
parser.add_argument("--clip_device", default="cpu")
parser.add_argument("--fm_checkpoint", required=True, help="the generator to grade against")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=40,
                    help="control steps to run; must exceed one re-plan period")
parser.add_argument("--fk_atol", type=float, default=2e-3,
                    help="tolerance (m) on the served body-link distances vs a fresh FK")
AppLauncher.add_app_launcher_args(parser)
# parse_known_args + the argv rewrite below, exactly as train.py does it: hydra_task_config
# re-parses sys.argv itself and rejects anything it does not own, so this script's own flags
# have to be taken out of the way first.
args_cli, hydra_args = parser.parse_known_args()
args_cli.headless = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AFTER SimulationApp, exactly as train.py does it: the contact sensors this task declares
# need the recursive-traversal patch, and importing it earlier is a no-op.
import robot_rl.sensors._recursive_contact_sensor_impl  # noqa: E402,F401

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import robot_rl.tasks  # noqa: E402,F401  (registers the task ids)
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from motion_gen import features as F  # noqa: E402
from motion_gen.fk import FKChain  # noqa: E402
from motion_gen.joint_names import strip_root_marker  # noqa: E402


class _VerifyFailed(RuntimeError):
    """A checked invariant broke.

    Deliberately NOT ``SystemExit``: raising that out of ``main`` skips
    ``simulation_app.close()``, and Kit then shuts down on its own and **replaces the exit
    code with 0** -- so the failure prints and the script reports success.  The old
    ``verify_finetune_serves_fm.py`` had exactly that shape, which means its documented
    "exits non-zero" never held.  Caught in ``__main__`` instead, after the app is closed.
    """


def _fail(msg: str) -> None:
    """Abort the run, naming the invariant that broke.

    Args:
        msg: What broke, and where to look.

    Raises:
        _VerifyFailed: Always.
    """
    raise _VerifyFailed(msg)


def _equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether two tensors match, counting NaN as equal to NaN in the same slot.

    ``strict_fm_only`` writes NaN into every channel the generator does not supply -- 28 of
    155 position channels on the shipped block selection -- and ``torch.equal`` is False
    whenever a NaN is present, however identical the two buffers are.  A plain ``equal``
    here reports "the applied reference is not the generator's" on a perfectly healthy run.

    Args:
        a: First tensor.
        b: Second tensor, same shape.

    Returns:
        True if every slot either matches or is NaN in both.
    """
    same = a == b
    both_nan = torch.isnan(a) & torch.isnan(b)
    return bool((same | both_nan).all())


def _check_reference_is_the_generators(cmd, y_hist: list[torch.Tensor]) -> None:
    """(1) The applied reference is the generator's, and it is moving.

    Args:
        cmd: The live ``FmTrackingCommand``.
        y_hist: One ``y_des`` clone per step, oldest first.

    Raises:
        _VerifyFailed: If the task grades something else, or the reference never moves.
    """
    if cmd.cfg.reference_source != "fm":
        _fail(f"reference_source={cmd.cfg.reference_source!r}; this task does not grade the "
              "generator, so there is nothing here to verify")
    if not cmd.fm_enabled:
        _fail("the command reports fm_enabled=False, so no generator is loaded")
    # The applied buffer must be the FM cache, not the clip one. Cheap, and it is the single
    # field the whole reward re-pointing rests on.
    if getattr(cmd, "_y_des_fm", None) is None:
        _fail("no cached FM reference; _build_fm_reference never ran")
    if not _equal_with_nan(cmd.y_des, cmd._y_des_fm):
        _fail("the applied y_des is not the generator's cached reference")

    # Over the GRADED channels only: the unwritten ones are NaN, and NaN - NaN is NaN, which
    # would make `moved` NaN and the comparison below vacuously false.
    graded = [y[:, cmd.clf.pos_gather] for y in y_hist]
    moved = max(float((graded[i] - graded[0]).abs().max()) for i in range(1, len(graded)))
    print(f"[verify] worst |y_des(t) - y_des(0)| over {len(y_hist)} steps : {moved:.5f}")
    if moved < 1e-4:
        _fail("the reference never moved; the generator is serving a frozen plan (a frozen "
              "reference is perfectly finite to track, so nothing else here would notice)")


def _check_graded_channels_are_finite(cmd) -> None:
    """(2) Every channel the CLF actually grades is finite.

    Reads the CLF's own gather indices rather than a channel-name list, so this covers
    exactly what enters ``eta`` under the live ``tracking_mode`` and nothing else.

    Raises:
        _VerifyFailed: If any graded channel is NaN, or if a ``@root`` slot is unwritten.
    """
    pos_bad = int((~torch.isfinite(cmd.y_des[:, cmd.clf.pos_gather])).sum())
    vel_bad = int((~torch.isfinite(cmd.dy_des[:, cmd.clf.vel_gather])).sum())
    print(f"[verify] graded channels: {cmd.clf.pos_gather.numel()} pos / "
          f"{cmd.clf.vel_gather.numel()} vel, non-finite {pos_bad} / {vel_bad}")
    if pos_bad or vel_bad:
        _fail(f"{pos_bad} position and {vel_bad} velocity channels the CLF grades are NaN. "
              "Under strict_fm_only that is an unwritten channel -- in body mode the '@root' "
              "links are written by _write_fm_root_body_frames and registered in "
              "_build_strict_masks, so check that registration first")

    # And say it the other way round: no '@root' slot may appear in the unwritten sets. The
    # counts are derived here rather than restated -- 91 pos + 78 vel for 13 links.
    maps = cmd._root_frame_maps()
    if maps is None:
        _fail("the library carries no '@root' channels, so body-link tracking is impossible")
    pos_i = torch.cat([maps[0], maps[1]])
    vel_i = torch.cat([maps[2], maps[3]])
    for name, idx, unwritten in (("position", pos_i, cmd._unwritten_pos),
                                 ("velocity", vel_i, cmd._unwritten_vel)):
        overlap = len(set(idx.tolist()) & set(unwritten.tolist()))
        if overlap:
            _fail(f"{overlap} '@root' {name} channels are in the unwritten set")
    print(f"[verify] '@root' block: {pos_i.numel()} pos / {vel_i.numel()} vel channels, "
          "all registered as written")


def _check_the_reward_is_alive(env, cmd) -> None:
    """(3) ``clf_reward`` is finite and non-zero.

    Both interesting failures land here: a NaN ``v`` from an unwritten graded channel, and a
    reward pinned at 0 from ``@root`` poses left pelvis-relative while the CLF reads them as
    world.  The second has no other symptom at all.

    Raises:
        _VerifyFailed: If ``v`` or the reward is non-finite, or the reward is identically 0.
    """
    if not bool(torch.isfinite(cmd.v).all()):
        _fail("the CLF value v is non-finite; a graded channel is NaN")
    term = env.reward_manager.get_term_cfg("clf_reward")
    rew = term.func(env, **term.params)
    print(f"[verify] v          min {float(cmd.v.min()):.4f}  max {float(cmd.v.max()):.4f}")
    print(f"[verify] clf_reward min {float(rew.min()):.6f}  max {float(rew.max()):.6f}  "
          f"mean {float(rew.mean()):.6f}")
    if not bool(torch.isfinite(rew).all()):
        _fail("clf_reward is non-finite")
    if float(rew.max()) <= 0.0:
        _fail("clf_reward is identically 0 across every env. The classic cause is the '@root' "
              "poses being written AFTER _write_world_des, which leaves them pelvis-relative "
              "in channels the CLF reads as world -- finite, enormous, and invisible "
              "otherwise")


def _check_the_body_links_are_the_served_posture(cmd, atol: float) -> None:
    """(4) The served ``@root`` links agree with a fresh FK of the served joints.

    Compared through the DISTANCES from the base and between the links, because ``y_des``
    holds them lifted to world by then and a rigid transform preserves exactly those.  A
    wrong FK model, a wrong joint order and a wrong body order all move them.

    Args:
        cmd: The live command.
        atol: Tolerance in metres.

    Raises:
        _VerifyFailed: If the distances disagree.
    """
    frames = [strip_root_marker(f) for f in cmd.root_body_frames]
    chain = FKChain(str(F.DEFAULT_MJCF), list(cmd.ctx.joint_names), frames,
                    base_link=cmd.cfg.base_frame_name).to(cmd.device)
    served = cmd._pred_world(cmd._first_two_offsets)["joint_pos"][:, 0]     # [N, nj]
    fk_pos, _rot, _jv, _jw = chain.pose_and_jacobian(served)                # [N, B, 3] base frame

    maps = cmd._root_frame_maps()
    n, nb = served.shape[0], len(frames)
    world = cmd.y_des[:, maps[0]].view(n, nb, 3)                            # lifted to world

    # From the base: |p_i - p_base| in world == |FK_i| in the base frame.
    base = cmd.y_des[:, cmd._wf_pos_flat[:3]].view(n, 1, 3)
    d_world = (world - base).norm(dim=-1)
    d_fk = fk_pos.norm(dim=-1)
    worst_radial = float((d_world - d_fk).abs().max())

    # And between the links, which additionally catches a permuted body order.
    pair_world = torch.cdist(world, world)
    pair_fk = torch.cdist(fk_pos, fk_pos)
    worst_pair = float((pair_world - pair_fk).abs().max())

    print(f"[verify] body links vs fresh FK: worst radial {worst_radial:.6f} m, "
          f"worst pairwise {worst_pair:.6f} m  (atol {atol})")
    if worst_radial > atol or worst_pair > atol:
        _fail(f"the served '@root' links do not match an FK of the served joint_pos "
              f"(radial {worst_radial:.6f} m, pairwise {worst_pair:.6f} m). Suspect the FK "
              "model, the joint order (ctx.joint_names, not the library's), or the body order")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg) -> None:
    """Step the fine-tune env and assert the generator is what gets graded.

    Composed through hydra rather than ``parse_env_cfg`` so the env is built the way
    ``train.py`` builds it -- the two paths do not resolve the same config.

    Args:
        env_cfg: The task's env config, composed by hydra.
        agent_cfg: The paired rsl-rl config; unused, but the decorator supplies it.

    Raises:
        _VerifyFailed: On any of the four checks above.
    """
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.commands.traj_ref.path = args_cli.clip
    env_cfg.commands.traj_ref.clip_device = args_cli.clip_device
    env_cfg.scene.terrain.clip_library_path = args_cli.clip
    env_cfg.commands.traj_ref.fm_checkpoint = args_cli.fm_checkpoint

    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    cmd = unwrapped.command_manager.get_term("traj_ref")
    print(f"[verify] task={args_cli.task}")
    print(f"[verify] reference_source={cmd.cfg.reference_source!r}  "
          f"history_from={cmd.cfg.history_from!r}  "
          f"tracking_mode={getattr(cmd.cfg, 'tracking_mode', 'joint')!r}  "
          f"fm_update_every={cmd.cfg.fm_update_every}  fm_enabled={cmd.fm_enabled}")
    if getattr(cmd.cfg, "tracking_mode", "joint") != "body":
        print("[verify] NOTE: tracking_mode is not 'body', so check (4) covers nothing and "
              "the '@root' registration in (2) is vacuous")

    # Printed from the live manager rather than computed by hand: a 'critic' width that does
    # not match the resumed checkpoint's means the observation cfg drifted, and it would
    # otherwise surface as a load error far from the cause.
    for group, dim in unwrapped.observation_manager.group_obs_dim.items():
        width = int(dim[0]) if not isinstance(dim, int) else int(dim)
        print(f"[verify] obs group {group:12s} width {width}")

    env.reset()
    zero_action = torch.zeros(
        unwrapped.num_envs, unwrapped.action_manager.total_action_dim, device=unwrapped.device
    )
    if args_cli.steps <= int(cmd.cfg.fm_update_every):
        _fail(f"--steps {args_cli.steps} does not exceed the re-plan period "
              f"{cmd.cfg.fm_update_every}, so the 'is it moving' check cannot see a re-plan")

    y_hist: list[torch.Tensor] = []
    for _ in range(args_cli.steps):
        env.step(zero_action)
        y_hist.append(cmd.y_des.clone())

    _check_reference_is_the_generators(cmd, y_hist)
    _check_graded_channels_are_finite(cmd)
    _check_the_reward_is_alive(unwrapped, cmd)
    _check_the_body_links_are_the_served_posture(cmd, args_cli.fk_atol)

    env.close()
    print("[verify] PASS: the generator is applied, its body links reach the CLF, and the "
          "reward is alive.")


if __name__ == "__main__":
    # The app is closed before the exit code is set, and the exit code is set last. Kit
    # replaces a non-zero code with 0 if it has to shut itself down, so a bare
    # `raise SystemExit` out of main() would print the failure and report success.
    _failure: str | None = None
    try:
        main()
    except _VerifyFailed as exc:
        _failure = str(exc)
        print(f"[verify] FAIL: {_failure}")
    simulation_app.close()
    sys.exit(1 if _failure is not None else 0)
