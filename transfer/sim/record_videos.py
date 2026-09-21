#!/usr/bin/env python3
# Copyright 2026 Zachary Olkin. All rights reserved.
"""Named MuJoCo videos for the paper, the talk and the website.

Every clip this produces has a NAME, so it can be asked for again:

    python record_videos.py --stage list
    python record_videos.py --stage all  --shots wall_stairs_parent_a,wall_stairs_ft_a
    python record_videos.py --stage render --shots four_box_sweep --vel_plot bottom --rt 0.5

**Two stages, and the split is the point.**  ``roll`` runs the policy and writes a trace;
``render`` turns a trace into pixels.  The look is going to be re-tuned -- colours, camera,
the velocity panel -- and re-tuning it must not re-run the policy, or each pass would show a
different episode and the A/B would quietly stop being an A/B.  ``--stage all`` does both;
``--stage render`` on its own re-renders from what is already on disk and takes seconds.

**The A/B claim these clips make.**  The mode-probe shots come in ``parent`` / ``ft`` pairs
that share EVERYTHING except the generator checkpoint: the same tile, the same spawn, the
same tracker, the same commanded speed, and the same flow-noise seed -- ``mode_probe``
re-seeds the generator's RNG per trial, so trial ``s`` of both arms starts from the same
tape.  (The two tapes stay aligned only while the arms agree on when to re-plan, which is
until they diverge; the SEED is identical, the realised draws after the divergence are not,
and that is what "same seed" means here.)

**A rolled shot is verified against the probe's own table.**  ``mode_probe`` already stored
each trial's verdict, step count and furthest x; a re-roll that does not reproduce them
exactly is refused, because then the footage would not be of the episode that was labelled.
That check is what makes the human mode labels -- ``labels.json`` -- quotable about the clip.

Output lands in ``logs/video/<shot>/``: ``trace.npz`` (poses + the velocity trace),
``shot.json`` (provenance and what the roll reproduced), and the mp4s.  ``INDEX.md`` at the
top lists everything that has been made.

**Run it in ``env_isaaclab``, both stages.**  ``render`` needs no onnxruntime, but it does need
``imageio`` + ``imageio-ffmpeg``, and ``transfer/requirements.txt`` deliberately carries neither
-- ``env_transfer`` exists to be small enough for a Jetson and a video encoder has no business
in it.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

#: This file's directory and the checkout it lives in.  ``abspath``, never ``realpath``: the
#: worktree's ``logs/`` is a symlink into the main checkout and resolving through it would
#: silently import and load the OTHER tree's code (see ``logs/mode_lag/_treepath.py``).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for _d in (_HERE, os.path.join(_REPO, "source", "motion_gen")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import video_render as vr                                              # noqa: E402
from mjeval import terrains                                            # noqa: E402

#: Where the shots go.
VIDEO_ROOT = os.path.join(_REPO, "logs", "video")


# =============================================================================================
# Shots
# =============================================================================================

@dataclass(frozen=True)
class Shot:
    """One named video.

    Attributes:
        name: The handle.  Names the directory, the mp4 and the scene file.
        story: One line saying what the clip shows.  Printed by ``--stage list`` and stamped
            into the shot manifest.
        kind: ``"probe"`` replays one trial of a ``mode_probe`` directory; ``"lane"`` rolls a
            fresh episode on an authored lane from :data:`LANES`.
        probe: For ``probe`` shots, the directory under ``logs/`` holding ``probe.json``.
        arm: ``parent`` or ``finetuned`` -- a key of that probe's ``arms``.
        seed: The trial, which is also the flow-noise seed.
        lane: For ``lane`` shots, a key of :data:`LANES`.
        speed: Commanded forward speed (m/s) for a lane shot.  Probe shots take the probe's.
        lane_width: Full lateral extent of a lane shot's features (m); 0 keeps the source
            scene's own.  See :func:`terrain_scene_boxes` for why the four-box lane is
            widened and what it does and does not change.
        lane_spacing: Distance between a lane shot's feature centres (m); 0 keeps the source
            scene's own.
        steer_kp: Heading-controller gain for a LANE shot (1/s); a probe shot takes the
            probe's own, which is stamped in its manifest.  Unsteered, a held forward command
            on an open lane drifts: the first roll of ``four_box_sweep`` was 5.0 m off centre
            by x=11 m and walked PAST the last two boxes on the floor.  1.0 is what the
            free-form fine-tune task itself runs (``fm_tracking_cmd.freeform_yaw_rate``).
            It is off-distribution on the STAIR families and has no business on one.
        tracker_onnx / fm_onnx: Artifacts for a lane shot, relative to the repo root.
        camera / theme / terrain_shade / wall_alpha / vel_plot / vel_channels / rt: Render
            defaults; every one is overridable from the command line.
        cam_azimuth / cam_elevation / cam_distance / cam_height: Per-shot overrides on the
            named preset, ``None`` to take the preset's own.  A preset is a house style and
            some shots need more room than it gives -- a 20-riser flight framed at the
            preset's 4.6 m shows five treads.
        stop_x_pad: Metres past the course's GOAL LINE at which the footage is cut.  These
            tiles author their own ground and get no infinite floor, so an episode that
            arrives keeps walking and falls off the end of the world -- which is an artifact
            of the recording, not a result, and has no business in a clip.  ``None`` keeps
            the whole episode.
        hold_end_s: Seconds of freeze on the last frame.
    """

    name: str
    story: str
    kind: str
    probe: str = ""
    arm: str = ""
    seed: int = -1
    lane: str = ""
    speed: float = 0.0
    steer_kp: float = 0.0
    lane_width: float = 0.0
    lane_spacing: float = 0.0
    tracker_onnx: str = ""
    fm_onnx: str = ""
    camera: str = "quarter"
    cam_azimuth: float | None = None
    cam_elevation: float | None = None
    cam_distance: float | None = None
    cam_height: float | None = None
    theme: str = "bright"
    terrain_shade: str = "height"
    wall_alpha: float = 0.30
    vel_plot: str = "off"
    vel_channels: tuple[str, ...] = ("vx",)
    rt: float = 1.0
    stop_x_pad: float | None = 0.4
    hold_end_s: float = 1.0


#: The generator + tracker snapshot the ftqueue_lag mode probes were rolled against; see
#: ``logs/mode_lag/roll_lag.sh`` for why every artifact is read from that directory and not
#: from a run's ``exported/``, which every ``play.py`` overwrites.
_CKPT = "logs/mode_lag/ckpt"

SHOTS: tuple[Shot, ...] = (
    # -- narrow walled up-into-down staircase (varied_updown tile 23) ------------------------
    # 50 blind human labels per arm: the parent calls the first feature a BOX on 32 of 50 and
    # is never sure it is stairs; the fine-tune calls it STAIRS on 39 of 50.  Arrival follows:
    # 2/25 against 22/25 on the scored half of the probe.
    Shot("wall_stairs_parent_a",
         "Parent generator, narrow walled staircase: plans a BOX onto the first riser and "
         "falls there (human label 'box'; furthest x 4.05 m of 14.2).",
         "probe", probe="logs/mode_lag/updown_wall_t23", arm="parent", seed=9,
         camera="chase", wall_alpha=0.28),
    Shot("wall_stairs_ft_a",
         "Same tile, same seed, same tracker, AWR fine-tuned generator: plans STAIRS and "
         "walks the flight (human label 'stairs'; reaches the goal).",
         "probe", probe="logs/mode_lag/updown_wall_t23", arm="finetuned", seed=9,
         camera="chase", wall_alpha=0.28),
    # The OTHER shape of the same failure, and the reason this pair is seed 28 rather than
    # another start-of-flight one: the parent climbs the flight, crosses the platform, and
    # then plans the JUMP at the descent lip. Measured from the model's own ankle sites,
    # both feet are >12 cm clear of the terrain for 0.36 s at x 8.41 -> 8.94, which is the
    # lip; the seed-9 clip's airborne phase is at x 3.95 -> 4.16, the FIRST riser.
    Shot("wall_stairs_parent_b",
         "Parent generator, same tile: it climbs the flight and crosses the platform, then "
         "LAUNCHES off the top instead of stepping down -- 0.36 s with both feet clear at "
         "the descent lip -- and goes down at x 11.15 m. Its blind human label is 'unclear' "
         "because that label covers only the FIRST decision, 4 m earlier.",
         "probe", probe="logs/mode_lag/updown_wall_t23", arm="parent", seed=28,
         camera="chase", wall_alpha=0.28),
    # NOTE on its human label, which is 'unclear' and NOT evidence against this clip. The
    # blind labelling window ends 2 m past the FIRST feature, so it covers the ascent and
    # nothing else; the decision this clip is about happens 4 m further on, at the lip, and
    # the protocol does not score it. The airborne measurement above does.
    Shot("wall_stairs_ft_b",
         "Same seed, fine-tuned: up the flight, across the platform and STEPPED down the "
         "far side, out to 14.41 m.",
         "probe", probe="logs/mode_lag/updown_wall_t23", arm="finetuned", seed=28,
         camera="chase", wall_alpha=0.28),

    # -- two boxes (varied_boxes tile 6, the library's tallest first box at 0.722 m) ---------
    Shot("two_box_parent_a",
         "Parent generator, two-box course: falls at the FIRST box (furthest x 4.94 m).",
         "probe", probe="logs/mode_lag/boxes_t6", arm="parent", seed=20, camera="quarter"),
    Shot("two_box_ft_a",
         "Same seed, fine-tuned generator: clears both boxes and reaches the goal.",
         "probe", probe="logs/mode_lag/boxes_t6", arm="finetuned", seed=20, camera="quarter"),
    Shot("two_box_parent_b",
         "Parent clears the first box and falls at the SECOND (furthest x 10.20 m).",
         "probe", probe="logs/mode_lag/boxes_t6", arm="parent", seed=7, camera="quarter"),
    Shot("two_box_ft_b",
         "Same seed, fine-tuned: both boxes, reaches the goal.",
         "probe", probe="logs/mode_lag/boxes_t6", arm="finetuned", seed=7, camera="quarter"),

    # -- the velocity-tracking claim ---------------------------------------------------------
    Shot("four_box_sweep",
         "One held 1.5 m/s forward command, unsteered, over terrain_scene's four-box ladder "
         "(0.25 / 0.40 / 0.55 / 0.70 m at 3 m spacing), with commanded against achieved "
         "velocity drawn. Draw 0; it climbs all four. Mean achieved vx 0.28 m/s while over "
         "a box against 1.61 on the open floor past the ladder -- but the 3 m gaps are "
         "SHORTER than the approach plus the recovery, so between two boxes the pair never "
         "gets back to the command and the panel shows a surge-stall cycle rather than a "
         "flat line with dips. The boxes are widened to 8 m LATERALLY (only): nothing in "
         "this stack keeps a lane, and at the scene's own 2.59 m not one of 40 draws "
         "crossed all four. `--lane_spacing 6` is the variant where the claim reads.",
         "lane", lane="terrain_scene_boxes", speed=1.5, steer_kp=0.0, lane_width=8.0,
         seed=0, stop_x_pad=4.0,
         tracker_onnx=f"{_CKPT}/exported/tracker.onnx",
         fm_onnx=f"{_CKPT}/ft_multi_lag_i5_fm.onnx",
         camera="quarter", vel_plot="bottom", vel_channels=("vx",)),
    Shot("four_box_sweep_steered",
         "The same four-box ladder with the heading controller ON (steer_kp 1.0, what the "
         "free-form fine-tune task itself runs), so the robot holds the lane instead of "
         "drifting off it. 10 of 12 draws climb all four at a 6 m lateral width. The trade "
         "is steep and the panel shows it: steering costs most of the forward speed.",
         "lane", lane="terrain_scene_boxes", speed=1.5, steer_kp=1.0, lane_width=6.0,
         seed=0, stop_x_pad=4.0,
         tracker_onnx=f"{_CKPT}/exported/tracker.onnx",
         fm_onnx=f"{_CKPT}/ft_multi_lag_i5_fm.onnx",
         camera="quarter", vel_plot="bottom", vel_channels=("vx", "wz")),

    # -- the playground's own long flight, and a real building's entrance ---------------------
    Shot("tall_stairs_up",
         "terrain_scene's LONG staircase: 20 risers to +2.79 m at 0.14 rise / 0.32 run, one "
         "held forward command at 1.0 m/s, cut at the top platform. Draw 4 -- 6 of 10 draws "
         "climbed the whole flight, and 4 is the one that stayed most nearly on the "
         "centreline of a 1.52 m-wide flight.",
         "lane", lane="terrain_scene_tall_stairs_up", speed=1.0, steer_kp=0.0,
         seed=4, stop_x_pad=1.2,
         tracker_onnx=f"{_CKPT}/exported/tracker.onnx",
         fm_onnx=f"{_CKPT}/ft_multi_lag_i5_fm.onnx",
         # Side-on and well back: a 20-riser flight is 6.4 m long and 2.8 m tall, and the
         # three-quarter preset frames about five treads of it.
         camera="side", cam_distance=7.5, cam_elevation=-6.0, cam_height=0.6,
         vel_plot="off"),
    Shot("entry_stairs_down",
         "Descending the measured building entrance (entry_stairs_down, tile 45): off the "
         "raised clearing, through the narrowing doorway, down both flights. Chosen for a "
         "CLEAN descent -- zero frames with both feet clear of the terrain anywhere on the "
         "flights, and 0.38 m of lateral excursion across them.",
         "probe", probe="logs/mode_lag/multi_entry_down_t45", arm="finetuned", seed=56,
         # `chase` and not `quarter`: the descent is flanked by two 1.96 m walls and entered
         # through a doorway, and a three-quarter view puts the near wall between the camera
         # and the robot for the whole traverse -- fading it is not enough when it is what
         # the shot is partly about.
         camera="chase", wall_alpha=0.30),
)

SHOT_BY_NAME: dict[str, Shot] = {s.name: s for s in SHOTS}


# =============================================================================================
# Authored lanes
# =============================================================================================

def terrain_scene_boxes(speed: float, name: str, width: float = 0.0,
                        spacing: float = 0.0) -> terrains.Course:
    """The four-box ladder of ``g1_perceptive_terrain_scene.xml``, as an mjeval course.

    **The downrange geometry is READ OUT of the committed scene, not restated.**
    ``terrain_scene`` is the in-distribution sim2sim playground and its box lane is the one
    this clip is about; a second copy of "0.25, 0.40, 0.55, 0.70 m at 3 m spacing" here would
    be one more thing to drift.  Two things ARE changed, both lateral:

    * the lane lives at ``y = -6.5`` in that scene, beside four staircases, and mjeval's
      spawn is the world origin, so the boxes are moved to ``y = 0``; and
    * ``width`` may widen them.

    **Why widening is offered, and what it costs.**  Nothing in this stack keeps a lane.  On
    an open floor under a held command the pair drifts sideways -- measured here, 40 draws at
    1.5 m/s and ``steer_kp=1.0``: **not one climbed all four boxes**, 24 of 40 walked PAST one
    of them on the floor, and the peak lateral excursion ran to 3.4 m against a box half-width
    of 1.30 m.  Slowing to 1.0 m/s made it worse, since the drift has longer to build.  So at
    terrain_scene's own 2.59 m width the clip would be about lane-keeping, which this stack
    deliberately has none of (see the root ``CLAUDE.md``), and not about what it is for --
    velocity tracking across a traversal.  Widening removes the lane-keeping failure and
    changes the CLIMB not at all: every height, depth and spacing is the scene's own.  It is
    a property of the recording and is stated in the course note and in ``INDEX.md``.

    Args:
        speed: Commanded forward speed (m/s); sets the time budget.
        name: Scene name for the emitted XML.
        width: Full lateral extent of every box (m).  ``0`` keeps the scene's own.
        spacing: Distance between consecutive box CENTRES (m).  ``0`` keeps the scene's own
            3 m.  **What it is for:** at 3 m the pair never gets back to the command between
            features -- measured unsteered on the widened lane, mean achieved 0.77 m/s of a
            commanded 1.5, and 0.89 even off a feature, while the same pair holds 1.45 m/s
            on open flat.  A clip meant to show "the command is tracked except across a
            traversal" therefore needs gaps long enough for the "except" to have somewhere
            to happen.  Heights, depths and the first box's face are untouched.

    Returns:
        The course.

    Raises:
        ValueError: If the scene holds no ``box<N>`` geoms -- which would mean the playground
            has been regenerated without the box lane, and the clip cannot be made.
    """
    import generate_clip_tile_scene as gcts                            # noqa: PLC0415

    src = os.path.join(_HERE, "robots", "g1", "g1_perceptive_terrain_scene.xml")
    root = ET.parse(src).getroot()
    boxes = []
    for g in root.findall("./worldbody/geom"):
        n = g.get("name") or ""
        if not (n.startswith("box") and n[3:].isdigit()):
            continue
        size = [float(v) for v in g.get("size").split()]
        pos = [float(v) for v in g.get("pos").split()]
        boxes.append((n, pos, size))
    if not boxes:
        raise ValueError(f"{src} holds no 'box<N>' geoms; there is no four-box lane to record")
    boxes.sort(key=lambda b: b[1][0])
    native_spacing = (boxes[1][1][0] - boxes[0][1][0]) if len(boxes) > 1 else 0.0
    if spacing > 0:
        x0 = boxes[0][1][0]
        boxes = [(n, [x0 + i * float(spacing), pos[1], pos[2]], size)
                 for i, (n, pos, size) in enumerate(boxes)]

    geoms = [gcts._box_geom(f"lane_{n}", pos[0], 0.0, pos[2],
                            2 * size[0], float(width) if width > 0 else 2 * size[1],
                            2 * size[2], 0.0)
             for n, pos, size in boxes]
    first_face = boxes[0][1][0] - boxes[0][2][0]
    last_face = boxes[-1][1][0] + boxes[-1][2][0]
    top = max(b[1][2] + b[2][2] for b in boxes)
    target_x = last_face + 2.5
    heights = ", ".join(f"{b[1][2] + b[2][2]:.2f}" for b in boxes)
    native_w = 2 * boxes[0][2][1]
    lane_w = float(width) if width > 0 else native_w
    return terrains.Course(
        name=name, klass="box", variant=0, geoms=tuple(geoms), floor_z=0.0,
        target_xy=(target_x, 0.0), course_length_m=target_x, flight_height_m=top,
        obstacle_x_m=first_face, speeds=(float(speed),),
        note=(f"the four-box ladder of terrain_scene (tops {heights} m), moved from y=-6.5 "
              f"to the spawn lane; first face at x={first_face:.2f} m; lateral width "
              + (f"{lane_w:.2f} m, WIDENED from the scene's {native_w:.2f} m so a "
                 "lane-keeping failure cannot stand in for a climbing one"
                 if lane_w > native_w + 1e-6 else f"{lane_w:.2f} m, the scene's own")
              + (f"; spacing RESPACED to {spacing:.2f} m from the scene's "
                 f"{native_spacing:.2f} m" if spacing > 0 else "")),
        params={"source_scene": src, "n_boxes": len(boxes),
                "lane_width_m": lane_w, "native_width_m": native_w,
                "spacing_m": float(spacing) if spacing > 0 else native_spacing,
                "native_spacing_m": native_spacing},
        # Bounded by the SPEED rather than by mjeval's 0.5 m/s default, which would keep an
        # arriving robot walking for another 20 s past the last box with nothing to show.
        # 3x the nominal crossing time and not tighter: the pair gives up most of the
        # commanded speed at each box -- 0.3-0.5 m/s while climbing against 1.45 on the flat,
        # which is the whole point of the clip -- so a budget sized on the command alone ends
        # the episode two boxes short.  Measured at 1.5 m/s: 2.2x ran out with the robot
        # still on the last box.  The tail is cut at render time, not here.
        budget_s_override=float(terrains.CMD_HOLD_S + 4.0 + 3.0 * target_x / max(speed, 0.2)),
    )


#: Path of the in-distribution sim2sim playground, the source of every ``terrain_scene_*`` lane.
_PLAYGROUND = os.path.join(_HERE, "robots", "g1", "g1_perceptive_terrain_scene.xml")


def _scene_lane_geoms(prefix: str, width: float = 0.0) -> tuple[list[str], list[dict]]:
    """Lift one named lane out of ``terrain_scene`` and re-emit it on the spawn centreline.

    The geom ATTRIBUTES are copied, not rebuilt: group, ``contype``/``conaffinity`` and the
    material come across untouched, so a lane whose walls are visual-only in the playground
    stays visual-only here.  Only the name (prefixed), the lateral position (moved to ``y=0``)
    and optionally the lateral half-extent are rewritten.

    Args:
        prefix: Lane name in the scene -- ``center``, ``indoor``, ``outdoor``, ``slow_tread``
            or ``tall``.  Matched as ``<prefix>_``.
        width: Full lateral extent to force on every geom (m); ``0`` keeps the scene's own.

    Returns:
        ``(geom lines, [{name, pos, size, group}, ...])``, in the scene's own order.

    Raises:
        ValueError: If the scene holds no geom with that prefix.
    """
    root = ET.parse(_PLAYGROUND).getroot()
    picked = [g for g in root.findall("./worldbody/geom")
              if (g.get("name") or "").startswith(f"{prefix}_")]
    if not picked:
        names = sorted({(g.get("name") or "").split("_")[0]
                        for g in root.findall("./worldbody/geom")})
        raise ValueError(f"{_PLAYGROUND} has no lane {prefix!r}; it has {names}")
    lane_y = float(picked[0].get("pos").split()[1])
    lines, meta = [], []
    for g in picked:
        a = dict(g.attrib)
        pos = [float(v) for v in a["pos"].split()]
        size = [float(v) for v in a["size"].split()]
        pos[1] -= lane_y
        if width > 0:
            size[1] = width / 2.0
        a["name"] = f"lane_{a['name']}"
        a["pos"] = " ".join(f"{v:.6f}" for v in pos)
        a["size"] = " ".join(f"{v:.6f}" for v in size)
        head = " ".join(f'{k}="{a[k]}"' for k in ("name", "type", "size", "pos") if k in a)
        tail = " ".join(f'{k}="{v}"' for k, v in a.items()
                        if k not in ("name", "type", "size", "pos"))
        lines.append(f"        <geom {head}{' ' + tail if tail else ''}/>")
        meta.append({"name": g.get("name"), "pos": pos, "size": size,
                     "group": int(a.get("group", 2)),
                     "solid": a.get("contype", "1") != "0"})
    return lines, meta


def terrain_scene_stairs(speed: float, name: str, width: float = 0.0, spacing: float = 0.0, *,
                         lane: str = "tall", goal: str = "platform") -> terrains.Course:
    """One staircase lane of ``terrain_scene``, as an mjeval course.

    The playground's five flights are the in-distribution geometry this stack is benchmarked
    on, and ``tall`` -- 20 risers to +1.40 m, then 20 back down -- is the long one.  Geometry
    is lifted out of the committed scene by :func:`_scene_lane_geoms`, so the treads here and
    the treads a joystick run sees are the same objects.

    **The whole lane is emitted even for an ``up`` shot.**  Dropping the descent would leave
    the top platform a cliff and the episode would walk off it; keeping it and putting the
    GOAL on the platform means the footage is cut at the top (``stop_x_pad``) while the robot
    is still standing on ground the scene actually has.

    Args:
        speed: Commanded forward speed (m/s); sets the time budget.
        name: Scene name for the emitted XML.
        width: Full lateral extent to force on every geom (m); ``0`` keeps the flight's own.
        spacing: Not meaningful for a staircase -- respacing treads would change the rise/run
            the flight is defined by -- and refused rather than ignored.
        lane: Which flight; see :func:`_scene_lane_geoms`.
        goal: ``"platform"`` puts the goal on the top platform (the ascent); ``"end"`` puts it
            2.5 m past the last geom (the whole up-and-over).

    Returns:
        The course.

    Raises:
        ValueError: On a non-zero ``spacing``, an unknown ``goal``, or a lane with no steps.
    """
    if spacing:
        raise ValueError("terrain_scene_stairs takes no --lane_spacing: a flight IS its "
                         "rise and run, and respacing the treads would author a different "
                         "staircase rather than record this one.")
    if goal not in ("platform", "end"):
        raise ValueError(f"goal must be 'platform' or 'end', got {goal!r}")
    lines, meta = _scene_lane_geoms(lane, width)
    solid = [g for g in meta if g["solid"]]
    steps = [g for g in solid if "_step" in g["name"]]
    plat = next((g for g in solid if g["name"].endswith("_platform")), None)
    if not steps or plat is None:
        raise ValueError(f"lane {lane!r} has no '_step'/'_platform' geoms to aim at")
    first_face = min(g["pos"][0] - g["size"][0] for g in steps)
    top = max(g["pos"][2] + g["size"][2] for g in solid)
    if goal == "platform":
        target_x = float(plat["pos"][0])
        flight = top
    else:
        target_x = max(g["pos"][0] + g["size"][0] for g in solid) + 2.5
        flight = 0.0                      # up and back down: the gate would be meaningless
    lane_w = 2.0 * max(g["size"][1] for g in steps)
    note = (f"terrain_scene's {lane!r} flight: {len(steps)} risers to +{top:.2f} m, first "
            f"face at x={first_face:.2f} m, lateral width {lane_w:.2f} m"
            + (", WIDENED from the scene's own" if width > 0 else ", the scene's own")
            + f"; goal on the {'top platform' if goal == 'platform' else 'far floor'}")
    return terrains.Course(
        name=name, klass="stairs_up", variant=0, geoms=tuple(lines), floor_z=0.0,
        target_xy=(target_x, 0.0), course_length_m=target_x, flight_height_m=flight,
        obstacle_x_m=first_face, speeds=(float(speed),), note=note,
        params={"source_scene": _PLAYGROUND, "lane": lane, "goal": goal,
                "n_risers": len(steps), "top_m": top, "lane_width_m": lane_w},
        # 4x the nominal crossing time: a 20-riser flight is climbed at well under the
        # commanded speed, and the four-box lane already showed what a budget sized on the
        # command alone does -- it ends the episode part-way up.
        budget_s_override=float(terrains.CMD_HOLD_S + 4.0
                                + 4.0 * target_x / max(speed, 0.2)),
    )


def _stairs_lane(lane: str, goal: str):
    """A :data:`LANES` builder bound to one flight of the playground.

    Args:
        lane: Lane prefix in ``terrain_scene``.
        goal: ``"platform"`` or ``"end"``; see :func:`terrain_scene_stairs`.

    Returns:
        A ``(speed, name, width, spacing) -> Course`` callable.
    """
    def build(speed: float, name: str, width: float = 0.0,
              spacing: float = 0.0) -> terrains.Course:
        """Build this flight's course."""
        return terrain_scene_stairs(speed, name, width, spacing, lane=lane, goal=goal)
    return build


#: Lane builders, by name: ``(speed, scene_name, width, spacing) -> Course``.
LANES = {
    "terrain_scene_boxes": terrain_scene_boxes,
    "terrain_scene_tall_stairs_up": _stairs_lane("tall", "platform"),
    "terrain_scene_tall_stairs_over": _stairs_lane("tall", "end"),
}


# =============================================================================================
# Rolling
# =============================================================================================

def _sha256(path: str) -> str:
    """First 8 hex digits of a file's sha256, for the manifest."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _abs(path: str) -> str:
    """Resolve a repo-relative path against THIS checkout."""
    return path if os.path.isabs(path) else os.path.join(_REPO, path)


def _shot_dir(shot: Shot) -> str:
    """Where a shot's trace, manifest and mp4s live."""
    return os.path.join(VIDEO_ROOT, shot.name)


def _load_probe(shot: Shot) -> dict:
    """The probe manifest behind a ``probe`` shot, with its arm and seed checked."""
    path = os.path.join(_abs(shot.probe), "probe.json")
    if not os.path.isfile(path):
        raise SystemExit(f"shot {shot.name!r}: no probe manifest at {path}")
    with open(path, encoding="utf-8") as fh:
        pm = json.load(fh)
    if shot.arm not in (pm.get("arms") or {}):
        raise SystemExit(f"shot {shot.name!r}: probe has no arm {shot.arm!r}; it has "
                         f"{sorted(pm['arms'])}")
    if not any(int(r["seed"]) == shot.seed for r in pm["arms"][shot.arm]["rows"]):
        raise SystemExit(f"shot {shot.name!r}: arm {shot.arm!r} has no seed {shot.seed}")
    return pm


def _probe_row(pm: dict, arm: str, seed: int) -> dict:
    """The recorded table row for one trial."""
    return next(r for r in pm["arms"][arm]["rows"] if int(r["seed"]) == int(seed))


def _probe_label(shot: Shot) -> str:
    """The human mode label for this trial, or ``""`` if the probe was never labelled."""
    path = os.path.join(_abs(shot.probe), "labels.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        labels = json.load(fh)
    v = labels.get(f"{shot.arm}:{int(shot.seed)}")
    return "" if v is None else str(v["label"] if isinstance(v, dict) else v)


@dataclass
class _Loaded:
    """One shot's loaded pair, rig and resolved run parameters.

    Built once and reused across seeds: loading the two ONNX graphs and compiling the scene
    is most of the wall clock of a short episode, and :func:`scan_lane_seeds` rolls dozens.
    """

    policy: object
    fm_lag: object
    spawn: object
    rig: object
    cfg: object
    course: object
    speed: float
    steer_kp: float
    stop_on_goal: bool
    robot_name: str
    tracker: str
    fm: str
    provenance: dict


def _load_shot(shot: Shot, args) -> _Loaded:
    """Resolve a shot's course and artifacts and load the pair.

    Args:
        shot: The shot.
        args: Parsed command line.

    Returns:
        Everything :func:`_run_seed` needs.

    Raises:
        SystemExit: On an unknown shot kind or a missing probe.
    """
    from mjeval import runner as mjrunner                              # noqa: PLC0415

    out = _shot_dir(shot)
    os.makedirs(out, exist_ok=True)
    scene = f"video_{shot.name}"
    providers = tuple(x.strip() for x in args.providers.split(",") if x.strip())

    if shot.kind == "probe":
        pm = _load_probe(shot)
        row = _probe_row(pm, shot.arm, shot.seed)
        tracker = pm["tracker_onnx"]
        fm = pm["arms"][shot.arm]["fm_onnx"]
        speed = float(pm["speed"])
        steer_kp = float(pm.get("steer_kp", 0.0))
        course = terrains.isaac_tile_course(
            pm["clip"], int(pm["terrain_id"]), speeds=(speed,), klass=pm["klass"], name=scene,
            spawn_from="route", obstacle_from=str(pm.get("obstacle_from", "raised")),
            goal_from="route")
        stop_on_goal = False           # what mode_probe recorded under; see its stage_run
        robot_name = str(pm.get("robot_name", "g1_perceptive"))
        provenance = {"probe": shot.probe, "arm": shot.arm, "seed": shot.seed,
                      "clip": pm["clip"], "terrain_id": int(pm["terrain_id"]),
                      "expected_mode": pm.get("expected_mode", ""),
                      "human_label": _probe_label(shot), "probe_row": row}
    elif shot.kind == "lane":
        tracker, fm = _abs(shot.tracker_onnx), _abs(shot.fm_onnx)
        speed = float(args.speed if args.speed else shot.speed)
        steer_kp = float(shot.steer_kp if args.steer_kp is None else args.steer_kp)
        course = LANES[shot.lane](speed, scene,
                                  args.lane_width or shot.lane_width,
                                  args.lane_spacing or shot.lane_spacing)
        stop_on_goal = False
        robot_name = "g1_perceptive"
        provenance = {"lane": shot.lane, "speed": speed, "steer_kp": steer_kp,
                      "note": course.note,
                      "params": {k: str(v) for k, v in course.params.items()}}
    else:
        raise SystemExit(f"shot {shot.name!r}: unknown kind {shot.kind!r}")

    cfg = mjrunner.ArmRun(
        arm=0, label=f"video_{shot.name}", tracker_onnx=tracker, fm_onnx=fm, out_dir=out,
        robot_name=robot_name, providers=providers,
        fall_height=float(args.fall_height), fall_tilt_deg=float(args.fall_tilt_deg),
        # zero_noise False + a per-trial RNG is exactly what mode_probe rolled under, and it
        # is what makes "the same seed in both arms" a thing that can be said at all.
        steer_kp=steer_kp, zero_noise=False, fm_seed=0)
    policy, fm_lag, spawn = mjrunner.load_arm(cfg)
    rig = mjrunner.CourseRig(course, policy, cfg, spawn)
    return _Loaded(policy, fm_lag, spawn, rig, cfg, course, speed, steer_kp, stop_on_goal,
                   robot_name, tracker, fm, provenance)


def _run_seed(ld: _Loaded, seed: int) -> tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    """Roll one episode at one flow-noise seed.

    The seeding is ``mode_probe``'s, verbatim: rebind the generator's RNG, then clear the
    episode state.  The ONE thing that differs between two calls is the draw.

    Args:
        ld: The loaded shot.
        seed: Flow-noise seed, which for a probe shot is also the trial index.

    Returns:
        ``(verdict, failure, qpos, time, trace)``.
    """
    from mjeval import runner as mjrunner                              # noqa: PLC0415
    from vel_tracking_sweep import TraceRecorder                       # noqa: PLC0415

    frames: list[tuple[float, np.ndarray]] = []

    def record(robot, _policy) -> None:
        """Capture one control step's pose.  Physics is untouched."""
        frames.append((float(robot.mj_data.time), robot.mj_data.qpos.copy()))

    trace = TraceRecorder(ld.policy.base_frame_name)
    ld.policy._rng = np.random.default_rng(int(seed))
    mjrunner._reset_episode_state(ld.policy, ld.fm_lag)
    cmd = np.array([ld.speed, 0.0, 0.0], dtype=np.float64)
    verdict, failure, _acc = mjrunner.run_episode(
        ld.rig, ld.policy, ld.fm_lag, ld.spawn, cmd, None, recorder=record, probe=trace,
        steer_kp=ld.cfg.steer_kp, cfg_hold=ld.cfg.cmd_hold_s, stop_on_goal=ld.stop_on_goal)
    return (verdict, failure,
            np.array([f[1] for f in frames], dtype=np.float32),
            np.array([f[0] for f in frames], dtype=np.float32),
            trace.array())


#: Fraction of a feature's height the root must gain, while downrange of that feature, for
#: the episode to have gone OVER it rather than around it.  0.6 and not 1.0 because the
#: pelvis sags into a climb; measured on this lane, a clean box crossing puts the root at
#: 0.79 + h to within a couple of centimetres and a walk-past leaves it at 0.79 flat.
_CLIMB_FRAC: float = 0.6


def feature_climbs(model, qpos: np.ndarray) -> list[dict]:
    """Per terrain feature, whether the root went OVER it.

    A four-box lane is 2.59 m wide on an infinite floor, and an unsteered-enough episode
    walks BESIDE a box rather than onto it -- which looks like a traverse in the ``x_max``
    column and is the opposite of what a clip about crossing boxes should show.  This is the
    column that tells them apart.

    Args:
        model: The compiled scene (physics or render copy -- same geometry).
        qpos: ``(steps, nq)`` recorded poses.

    Returns:
        One dict per feature, ascending in x: its span, its top, and whether it was climbed.
    """
    floor_z = vr.scene_floor_z(model)
    walls = set(vr.wall_geoms(model))
    out = []
    for lo, hi in vr.terrain_feature_spans(model):
        top = floor_z
        for i in vr.world_boxes(model):
            if i in walls:
                continue
            cx, sx = float(model.geom_pos[i][0]), float(model.geom_size[i][0])
            if cx - sx >= lo - 1e-6 and cx + sx <= hi + 1e-6:
                top = max(top, float(model.geom_pos[i][2] + model.geom_size[i][2]))
        over = (qpos[:, 0] >= lo) & (qpos[:, 0] <= hi)
        z = float(qpos[over, 2].max()) if over.any() else float("nan")
        gate = floor_z + terrains.SPAWN_BASE_Z + _CLIMB_FRAC * (top - floor_z)
        out.append({"x_lo": float(lo), "x_hi": float(hi), "top_m": float(top),
                    "max_root_z_m": z, "reached": bool(over.any()),
                    "climbed": bool(over.any() and z >= gate)})
    return out


def velocity_summary(trace: np.ndarray, columns: tuple[str, ...],
                     spans: list[tuple[float, float]], hold_s: float,
                     margin_m: float = 1.5) -> dict:
    """Commanded-vs-achieved forward speed, split ON and OFF the terrain features.

    The number the velocity panel is a picture of, so the clip's own manifest can be checked
    against the claim it is used to make rather than the claim being read off the picture.
    The opening standing hold is excluded: a zero command met by a standing robot is a
    near-perfect tracking number that says nothing, which is the same exclusion
    ``accumulate.fold`` makes.

    Args:
        trace: ``(steps, len(columns))`` velocity trace.
        columns: Its column names.
        spans: Downrange feature intervals, from :func:`video_render.terrain_feature_spans`.
        hold_s: Seconds of opening hold to drop.
        margin_m: How far from every feature the ``clear`` regime starts.  ``off`` is
            everything not literally over a feature, which around a close-spaced ladder is
            almost all deceleration and recovery and is not a cruising number.

    Returns:
        Per-regime mean achieved ``vx``, mean commanded ``vx`` and RMS error, plus the share
        of scored time spent over a feature.

        **``clear`` is split, and the split is the whole point.**  ``clear_between`` is the
        part of a GAP the robot gets to use; ``clear_outside`` is the run-in and the run-out,
        which are open floor and much easier.  Pooling them reads as "the pair holds the
        command between features" when it may have no samples between features at all: on
        terrain_scene's own 3 m spacing with a 0.46 m box, ``margin_m = 1.5`` leaves the
        interior gaps EMPTY, and the pooled 1.45 m/s was 366 samples of the run-out at 1.61
        against 87 of the run-in at 0.74.
    """
    c = {n: i for i, n in enumerate(columns)}
    if not trace.size:
        return {}
    t = trace[:, c["t"]] - trace[0, c["t"]]
    x, vx, cmd = trace[:, c["x"]], trace[:, c["vx"]], trace[:, c["cmd_vx"]]
    scored = t >= hold_s
    on = np.zeros_like(scored)
    near = np.zeros_like(scored)
    for lo, hi in spans:
        on |= (x >= lo) & (x <= hi)
        near |= (x >= lo - margin_m) & (x <= hi + margin_m)

    def block(m):
        """Mean achieved, mean commanded and RMS error over a boolean mask."""
        if not m.any():
            return {"n": 0}
        return {"n": int(m.sum()), "mean_vx": float(vx[m].mean()),
                "mean_cmd_vx": float(cmd[m].mean()),
                "rms_err_vx": float(np.sqrt(np.mean((vx[m] - cmd[m]) ** 2)))}

    clear = scored & ~near
    interior = (np.ones_like(clear) if not spans
                else (x > spans[0][1]) & (x < spans[-1][0]))
    return {"all": block(scored), "on_feature": block(scored & on),
            "off_feature": block(scored & ~on), "clear": block(clear),
            "clear_between": block(clear & interior),
            "clear_outside": block(clear & ~interior),
            "feature_time_frac": float((scored & on).sum() / max(int(scored.sum()), 1)),
            "clear_margin_m": float(margin_m), "hold_s": float(hold_s)}


def roll_shot(shot: Shot, args) -> dict:
    """Run the episode behind one shot and write its trace.

    For a ``probe`` shot this re-rolls a trial ``mode_probe`` already ran, with the same
    course, arm, seed and steering, and REFUSES to write anything if the re-roll does not
    reproduce the stored row -- verdict, failure code, step count and furthest x.  The point
    of the clip is that it IS the labelled episode.

    Args:
        shot: The shot to roll.
        args: Parsed command line.

    Returns:
        The shot manifest, already written to disk.

    Raises:
        SystemExit: On a failed reproduction check, or a missing artifact.
    """
    from vel_tracking_sweep import TRACE_COLUMNS                       # noqa: PLC0415

    out = _shot_dir(shot)
    ld = _load_shot(shot, args)
    seed = int(shot.seed) if shot.seed >= 0 else 0
    if shot.kind == "lane" and args.seed is not None:
        seed = int(args.seed)
    t0 = time.time()
    verdict, failure, q, t, tr = _run_seed(ld, seed)
    course, provenance = ld.course, ld.provenance
    max_x = float(q[:, 0].max()) if q.size else 0.0
    print(f"[video] {shot.name}: {verdict}{'/' + failure if failure else ''}  "
          f"{q.shape[0]} steps  x_max {max_x:.2f} m  ({time.time() - t0:.0f} s)")

    if shot.kind == "probe":
        row = provenance["probe_row"]
        drift = []
        if str(row["verdict"]) != str(verdict):
            drift.append(f"verdict {row['verdict']!r} -> {verdict!r}")
        if str(row["failure"]) != str(failure):
            drift.append(f"failure {row['failure']!r} -> {failure!r}")
        if int(row["steps"]) != int(q.shape[0]):
            drift.append(f"steps {row['steps']} -> {q.shape[0]}")
        if abs(float(row["max_x_m"]) - max_x) > 1e-3:
            drift.append(f"max_x_m {row['max_x_m']:.4f} -> {max_x:.4f}")
        if drift:
            raise SystemExit(
                f"REFUSING to write {shot.name}: the re-roll did not reproduce the probe's "
                f"own row ({'; '.join(drift)}). The clip would then not be the episode that "
                "was labelled. Something in the roll path has changed since the probe ran.")
        print(f"[video]   reproduces {shot.probe}:{shot.arm}:{shot.seed} exactly"
              + (f"  (human label: {provenance['human_label']!r})"
                 if provenance["human_label"] else ""))

    np.savez_compressed(os.path.join(out, "trace.npz"), qpos=q, time=t, trace=tr)
    man = {
        "shot": dataclasses.asdict(shot),
        "story": shot.story,
        "kind": shot.kind,
        "provenance": provenance,
        "artifacts": {"tracker_onnx": ld.tracker, "tracker_sha256": _sha256(ld.tracker),
                      "fm_onnx": ld.fm, "fm_sha256": _sha256(ld.fm)},
        "course": {"name": course.name, "klass": course.klass,
                   "scene_xml": terrains.scene_path(course, ld.robot_name),
                   "obstacle_x_m": float(course.obstacle_x_m),
                   "goal_x_m": float(course.target_xy[0]),
                   "goal_half_width_m": float(course.goal_half_width_m),
                   "floor_z": float(course.floor_z), "budget_s": float(course.budget_s),
                   "note": course.note},
        "roll": {"verdict": verdict, "failure": failure, "steps": int(q.shape[0]),
                 "max_x_m": max_x, "speed": ld.speed, "steer_kp": ld.steer_kp,
                 "stop_on_goal": ld.stop_on_goal, "seed": seed,
                 "max_abs_y_m": float(np.abs(q[:, 1]).max()) if q.size else 0.0,
                 "features": feature_climbs(ld.rig.robot.mj_model, q),
                 "velocity": velocity_summary(
                     tr, TRACE_COLUMNS,
                     vr.terrain_feature_spans(ld.rig.robot.mj_model),
                     float(ld.cfg.cmd_hold_s)),
                 "dt": float(ld.policy.get_dt()),
                 "fm_depth_lag_steps": int(ld.fm_lag.steps),
                 "trace_columns": list(TRACE_COLUMNS)},
    }
    with open(os.path.join(out, "shot.json"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    del ld
    return man


def scan_lane_seeds(shot: Shot, args, n: int) -> str:
    """Roll ``n`` flow-noise draws of a LANE shot and report how far each got.

    A held command on a four-box ladder is not a sure thing -- the generator's own sampling
    variance decides roughly one outcome in five on much easier terrain -- so the clip is a
    CHOSEN draw, and choosing one silently would be the dishonest half of that.  This writes
    the whole scan next to the shot, so the pinned seed can be quoted with the rate: "draw
    ``k``, and ``m`` of ``n`` crossed all four".

    Args:
        shot: A ``lane`` shot.
        args: Parsed command line.
        n: How many seeds, starting at 0.

    Returns:
        Path of the scan JSON.

    Raises:
        SystemExit: If asked for a probe shot, whose seed is not ours to choose.
    """
    if shot.kind != "lane":
        raise SystemExit(f"--scan_seeds is for lane shots; {shot.name!r} is a probe shot, and "
                         "its seed is the probe's trial index, not a free choice.")
    ld = _load_shot(shot, args)
    goal_x = float(ld.course.target_xy[0])
    model = ld.rig.robot.mj_model
    rows = []
    for seed in range(int(n)):
        verdict, failure, q, _t, _tr = _run_seed(ld, seed)
        max_x = float(q[:, 0].max()) if q.size else 0.0
        off = float(np.abs(q[:, 1]).max()) if q.size else 0.0
        feats = feature_climbs(model, q)
        bits = "".join("1" if f["climbed"] else "0" for f in feats)
        rows.append({"seed": seed, "verdict": verdict, "failure": failure,
                     "steps": int(q.shape[0]), "max_x_m": max_x, "max_abs_y_m": off,
                     "climbed": bits, "all_climbed": bool(bits and set(bits) == {"1"}),
                     "reached_goal_x": bool(max_x >= goal_x), "features": feats})
        print(f"[scan] seed {seed:>3}  {verdict:<9}{failure:<7} x_max {max_x:6.2f} m  "
              f"|y|max {off:5.2f} m  climbed {bits}  {q.shape[0]:4d} steps", flush=True)
    ok = [r["seed"] for r in rows if r["all_climbed"] and r["reached_goal_x"]]
    print(f"[scan] {len(ok)}/{len(rows)} draws climbed EVERY feature and reached "
          f"x={goal_x:.2f} m: {ok}")
    path = os.path.join(_shot_dir(shot), "seed_scan.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"shot": shot.name, "speed": ld.speed, "steer_kp": ld.steer_kp,
                   "goal_x_m": goal_x, "n": int(n), "all_climbed": ok, "rows": rows},
                  fh, indent=2)
    del ld
    return path


# =============================================================================================
# Rendering
# =============================================================================================

def render_shot(shot: Shot, args) -> list[str]:
    """Render one already-rolled shot, once per requested velocity-panel placement.

    Args:
        shot: The shot.
        args: Parsed command line; every render knob overrides the shot's default.

    Returns:
        The mp4 paths written.

    Raises:
        SystemExit: If the shot has not been rolled.
    """
    out = _shot_dir(shot)
    man_path = os.path.join(out, "shot.json")
    if not os.path.isfile(man_path):
        raise SystemExit(f"shot {shot.name!r} has not been rolled; run --stage roll first "
                         f"(nothing at {man_path})")
    with open(man_path, encoding="utf-8") as fh:
        man = json.load(fh)
    z = np.load(os.path.join(out, "trace.npz"))
    qpos, t, tr = z["qpos"], z["time"], z["trace"]

    # Cut the tail where the course ends.  These tiles author their own footprint and get no
    # infinite floor, so an episode that arrives keeps walking and drops off the edge of the
    # world; that is the recording running past the course, not a result.
    pad = args.stop_x_pad if args.stop_x_pad is not None else shot.stop_x_pad
    if pad is not None:
        stop_x = float(man["course"]["goal_x_m"]) + float(pad)
        past = np.nonzero(qpos[:, 0] >= stop_x)[0]
        if past.size:
            k = int(past[0]) + 1
            print(f"[video] {shot.name}: cut at x={stop_x:.2f} m "
                  f"({k}/{qpos.shape[0]} steps kept)")
            qpos, t = qpos[:k], t[:k]
            tr = tr[:k] if tr.size else tr

    placements = (("bottom", "inset") if args.vel_plot == "both"
                  else (args.vel_plot or shot.vel_plot,))
    channels = tuple(c.strip() for c in args.vel_channels.split(",") if c.strip()) \
        if args.vel_channels else shot.vel_channels
    caption = ()
    if args.caption:
        lab = man["provenance"].get("human_label", "")
        caption = tuple(x for x in (
            f"{man['provenance'].get('arm', shot.kind)}   seed {shot.seed}"
            if shot.kind == "probe" else f"{shot.lane}   {man['roll']['speed']:.2f} m/s",
            f"first mode (human label): {lab}" if lab else "") if x)

    written = []
    for place in placements:
        spec = vr.RenderSpec(
            width=args.width, height=args.height, fps=args.fps,
            rt=float(args.rt if args.rt else shot.rt),
            camera=args.camera or shot.camera,
            # CLI beats the shot, the shot beats the preset.
            cam_overrides={k: (cli if cli is not None else own) for k, cli, own in (
                ("azimuth", args.azimuth, shot.cam_azimuth),
                ("elevation", args.elevation, shot.cam_elevation),
                ("distance", args.distance, shot.cam_distance),
                ("height", args.cam_height, shot.cam_height))},
            theme=args.theme or shot.theme,
            terrain_shade=args.terrain_shade or shot.terrain_shade,
            wall_alpha=(shot.wall_alpha if args.wall_alpha is None else args.wall_alpha),
            geom_groups=tuple(int(g) for g in args.geom_groups.split(",") if g.strip()),
            vel_plot=place, vel_channels=channels,
            vel_smooth_s=float(args.vel_smooth), span_pad_m=float(args.span_pad),
            caption=caption,
            shadows=not args.no_shadows,
            hold_end_s=(shot.hold_end_s if args.hold_end_s is None else args.hold_end_s),
        )
        suffix = "" if place == "off" else ("_vel" if place == "bottom" else "_velinset")
        mp4 = os.path.join(out, f"{shot.name}{suffix}.mp4")
        info = vr.render_trace(man["course"]["scene_xml"], qpos, t, tr, mp4, spec)
        info["spec"] = {k: (list(v) if isinstance(v, tuple) else v)
                        for k, v in dataclasses.asdict(spec).items()}
        man.setdefault("renders", {})[os.path.basename(mp4)] = info
        print(f"[video] {shot.name}: {info['frames']} frames, {info['seconds']:.1f} s "
              f"({info['sim_seconds']:.1f} s of sim at {spec.rt:g}x) -> {mp4}")
        written.append(mp4)
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2)
    return written


def write_index() -> str:
    """Rewrite ``logs/video/INDEX.md`` from the shot manifests on disk."""
    rows = []
    for shot in SHOTS:
        man_path = os.path.join(_shot_dir(shot), "shot.json")
        if not os.path.isfile(man_path):
            rows.append((shot, None))
            continue
        with open(man_path, encoding="utf-8") as fh:
            rows.append((shot, json.load(fh)))
    os.makedirs(VIDEO_ROOT, exist_ok=True)
    path = os.path.join(VIDEO_ROOT, "INDEX.md")
    lines = ["# Recorded videos", "",
             "Written by `transfer/sim/record_videos.py`. Re-render any of these with",
             "`python record_videos.py --stage render --shots <name>`; the roll is cached, so",
             "only the look changes and the episode stays the one that was measured.", ""]
    for shot, man in rows:
        lines.append(f"## `{shot.name}`")
        lines.append("")
        lines.append(shot.story)
        lines.append("")
        if man is None:
            lines.append("_not rolled yet_")
            lines.append("")
            continue
        p, r = man["provenance"], man["roll"]
        if shot.kind == "probe":
            lines.append(f"- probe `{p['probe']}` arm `{p['arm']}` seed {p['seed']}"
                         + (f", human first-mode label **{p['human_label']}**"
                            if p.get("human_label") else ""))
            lines.append(f"- expected first mode: `{p.get('expected_mode', '?')}`")
        else:
            lines.append(f"- lane `{p['lane']}`, {p['speed']:.2f} m/s")
        lines.append(f"- outcome: {r['verdict']}"
                     + (f"/{r['failure']}" if r["failure"] else "")
                     + f", {r['steps']} control steps, furthest x {r['max_x_m']:.2f} m")
        feats = r.get("features") or []
        if feats:
            bits = "".join("1" if f["climbed"] else "0" for f in feats)
            lines.append(f"- features climbed: {bits.count('1')} of {len(feats)}"
                         + (f" (near to far: {bits})" if len(feats) > 1 else ""))
        v = r.get("velocity") or {}
        # Only meaningful where there ARE gaps: a staircase merges into ONE feature, and the
        # note then claims something about a gap the course does not have.
        if (len(feats) > 1 and v.get("all", {}).get("n")
                and not v.get("clear_between", {}).get("n")):
            lines.append(
                f"- NOTE: the gaps between features are shorter than "
                f"2 x {v['clear_margin_m']:.1f} m, so there is no sample in one that is "
                "clear of a feature -- every between-features number here is approach or "
                "recovery, not cruising.")
        if v.get("all", {}).get("n"):
            lines.append(
                f"- forward speed, commanded {v['all']['mean_cmd_vx']:.2f} m/s: achieved "
                f"{v['all']['mean_vx']:.2f} m/s overall"
                + (f", {v['off_feature']['mean_vx']:.2f} off a feature"
                   if v.get("off_feature", {}).get("n") else "")
                + (f", {v['on_feature']['mean_vx']:.2f} over one"
                   if v.get("on_feature", {}).get("n") else "")
                + (f", {v['clear_between']['mean_vx']:.2f} in a GAP clear of one by "
                   f"{v['clear_margin_m']:.1f} m"
                   if v.get("clear_between", {}).get("n") else "")
                + (f", {v['clear_outside']['mean_vx']:.2f} on the open floor before and "
                   "after the ladder" if v.get("clear_outside", {}).get("n") else "")
                + f" ({100 * v['feature_time_frac']:.0f}% of scored time over a feature)")
        lines.append(f"- generator `{os.path.basename(man['artifacts']['fm_onnx'])}` "
                     f"(sha {man['artifacts']['fm_sha256']}), tracker sha "
                     f"{man['artifacts']['tracker_sha256']}")
        for mp4, info in (man.get("renders") or {}).items():
            # The manifest accumulates every render ever made of a shot, so an mp4 that has
            # since been deleted would otherwise be listed as if it were there.
            if not os.path.isfile(os.path.join(_shot_dir(shot), mp4)):
                continue
            lines.append(f"- `{mp4}` -- {info['seconds']:.1f} s, "
                         f"{info['spec']['width']}x{info['spec']['height']}, "
                         f"theme `{info['spec']['theme']}`, camera "
                         f"`{info['spec']['camera']}`, velocity panel "
                         f"`{info['spec']['vel_plot']}`")
        lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def stage_list() -> int:
    """Print the registry and what each shot has on disk."""
    print(f"{'shot':<24}{'kind':<7}{'source':<40}{'rolled':<8}{'renders'}")
    for shot in SHOTS:
        man_path = os.path.join(_shot_dir(shot), "shot.json")
        man = None
        if os.path.isfile(man_path):
            with open(man_path, encoding="utf-8") as fh:
                man = json.load(fh)
        src = (f"{os.path.basename(shot.probe)}:{shot.arm}:{shot.seed}"
               if shot.kind == "probe" else f"{shot.lane} @ {shot.speed:g} m/s")
        n = len((man or {}).get("renders") or {})
        print(f"{shot.name:<24}{shot.kind:<7}{src:<40}"
              f"{'yes' if man else 'no':<8}{n}")
        print(f"    {shot.story}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The command line."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="all", choices=("list", "roll", "render", "all"))
    p.add_argument("--shots", default="",
                   help="Comma-separated shot names; default is every shot in the registry.")
    # --- roll ---
    p.add_argument("--providers", default="auto", help="onnxruntime providers.")
    p.add_argument("--fall_height", type=float, default=0.2,
                   help="mode_probe's own default; a probe shot's reproduction check fails "
                        "loudly if this does not match what the probe ran under.")
    p.add_argument("--fall_tilt_deg", type=float, default=70.0)
    p.add_argument("--speed", type=float, default=0.0,
                   help="Override a LANE shot's commanded forward speed (m/s). A probe shot "
                        "takes the probe's own speed and ignores this: changing it would "
                        "roll a different episode from the labelled one. The trained forward "
                        "range runs to about 3.5 m/s (terrains.BOX_SPEEDS).")
    p.add_argument("--lane_width", type=float, default=0.0,
                   help="Full lateral extent of a LANE shot's features (m); 0 keeps the "
                        "source scene's own. See terrain_scene_boxes for the measurement "
                        "that motivates widening: nothing in this stack keeps a lane, and at "
                        "terrain_scene's 2.59 m not one of 40 draws crossed all four boxes.")
    p.add_argument("--lane_spacing", type=float, default=0.0,
                   help="Distance between a LANE shot's feature centres (m); 0 keeps the "
                        "source scene's own. At terrain_scene's 3 m the pair never recovers "
                        "the command between boxes, so a velocity-tracking clip needs more.")
    p.add_argument("--steer_kp", type=float, default=None,
                   help="Override a LANE shot's heading gain (1/s); 0 drives the raw "
                        "constant command and drifts off an open lane. Ignored by a probe "
                        "shot, which takes the probe's own.")
    p.add_argument("--force", action="store_true",
                   help="Re-roll a shot that already has a trace on disk.")
    p.add_argument("--seed", type=int, default=None,
                   help="Override a LANE shot's flow-noise draw. A probe shot's seed IS its "
                        "trial index and is not overridable.")
    p.add_argument("--scan_seeds", type=int, default=0,
                   help="Roll this many draws of a LANE shot and report how far each got, "
                        "writing seed_scan.json and no trace. The clip is a chosen draw and "
                        "this is what lets the choice be stated with its rate.")
    # --- render ---
    p.add_argument("--width", type=int, default=1920,
                   help="3-D view width in px. The themes give the offscreen buffer "
                        "3840x2160, so 4K renders; the frame rate of the render, not "
                        "the file, is what pays for it.")
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rt", type=float, default=0.0,
                   help="Sim seconds per output second; 0 keeps the shot's own (1.0). 0.5 is "
                        "half speed.")
    p.add_argument("--camera", default="", choices=("", *vr.CAMERAS),
                   help=f"Camera preset; one of {sorted(vr.CAMERAS)}.")
    p.add_argument("--azimuth", type=float, default=None,
                   help="Override the preset. 0 looks downrange from behind, 90 across the "
                        "lane from the -y side.")
    p.add_argument("--elevation", type=float, default=None, help="Negative looks down.")
    p.add_argument("--distance", type=float, default=None, help="Camera distance (m).")
    p.add_argument("--cam_height", type=float, default=None,
                   help="Metres above the pelvis the look-at sits.")
    p.add_argument("--theme", default="", choices=("", *vr.THEMES),
                   help=f"Scene look; one of {sorted(vr.THEMES)}. 'asis' is the debug look "
                        "the scenes are authored in.")
    p.add_argument("--terrain_shade", default="",
                   choices=("", "off", "height", "alternate"),
                   help="Recolour the terrain per feature so a flight's treads are legible.")
    p.add_argument("--wall_alpha", type=float, default=None,
                   help="Opacity of the tall thin world geoms, i.e. the stairwell walls. "
                        "Below 1 they are see-through, which is the only way to watch a robot "
                        "inside a 1.75 m corridor from the side.")
    p.add_argument("--vel_plot", default="", choices=("", "off", "bottom", "inset", "both"),
                   help="Commanded-vs-achieved velocity panel. 'both' writes one mp4 per "
                        "placement, so the plotted and unplotted cuts stay in step.")
    p.add_argument("--vel_channels", default="",
                   help="Comma-separated subset of vx,vy,wz for the panel.")
    p.add_argument("--vel_smooth", type=float, default=0.4,
                   help="Moving-average window for the panel's achieved line (s). The pelvis "
                        "velocity swings ~1 m/s inside a single stride, so the raw line is "
                        "honest and unreadable; the raw one stays drawn thin underneath. 0 "
                        "draws the raw line alone.")
    p.add_argument("--span_pad", type=float, default=0.0,
                   help="Widen each shaded feature band by this much at each end (m). 0 is "
                        "exactly 'the root is over the feature'; the pair also gives up "
                        "speed on the APPROACH, so a pad reads better and measures less.")
    p.add_argument("--geom_groups", default="1,2",
                   help="MuJoCo geom groups to draw: 1 is the robot's visual meshes and 2 "
                        "the terrain; 0 is the collision primitives.")
    p.add_argument("--no_shadows", action="store_true")
    p.add_argument("--hold_end_s", type=float, default=None,
                   help="Seconds of freeze on the last frame.")
    p.add_argument("--stop_x_pad", type=float, default=None,
                   help="Cut the footage this far past the course's goal line (m). Pass a "
                        "large number to keep the whole episode, walk off the tile included.")
    p.add_argument("--caption", action="store_true",
                   help="Burn the arm, the seed and the human mode label into the frame.")
    return p


def main() -> int:
    """Roll and/or render the selected shots."""
    args = build_parser().parse_args()
    if args.stage == "list":
        return stage_list()

    want = [s.strip() for s in args.shots.split(",") if s.strip()] or list(SHOT_BY_NAME)
    missing = [n for n in want if n not in SHOT_BY_NAME]
    if missing:
        raise SystemExit(f"no such shot(s): {missing}. Known: {sorted(SHOT_BY_NAME)}")

    for n in want:
        shot = SHOT_BY_NAME[n]
        if args.scan_seeds:
            print(f"[scan] {scan_lane_seeds(shot, args, args.scan_seeds)}")
            continue
        if args.stage in ("roll", "all"):
            have = os.path.isfile(os.path.join(_shot_dir(shot), "trace.npz"))
            if have and not args.force:
                print(f"[video] {shot.name}: already rolled, keeping it (--force to re-roll)")
            else:
                roll_shot(shot, args)
        if args.stage in ("render", "all"):
            render_shot(shot, args)
    print(f"[video] index -> {write_index()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
