# Copyright 2026 Zachary Olkin. All rights reserved.

"""Tests for the fixed stair-tile geometry builder (``box_clip.stair_tile_boxes``).

Locks the shared-tile geometry conventions: uniform descending treads (x-spacing == depth, z drops
by one riser per step), pure tiles carry treads only, transition tiles add a top platform flush with
the highest tread and a bottom floor slab at z=0, and steps stay narrow while platforms are wide (so
the downstream pipeline distinguishes them).
"""

from __future__ import annotations

from motion_gen.motion_matching.box_clip import stair_tile_boxes


def _steps_and_platforms(boxes):
    min_sx = min(b["size"][0] for b in boxes)
    steps = [b for b in boxes if b["size"][0] <= 1.5 * min_sx]
    plats = [b for b in boxes if b["size"][0] > 1.5 * min_sx]
    return steps, plats


def test_pure_tile_is_treads_only():
    boxes = stair_tile_boxes(step_height=0.15, step_depth=0.19, num_steps=6, tread_width=1.0,
                             ttype="terrain")
    steps, plats = _steps_and_platforms(boxes)
    assert len(boxes) == 6 and not plats                     # treads only, no platforms
    for b in boxes:
        assert abs(b["size"][0] - 0.19) < 1e-6               # narrow: size_x == depth
        assert abs(b["size"][1] - 1.0) < 1e-6                # tread width
        assert abs(b["yaw"]) < 1e-9


def test_treads_uniform_and_descending_in_x():
    h, d, N = 0.15, 0.19, 8
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=1.0,
                             ttype="terrain")
    tops = [b["pos"][2] + b["size"][2] / 2.0 for b in boxes]  # tread-top z
    xs = [b["pos"][0] for b in boxes]
    # x increases, z decreases by exactly one riser per step (descending in +x).
    assert xs == sorted(xs)
    for i in range(1, N):
        assert abs((xs[i] - xs[i - 1]) - d) < 1e-5
        assert abs((tops[i - 1] - tops[i]) - h) < 1e-5
    assert abs(max(tops) - N * h) < 1e-5                      # highest tread top = N*h
    assert abs(min(tops) - h) < 1e-5                          # lowest tread top = h (one riser above floor)


def test_transition_tile_adds_flush_top_and_floor_bottom():
    h, d, N = 0.15, 0.19, 5
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=0.9,
                             ttype="transition", top_platform_len=3.0, bottom_platform_len=2.0,
                             nav_width=1.4)
    steps, plats = _steps_and_platforms(boxes)
    assert len(steps) == N and len(plats) == 2
    step_tops = sorted(b["pos"][2] + b["size"][2] / 2.0 for b in steps)
    plat_tops = sorted(b["pos"][2] + b["size"][2] / 2.0 for b in plats)
    assert abs(plat_tops[-1] - N * h) < 1e-5                  # top platform flush with highest tread (N*h)
    assert abs(plat_tops[-1] - step_tops[-1]) < 1e-5
    assert abs(plat_tops[0] - step_tops[0]) < 1e-5           # bottom platform flush with lowest tread
    # platforms carry the nav width, are wide, and are ordered top(low x) -> bottom(high x).
    assert all(abs(p["size"][1] - 1.4) < 1e-6 for p in plats)
    top_plat = max(plats, key=lambda b: b["pos"][2])
    bot_plat = min(plats, key=lambda b: b["pos"][2])
    assert top_plat["pos"][0] < min(b["pos"][0] for b in steps)   # top platform behind the treads
    assert bot_plat["pos"][0] > max(b["pos"][0] for b in steps)   # bottom platform after the treads


def test_ascent_treads_rise_in_x():
    """going_up=True (stairs_up): tread tops INCREASE along +x (mirror of the descent ordering)."""
    h, d, N = 0.15, 0.19, 8
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=1.0,
                             ttype="terrain", going_up=True)
    tops = [b["pos"][2] + b["size"][2] / 2.0 for b in boxes]
    xs = [b["pos"][0] for b in boxes]
    assert xs == sorted(xs)
    for i in range(1, N):
        assert abs((tops[i] - tops[i - 1]) - h) < 1e-5           # z RISES by one riser per step
    assert abs(min(tops) - h) < 1e-5 and abs(max(tops) - N * h) < 1e-5


def test_ascent_transition_platforms_flip():
    """Ascent transition: approach platform (low x) is LOW, recovery (high x) is HIGH -- the flip
    of the descent case, but each still flush with its adjacent tread."""
    h, d, N = 0.15, 0.19, 5
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=0.9,
                             ttype="transition", top_platform_len=3.0, bottom_platform_len=2.0,
                             nav_width=1.4, going_up=True)
    steps, plats = _steps_and_platforms(boxes)
    assert len(steps) == N and len(plats) == 2
    low_x_plat = min(plats, key=lambda b: b["pos"][0])           # approach (low x)
    high_x_plat = max(plats, key=lambda b: b["pos"][0])          # recovery (high x)
    low_x_plat_top = low_x_plat["pos"][2] + low_x_plat["size"][2] / 2.0
    high_x_plat_top = high_x_plat["pos"][2] + high_x_plat["size"][2] / 2.0
    assert abs(low_x_plat_top - h) < 1e-5                        # approach flush with lowest tread (h)
    assert abs(high_x_plat_top - N * h) < 1e-5                   # recovery flush with highest tread (N*h)


def test_descent_unchanged_by_going_up_flag():
    """going_up=False must reproduce the descent geometry exactly (regression guard)."""
    kw = dict(step_height=0.15, step_depth=0.19, num_steps=6, tread_width=0.9,
              ttype="transition", top_platform_len=3.0, bottom_platform_len=2.0, nav_width=1.4)
    assert stair_tile_boxes(**kw) == stair_tile_boxes(**kw, going_up=False)


def test_transition_x_layout_contiguous():
    h, d, N, top, bot = 0.15, 0.19, 4, 3.0, 2.0
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=1.0,
                             ttype="transition", top_platform_len=top, bottom_platform_len=bot)
    xmin = min(b["pos"][0] - b["size"][0] / 2 for b in boxes)
    xmax = max(b["pos"][0] + b["size"][0] / 2 for b in boxes)
    assert abs(xmin - 0.0) < 1e-5                             # starts at tile x=0
    assert abs((xmax - xmin) - (top + N * d + bot)) < 1e-4    # total x = top + stairs + bottom


def test_solid_treads_carry_no_variant_keys():
    """``float_thickness=0`` (default) must add nothing -- prior libraries stay byte-identical."""
    kw = dict(step_height=0.15, step_depth=0.19, num_steps=5, tread_width=1.0, ttype="transition",
              top_platform_len=3.0, bottom_platform_len=2.0)
    assert stair_tile_boxes(**kw) == stair_tile_boxes(**kw, float_thickness=0.0)
    for b in stair_tile_boxes(**kw):
        assert "variant" not in b and "slab_thickness" not in b


def test_floating_tags_treads_only_and_preserves_extents():
    """Floating treads get the overlay variant; platforms stay solid; pos/size are untouched."""
    h, d, N = 0.15, 0.19, 5
    kw = dict(step_height=h, step_depth=d, num_steps=N, tread_width=1.0, ttype="transition",
              top_platform_len=3.0, bottom_platform_len=2.0, nav_width=1.4)
    solid = stair_tile_boxes(**kw)
    floating = stair_tile_boxes(**kw, float_thickness=0.05)
    steps, plats = _steps_and_platforms(floating)
    assert len(steps) == N and len(plats) == 2
    for b in steps:
        assert b["variant"] == "floating"
        assert abs(b["slab_thickness"] - 0.05) < 1e-9
    for b in plats:                                              # ground/top surfaces stay supported
        assert "variant" not in b
    # The solid extents are the contract every placement / footprint / AABB consumer reads.
    for a, b in zip(solid, floating):
        assert a["pos"] == b["pos"] and a["size"] == b["size"]


def test_floating_slab_clamped_to_each_tread_height():
    """A thickness thicker than the lowest tread's own riser must clamp, not overhang below z=0."""
    h, d, N = 0.15, 0.19, 4
    boxes = stair_tile_boxes(step_height=h, step_depth=d, num_steps=N, tread_width=1.0,
                             ttype="terrain", float_thickness=10.0)
    for b in boxes:
        assert b["slab_thickness"] <= b["size"][2] + 1e-9        # never thicker than the tread itself
    assert abs(min(b["slab_thickness"] for b in boxes) - h) < 1e-9   # lowest tread: clamped to one riser
