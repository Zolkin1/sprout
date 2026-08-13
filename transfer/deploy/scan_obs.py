# Copyright 2026 Zachary Olkin. All rights reserved.

"""Height-scan deployment contract: parse and agree (numpy only).

The counterpart of :mod:`.depth_obs` for downward ray-cast grids.  The geometry the
generator was conditioned on is embedded in ``fm.onnx``
(``fm_meta.extero_spec``) by ``mdp.height_scan_specs``, and in ``tracker.onnx``
(``policy_parameters.height_scan_spec``) when the POLICY observes a scan; this module is the
consumer side, shared by the MuJoCo sim2sim cast (``Robot.compute_height_scan``) and any
other off-sim reader.

**Why this exists as its own module.** ``fm_meta.extero_spec`` carries whichever sensor the
checkpoint trained on, and until now every consumer ran it through
:func:`~deploy.depth_obs.parse_depth_spec` unconditionally -- so a scan-conditioned graph was
refused at load for "missing field(s) ['fx', 'fy', ...]", naming camera fields a height
scanner has no reason to carry.  The parse has to be chosen by the contract's own
``extero_source``, which is what :func:`parse_extero_spec` does.

There is no normalisation half here, and that asymmetry is the point: depth is normalised
*inside* ``mdp.depth_image`` (so the deploy must reproduce the affine), while a scan is read
raw from the sensor and the ``fm.onnx`` graph folds in its normalisation.  What a consumer
needs from a scan contract is therefore only *where each ray sampled the ground*.

No torch, no isaaclab.
"""

from __future__ import annotations

from .depth_obs import parse_depth_spec

#: Fields every scanner entry must carry for an off-sim consumer to reproduce the cast.
#:
#: ``sensor_offset`` is in here because it is not geometry: it is the constant
#: ``mdp.height_scan`` subtracts, so a reading cast without it is offset by 0.5 m and still
#: looks entirely plausible.
_REQUIRED_SCAN_FIELDS: tuple[str, ...] = (
    "sensor", "mount_body", "offset_pos", "ray_alignment", "direction",
    "resolution", "size", "ordering", "height", "width", "num_rays", "sensor_offset",
)

#: Contract fields two scanners never have to agree on -- naming, cadence, and the
#: randomisation a privileged sampling is defined by NOT having.  Everything else the two
#: contracts share is geometric and is compared, so a field added to both is checked without
#: touching this module.
_SCAN_NAMING_FIELDS: frozenset[str] = frozenset(
    {"sensor", "update_period", "privileged", "max_distance", "drift_range",
     "ray_cast_drift_range", "clip"}
)

#: The geometry two scan contracts must BOTH state before they can be said to agree.
#:
#: The comparison is over the shared keys, because the two artifacts carry different
#: contracts: ``fm.onnx`` embeds the full ``mdp.height_scan_specs`` dict while
#: ``tracker.onnx`` embeds the six fields ``export_parameters`` emits.  Intersecting without
#: this floor would let a contract that states almost nothing "agree" with anything; the old
#: inline check had the opposite bug, comparing ``fm``'s ``ordering`` against a tracker key
#: that does not exist and so refusing EVERY scan tracker + scan generator pair.
_SCAN_COMPARABLE_FLOOR: frozenset[str] = frozenset(
    {"resolution", "size", "offset_pos", "ray_alignment", "sensor_offset"}
)


def parse_scan_spec(raw, source: str = "spec") -> list[dict]:
    """Validate an embedded height-scanner contract.

    Checks the three things that are silent when wrong rather than merely absent: a cast
    direction other than straight down (the reading is then not ground clearance at all), a
    non-yaw ray alignment (the grid rolls with the torso, so a ray is no longer a fixed point
    in the generator's window frame), and a fold that disagrees with the ray count (which
    scrambles the image a convolutional encoder reads).  These mirror the assertions
    ``mdp.height_scan_specs`` makes on the producing side, so the same defect is caught
    whichever end you are standing at.

    Args:
        raw: The list of per-scanner dicts read from ONNX metadata (or ``None``).
        source: Label for error messages (e.g. ``"fm.onnx fm_meta"``).

    Returns:
        One validated dict per scanner, in the order given.  Empty for ``None``.

    Raises:
        ValueError: If an entry is missing a required field, has a non-positive resolution,
            does not cast straight down, is not yaw-aligned, or folds to a ray count other
            than the one it declares.
    """
    if not raw:
        return []
    out: list[dict] = []
    for i, scan in enumerate(raw):
        missing = [f for f in _REQUIRED_SCAN_FIELDS if f not in scan]
        if missing:
            raise ValueError(f"{source} height scanner {i} is missing field(s) {missing}.")
        if float(scan["resolution"]) <= 0.0:
            raise ValueError(
                f"{source} height scanner {i} has non-positive resolution "
                f"{scan['resolution']}."
            )
        direction = tuple(float(v) for v in scan["direction"])
        if direction != (0.0, 0.0, -1.0):
            raise ValueError(
                f"{source} height scanner {i} casts along {direction}, not straight down; "
                "'pos_z - hit_z - sensor_offset' is only ground clearance for (0, 0, -1)."
            )
        if str(scan["ray_alignment"]) != "yaw":
            raise ValueError(
                f"{source} height scanner {i} has ray_alignment="
                f"'{scan['ray_alignment']}'; only a yaw-aligned grid puts each ray at a "
                "FIXED point in the generator's xyz+yaw window frame."
            )
        height, width = int(scan["height"]), int(scan["width"])
        if height * width != int(scan["num_rays"]):
            raise ValueError(
                f"{source} height scanner {i} folds as {height}x{width} = {height * width} "
                f"but declares {scan['num_rays']} rays; any image built from it would be "
                "scrambled."
            )
        out.append(dict(scan))
    return out


def parse_extero_spec(raw, extero_source: str, source: str = "spec") -> list[dict]:
    """Validate an exteroception contract with the parser its own SOURCE names.

    The one entry point for ``fm_meta.extero_spec``, so no caller has to know that the same
    field holds cameras for one checkpoint and a ray grid for another -- which is how a
    scan-conditioned graph came to be refused for missing ``fx``.

    Args:
        raw: The embedded contract (or ``None``).
        extero_source: The checkpoint's ``extero_source`` -- ``"depth"`` or ``"height_scan"``.
        source: Label for error messages.

    Returns:
        One validated dict per sensor.  Empty for ``None``.

    Raises:
        ValueError: On an unknown ``extero_source``, or from the delegated parser.
    """
    if extero_source == "depth":
        return parse_depth_spec(raw, source=source)
    if extero_source == "height_scan":
        return parse_scan_spec(raw, source=source)
    raise ValueError(
        f"{source} declares extero_source '{extero_source}', which is neither 'depth' nor "
        "'height_scan'; there is no parser for its contract."
    )


def assert_scan_specs_agree(
    a: list[dict], b: list[dict], name_a: str = "tracker", name_b: str = "generator",
) -> None:
    """Raise unless two artifacts describe the same height-scan grid.

    Stricter than the depth sibling, and deliberately: the runner casts ONE grid per step
    (``Robot.compute_height_scan``), so unlike the camera pair there is no second sampling to
    hand a second consumer.  A tracker and a generator that want different grids therefore
    cannot both be served, and the failure would otherwise be a silently wrong-width array
    dying inside a normalizer broadcast.

    (Isaac *can* run both -- ``add_fm_height_scan`` adds the generator's dense scanner
    alongside the policy's -- so lifting this means teaching the runner to cast both, the way
    it already casts two depth samplings.)

    Args:
        a: Parsed contract from the first artifact.
        b: Contract from the second artifact -- possibly the SMALLER ``tracker.onnx`` shape.
        name_a: Label for the first artifact, used in the error.
        name_b: Label for the second artifact, used in the error.

    Raises:
        ValueError: On a differing scanner count, a contract too sparse to compare, or any
            differing geometric field -- listing every disagreement rather than only the
            first.
    """
    if len(a) != len(b):
        raise ValueError(
            f"{name_a} has {len(a)} height scanner(s) but {name_b} has {len(b)}; there is "
            "one physical robot casting one grid."
        )
    diffs: list[str] = []
    for i, (scan_a, scan_b) in enumerate(zip(a, b)):
        label = scan_a.get("sensor") or f"scanner {i}"
        shared = (set(scan_a) & set(scan_b)) - _SCAN_NAMING_FIELDS
        thin = _SCAN_COMPARABLE_FLOOR - shared
        if thin:
            raise ValueError(
                f"{name_a} and {name_b} share too little of the scan contract to be "
                f"compared: neither states {sorted(thin)} in common. Re-export whichever "
                "artifact is missing them rather than running an unchecked pair."
            )
        for field in sorted(shared):
            va, vb = scan_a.get(field), scan_b.get(field)
            if _values_differ(va, vb):
                diffs.append(f"  {label}.{field}: {name_a}={va!r} {name_b}={vb!r}")
    if diffs:
        raise ValueError(
            f"{name_a} and {name_b} describe DIFFERENT height-scan grids:\n"
            + "\n".join(diffs)
            + "\nThe runner casts one grid per step, so a tracker and a generator running "
              "together cannot each want their own."
        )


def is_privileged(spec: list[dict]) -> bool:
    """Whether any scanner in ``spec`` is the generator's exact, noise-free sampling.

    Args:
        spec: A parsed height-scan contract.

    Returns:
        True when at least one entry is marked ``privileged``.
    """
    return any(bool(s.get("privileged")) for s in spec)


def describe(spec: list[dict]) -> str:
    """One-line human description of a scan contract, for load-time prints and errors.

    Args:
        spec: A parsed height-scan contract.

    Returns:
        A string like ``"fm_height_scanner [3.0, 3.0] m @ 0.025 m (121x121 = 14641 rays)"``.
    """
    return ", ".join(
        f"{s.get('sensor')} {s.get('size')} m @ {s.get('resolution')} m "
        f"({s.get('height')}x{s.get('width')} = {s.get('num_rays')} rays)" for s in spec
    )


def _values_differ(va, vb) -> bool:
    """Whether two contract values differ, comparing numbers within float tolerance.

    A parsed contract holds only ``None``, strings, numbers, dicts and lists of numbers, so a
    JSON round-trip's last-bit wobble must not read as a geometry mismatch.

    Args:
        va: Value from the first contract.
        vb: Value from the second contract.

    Returns:
        True when the two are meaningfully different.
    """
    if isinstance(va, (list, tuple)) and isinstance(vb, (list, tuple)):
        return len(va) != len(vb) or any(_values_differ(x, y) for x, y in zip(va, vb))
    if isinstance(va, bool) or isinstance(vb, bool):
        return bool(va) != bool(vb)
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return abs(float(va) - float(vb)) > 1e-9
    return va != vb
