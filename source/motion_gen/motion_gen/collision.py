# Copyright 2026 Zachary Olkin. All rights reserved.

"""The robot's own collision geometry, as points with radii, read out of the MJCF.

A terrain-consistency reward has to ask "is this body inside the ground", which needs the
body's SHAPE and not just its origin.  Rather than restate any of it, this parses the
collision geoms out of the same MJCF :class:`~motion_gen.fk.FKChain` already FKs -- so the
reward's notion of how big a foot is comes from the robot file, and a future gear, hand or
payload change moves the reward with it instead of leaving a stale constant behind.

Each collider becomes a small set of ``(point, radius)`` pairs in its BODY frame, and the
terrain test is then ``sdf(point) - radius``:

===========  ==========================================  ======
geom type    points                                      radius
===========  ==========================================  ======
``box``      the 8 corners                                0
``sphere``   the center                                   ``size[0]``
``cylinder`` the two end-cap centers, along local z       ``size[0]``
``capsule``  the two end-cap centers, along local z       ``size[0]``
===========  ==========================================  ======

A cylinder read as a capsule over-covers its rim by up to ``r`` at the flat end faces, which
is conservative in the direction that matters (it can only over-report proximity, never miss
a penetration) and is the standard 2-sphere swept-volume reading.  A **collision mesh is
refused**: the G1's collision set is primitives only, and silently skipping one collider is
exactly the failure this module exists to prevent -- a body with no points is a body that
can pass through the terrain with no penalty and no error.

For the G1 (``g1_perceptive.xml``) this yields 11 bodies: two foot boxes, two shank and two
thigh cylinders, two forearm cylinders, two wrist spheres, the torso box and the backpack
box.  Note what is NOT there: **``pelvis_link`` and the shoulder links carry no collision
geometry at all** -- the trunk is one ``torso_link`` box spanning waist to shoulders -- so
"the pelvis" and "the shoulders" resolve onto that box rather than onto invented spheres.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

from .fk import _parse_mjcf, _quat_wxyz_to_matrix, _vec


@dataclass(frozen=True)
class BodyColliders:
    """One body's collision geometry as points with radii, in the body's own frame."""

    body: str
    """The MJCF body name."""

    points: np.ndarray
    """``(P, 3)`` collider sample points in the body frame (m)."""

    radii: np.ndarray
    """``(P,)`` radius per point (m); ``0`` for a surface sample or a box corner."""

    areas: np.ndarray
    """``(P,)`` surface area each point represents (m^2); ``0`` for an extreme point.

    This is what makes a SUM over points an integral rather than a point count. Without it a
    sum silently re-weights itself whenever the sampling density or the body mix changes --
    the same drift the window's one-producer tables exist to prevent. Extreme points (box
    corners, cylinder end-cap centres) carry zero area on purpose: they exist so a ``max`` or
    a contact test sees the true extremes, and giving them area would double-count the
    surface they sit on.
    """

    kinds: tuple[str, ...]
    """The geom type each point came from, one entry per point -- for reporting."""

    def __post_init__(self) -> None:
        """Check the arrays agree, since a mismatch would silently mis-pair radii."""
        if not (self.points.shape[0] == self.radii.shape[0] == self.areas.shape[0]):
            raise ValueError(
                f"{self.body}: {self.points.shape[0]} points, {self.radii.shape[0]} radii, "
                f"{self.areas.shape[0]} areas -- these must agree"
            )


def _trapezoid_weights(n: int, length: float) -> np.ndarray:
    """1-D trapezoid quadrature weights over ``n`` endpoint-inclusive samples of ``length``.

    Endpoint-inclusive so a box's samples include its edges and corners -- the extremes are
    what a penetration test cares about -- and trapezoid-weighted so the weights still sum to
    ``length``.  A shared edge point therefore appears on two faces with half weight each,
    which is correct for a sum and harmless for a min or a max.

    Args:
        n: Number of samples (``>= 1``).
        length: Total extent the samples span (m).

    Returns:
        ``(n,)`` weights summing to ``length``.
    """
    if n <= 1:
        return np.array([length], dtype=np.float64)
    w = np.full(n, length / (n - 1), dtype=np.float64)
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def _axis_samples(extent: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Endpoint-inclusive samples across ``[-extent/2, extent/2]`` at about ``spacing``.

    Args:
        extent: Full extent (m).
        spacing: Target sample spacing (m).

    Returns:
        ``(coordinates (n,), trapezoid weights (n,))``.
    """
    n = max(2, int(np.ceil(extent / max(spacing, 1e-9))) + 1)
    return (np.linspace(-extent / 2.0, extent / 2.0, n),
            _trapezoid_weights(n, extent))


def _box_surface(half: np.ndarray, spacing: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples of an axis-aligned box, with area weights.

    Args:
        half: ``(3,)`` half-extents (m).
        spacing: Target sample spacing (m).

    Returns:
        ``(points (P, 3), areas (P,))`` in the box's own frame.
    """
    pts, areas = [], []
    for axis in range(3):
        u, v = [a for a in range(3) if a != axis]
        cu, wu = _axis_samples(2.0 * half[u], spacing)
        cv, wv = _axis_samples(2.0 * half[v], spacing)
        gu, gv = np.meshgrid(cu, cv, indexing="ij")
        au, av = np.meshgrid(wu, wv, indexing="ij")
        for sign in (-1.0, 1.0):
            p = np.zeros((gu.size, 3))
            p[:, axis] = sign * half[axis]
            p[:, u] = gu.reshape(-1)
            p[:, v] = gv.reshape(-1)
            pts.append(p)
            areas.append((au * av).reshape(-1))
    return np.concatenate(pts), np.concatenate(areas)


def _cylinder_surface(radius: float, half_len: float, spacing: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples of a cylinder about local z: barrel plus both end caps.

    Args:
        radius: Cylinder radius (m).
        half_len: Half length along z (m).
        spacing: Target sample spacing (m).

    Returns:
        ``(points (P, 3), areas (P,))`` in the geom's own frame.
    """
    pts, areas = [], []
    # Barrel. theta is periodic, so its weights are uniform with no endpoint halving.
    n_th = max(4, int(np.ceil(2.0 * np.pi * radius / max(spacing, 1e-9))))
    th = np.arange(n_th) * (2.0 * np.pi / n_th)
    dz, wz = _axis_samples(2.0 * half_len, spacing)
    a_th = 2.0 * np.pi * radius / n_th
    for z, w in zip(dz, wz):
        pts.append(np.stack([radius * np.cos(th), radius * np.sin(th),
                             np.full(n_th, z)], axis=-1))
        areas.append(np.full(n_th, a_th * w))
    # End caps, as concentric rings of equal radial spacing plus the centre.
    n_r = max(1, int(np.ceil(radius / max(spacing, 1e-9))))
    edges = np.linspace(0.0, radius, n_r + 1)
    for sign in (-1.0, 1.0):
        pts.append(np.array([[0.0, 0.0, sign * half_len]]))
        areas.append(np.array([np.pi * (edges[1] / 2.0) ** 2]))
        for i in range(n_r):
            r_mid = 0.5 * (edges[i] + edges[i + 1])
            n_ring = max(4, int(np.ceil(2.0 * np.pi * r_mid / max(spacing, 1e-9))))
            a = np.arange(n_ring) * (2.0 * np.pi / n_ring)
            pts.append(np.stack([r_mid * np.cos(a), r_mid * np.sin(a),
                                 np.full(n_ring, sign * half_len)], axis=-1))
            ring_area = np.pi * (edges[i + 1] ** 2 - edges[i] ** 2)
            areas.append(np.full(n_ring, ring_area / n_ring))
    return np.concatenate(pts), np.concatenate(areas)


def _sphere_surface(radius: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Surface samples of a sphere on a Fibonacci lattice, with equal area weights.

    Args:
        radius: Sphere radius (m).
        spacing: Target sample spacing (m).

    Returns:
        ``(points (P, 3), areas (P,))``.
    """
    area = 4.0 * np.pi * radius * radius
    n = max(8, int(np.ceil(area / max(spacing * spacing, 1e-12))))
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * i
    p = radius * np.stack([np.cos(theta) * np.sin(phi),
                           np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)
    return p, np.full(n, area / n)


def _geom_points(
    geom: ET.Element, body: str, spacing: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Sample points, radii and area weights for one collision geom, in the body's frame.

    Two modes, and the difference is the whole design:

    * ``spacing is None`` -- **extreme points**: a box's 8 corners, a cylinder's two end-cap
      centres, a sphere's centre, each carrying the geom's ``radius`` so the test is
      ``sd - radius``.  Cheap (46 points for the whole G1) and it sees the extremes, but a
      cylinder read as two spheres over-covers its flat end faces by up to ``r``, and 46
      sparse points make a SUM meaningless.
    * ``spacing`` given -- **surface samples** at about that spacing, with trapezoid /
      equal-area weights and **radius 0**, because the points are already on the surface.
      This is exact for every primitive (the capsule approximation disappears rather than
      being densified around) and it is what makes an area-weighted sum an integral of
      penetration depth over the body's surface.

    Args:
        geom: The ``<geom>`` element.
        body: Parent body name, for error messages.
        spacing: Target surface-sample spacing (m), or ``None`` for extreme points.

    Returns:
        ``(points (P, 3), radii (P,), areas (P,), kind)``.

    Raises:
        ValueError: On a collision mesh, an unsupported geom type, or a missing ``size``.
    """
    kind = geom.get("type", "sphere")
    pos = _vec(geom.get("pos"), (0.0, 0.0, 0.0))
    rot = _quat_wxyz_to_matrix(_vec(geom.get("quat"), (1.0, 0.0, 0.0, 0.0)))

    def place(local: np.ndarray) -> np.ndarray:
        """Rotate and translate local geom-frame points into the body frame."""
        return local @ rot.T + pos

    fromto = geom.get("fromto")
    if fromto is not None:
        ft = np.array([float(v) for v in fromto.split()], dtype=np.float64)
        if ft.size != 6:
            raise ValueError(f"{body}: geom fromto needs 6 numbers, got {ft.size}")
        r = float(_vec(geom.get("size"), (0.0,))[0])
        if spacing is None:
            return ft.reshape(2, 3), np.full(2, r), np.zeros(2), kind
        # A fromto capsule: sample it as a cylinder along its own axis, then place it.
        a, b = ft[:3], ft[3:]
        axis = b - a
        half = float(np.linalg.norm(axis)) / 2.0
        pts, areas = _cylinder_surface(r, half, spacing)
        z = axis / max(np.linalg.norm(axis), 1e-12)
        tmp = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(tmp, z); x /= max(np.linalg.norm(x), 1e-12)
        basis = np.stack([x, np.cross(z, x), z], axis=-1)
        return pts @ basis.T + 0.5 * (a + b), np.zeros(len(pts)), areas, kind

    size_attr = geom.get("size")
    if size_attr is None:
        raise ValueError(
            f"{body}: collision geom of type {kind!r} has no 'size', so its extent is "
            "unknown. A collider with no extent cannot be tested against the terrain."
        )
    size = np.array([float(v) for v in size_attr.split()], dtype=np.float64)

    if kind == "box":
        if size.size != 3:
            raise ValueError(f"{body}: box geom needs 3 half-extents, got {size.size}")
        corners = np.array([[sx * size[0], sy * size[1], sz * size[2]]
                            for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                            for sz in (-1.0, 1.0)])
        if spacing is None:
            return place(corners), np.zeros(8), np.zeros(8), kind
        surf, areas = _box_surface(size, spacing)
        # The corners are already ON the endpoint-inclusive lattice, so no extras are added.
        return place(surf), np.zeros(len(surf)), areas, kind
    if kind == "sphere":
        r = float(size[0])
        if spacing is None:
            return pos[None, :], np.array([r]), np.zeros(1), kind
        surf, areas = _sphere_surface(r, spacing)
        return place(surf), np.zeros(len(surf)), areas, kind
    if kind in ("cylinder", "capsule"):
        if size.size < 2:
            raise ValueError(
                f"{body}: {kind} geom needs (radius, half-length), got {size.size} number(s)")
        r, half = float(size[0]), float(size[1])
        if spacing is None:
            ends = np.array([[0.0, 0.0, -half], [0.0, 0.0, half]])
            return place(ends), np.full(2, r), np.zeros(2), kind
        surf, areas = _cylinder_surface(r, half, spacing)
        return place(surf), np.zeros(len(surf)), areas, kind
    if kind == "mesh":
        raise ValueError(
            f"{body}: a COLLISION mesh geom. This robot's collision set is primitives only, "
            "and approximating a mesh here would be a guess -- while skipping it would give "
            "the body no collider points and therefore no terrain penalty, silently. Give "
            "the body a primitive collider, or exclude it explicitly."
        )
    raise ValueError(f"{body}: unsupported collision geom type {kind!r}")


def collider_points(
    mjcf_path: str,
    body_names: list[str] | None = None,
    spacing: float | None = None,
) -> dict[str, BodyColliders]:
    """Collision sample points, per body, in each body's own frame.

    A geom counts as collision geometry when its ``contype`` is not ``"0"`` -- the actual
    MuJoCo semantics, rather than the geom ``group``, which is a rendering/raycast channel
    (the G1's 37 visual meshes sit in group 1 with ``contype="0"``, and the depth cameras
    cast against exactly those).

    Args:
        mjcf_path: The MJCF to read.  Use ``motion_gen.features.DEFAULT_MJCF``.
        body_names: Bodies to report, or ``None`` for **every** body that has a collider.
            ``None`` is the recommended default: it makes the set a property of the robot
            file instead of a list that can fall behind it.
        spacing: Surface-sample spacing (m), or ``None`` for extreme points only. See
            :func:`_geom_points` for what the two modes are for.

    Returns:
        Mapping body name -> :class:`BodyColliders`, in MJCF tree order when
        ``body_names`` is ``None`` and in the requested order otherwise.

    Raises:
        ValueError: If a requested body is unknown or has no collision geometry -- both are
            refusals, since a body silently contributing no points is a body the reward
            cannot see through.
    """
    root = ET.parse(mjcf_path).getroot()
    found: dict[str, BodyColliders] = {}

    def walk(elem: ET.Element) -> None:
        for body in elem.findall("body"):
            name = body.get("name")
            if name is None:
                raise ValueError(f"{mjcf_path}: a <body> has no name")
            pts, rads, ars, kinds = [], [], [], []
            for geom in body.findall("geom"):
                if geom.get("contype") == "0":
                    continue
                p, r, a, kind = _geom_points(geom, name, spacing)
                pts.append(p)
                rads.append(r)
                ars.append(a)
                kinds.extend([kind] * p.shape[0])
            if pts:
                found[name] = BodyColliders(name, np.concatenate(pts, axis=0),
                                            np.concatenate(rads, axis=0),
                                            np.concatenate(ars, axis=0), tuple(kinds))
            walk(body)

    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"{mjcf_path}: no <worldbody>")
    walk(world)

    if body_names is None:
        return found
    out: dict[str, BodyColliders] = {}
    for name in body_names:
        if name not in found:
            known = ", ".join(sorted(found))
            raise ValueError(
                f"{name!r} has no collision geometry in {mjcf_path} (bodies that do: "
                f"{known}). Asking for it would add a body with zero collider points, which "
                "the terrain reward could never penalise."
            )
        out[name] = found[name]
    return out


def flatten_colliders(
    colliders: dict[str, BodyColliders],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a collider mapping into the arrays a batched FK + SDF pass wants.

    Args:
        colliders: As returned by :func:`collider_points`.

    Returns:
        ``(body_order, points (P, 3), radii (P,), areas (P,), body_index (P,))`` where
        ``body_index`` maps each point to its position in ``body_order`` -- which is what
        lets a per-body reduction (or a per-body contact exemption) be a scatter rather than
        a loop.
    """
    body_order = list(colliders)
    pts = [colliders[b].points for b in body_order]
    rads = [colliders[b].radii for b in body_order]
    ars = [colliders[b].areas for b in body_order]
    idx = [np.full(colliders[b].points.shape[0], i, dtype=np.int64)
           for i, b in enumerate(body_order)]
    return (body_order, np.concatenate(pts, axis=0), np.concatenate(rads, axis=0),
            np.concatenate(ars, axis=0), np.concatenate(idx, axis=0))
