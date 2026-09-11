# Copyright 2026 Zachary Olkin. All rights reserved.

"""Copy one env's complete controller-visible state onto other envs, in place.

The best-of-K test needs K envs that are the SAME robot in the SAME situation, differing
only in the flow-matching noise draw for one chunk.  Isaac Lab has no state save/restore, so
this assembles one: an inventory of every per-env buffer that feeds the next control step,
and a copy over it.

**What this can and cannot do, stated up front because the whole experiment rests on it.**

It restores the state the CONTROLLER sees: rigid-body pose and velocity, joint state,
observation history, action history, the generator's plan / anchor / history ring, the depth
latency ring, and the episode step counter that drives the re-plan cadence.  It does NOT
restore PhysX's internal solver state -- contact manifolds, friction anchors, warm-start
impulses -- because the tensor API exposes no handle on them.  So a restore is exact in
every quantity the policy and the generator read, and approximate in the simulator's own
warm-start.

That residue is not argued away, it is MEASURED, and continuously: candidate 0 is handed the
parent env's own noise draw and the same continuation seed, so candidate 0's rolled window
and the parent's own next W seconds are the same computation up to this restore.  Their
score difference is the harness noise floor, recorded at every branched state (see
``headroom_schema.PARENT_ROW``).  Read every spread against it.

**Discovery is by SHAPE, with the inventory printed and pinned.**  Hand-listing the buffers
was tried on paper and came to thirty-odd names across five objects, which is exactly the
kind of list that goes stale the next time a field is added to ``FmTrackingCommand``.  So
the walk finds every tensor whose leading dimension is ``num_envs`` on a small set of named
owners, and ``describe()`` renders it so a test can assert the ones that matter are present.
The cost of the shape test is that a buffer whose leading dimension coincidentally equals
``num_envs`` is copied too; every such copy is a no-op or harmless, and :data:`SKIP_ATTRS`
carries the ones that are neither.
"""

from __future__ import annotations

import collections

import torch

#: Attributes never copied, by owner-qualified name.  Each is either a shared constant that
#: happens to be ``num_envs`` long, or a buffer whose value must follow the DESTINATION env
#: rather than the source.
SKIP_ATTRS: frozenset[str] = frozenset({
    # The clip library and the terrain map: shared, read-only, and enormous.
    "cmd._clip_lib",
    "cmd.manager",
    # Marker prims and their pose buffers -- debug visualisation only, and copying a
    # VisualizationMarkers handle is meaningless.
    "cmd._debug_markers",
    "cmd._debug_body_markers",
    # The diagnostic exteroception stash: written by the generator every re-plan, read only
    # by the flagged-plan dump, which this harness does not run.
    "cmd._extero_stash",
})

#: Recursion depth for the container walk: enough for ``deque[dict[str, Tensor]]`` (the
#: generator's history ring) and ``list[Tensor]`` (the depth latency ring), and shallow
#: enough that it cannot wander into the scene graph.
_MAX_DEPTH: int = 3


class Slot:
    """One copyable per-env buffer: a tensor plus the path that names it."""

    __slots__ = ("path", "tensor")

    def __init__(self, path: str, tensor: torch.Tensor) -> None:
        """Bind a tensor to its inventory path.

        Args:
            path: Dotted owner-qualified name, e.g. ``cmd._pred_window``.
            tensor: The live buffer.  Held by reference, so a rebind on the owner (a list
                that is replaced rather than mutated) invalidates it -- which is why
                :meth:`EnvStateMirror.refresh` exists.
        """
        self.path = path
        self.tensor = tensor


def _walk(obj, path: str, num_envs: int, depth: int, out: list[Slot],
          seen: set[int]) -> None:
    """Collect per-env tensors reachable from ``obj``, depth-first.

    Args:
        obj: Object to inspect -- a module-like owner, a container, or a tensor.
        path: Dotted path to ``obj``, for the inventory.
        num_envs: Leading dimension a tensor must have to count as per-env.
        depth: Remaining recursion budget.
        out: Accumulator.
        seen: ``id()`` of objects already visited, so a shared buffer is listed once.
    """
    if path in SKIP_ATTRS or depth < 0:
        return
    if isinstance(obj, torch.Tensor):
        if obj.ndim >= 1 and obj.shape[0] == num_envs and id(obj) not in seen:
            seen.add(id(obj))
            out.append(Slot(path, obj))
        return
    if isinstance(obj, (list, tuple, collections.deque)):
        for i, item in enumerate(obj):
            _walk(item, f"{path}[{i}]", num_envs, depth - 1, out, seen)
        return
    if isinstance(obj, dict):
        for key, item in obj.items():
            _walk(item, f"{path}[{key!r}]", num_envs, depth - 1, out, seen)
        return
    # A plain object: only its own __dict__ / __slots__, never properties. A property can
    # recompute a buffer from the sim on access, so reading one here would both cost a
    # device sync and hand back a value that is not the stored state.
    names = list(getattr(obj, "__dict__", {}))
    for name in getattr(obj, "__slots__", ()) or ():
        if name not in names:
            names.append(name)
    for name in names:
        if name.startswith("__"):
            continue
        try:
            value = getattr(obj, name)
        except AttributeError:
            continue
        if isinstance(value, (torch.Tensor, list, tuple, dict, collections.deque)):
            _walk(value, f"{path}.{name}", num_envs, depth - 1, out, seen)


class EnvStateMirror:
    """Copy the full controller-visible state between envs of one running environment.

    Built once against a live env.  :meth:`refresh` re-walks the inventory and must be
    called after anything that REBINDS a container rather than mutating it -- the depth
    latency ring is rebuilt by slicing (``del buf[:-n]``) and the generator's history ring is
    a deque whose entries are new dicts every step.
    """

    def __init__(self, env, *, command_name: str = "traj_ref") -> None:
        """Resolve the owners and take the first inventory.

        Args:
            env: The unwrapped ``ManagerBasedRLEnv``.
            command_name: The command term holding the generator state.
        """
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = env.device
        self.robot = env.scene.articulations["robot"]
        self.cmd = env.command_manager.get_term(command_name)
        self._slots: list[Slot] = []
        self._buffers: list[tuple[str, object]] = []
        self._derived: list[tuple[str, object]] = []
        self.refresh()

    # ------------------------------------------------------------------ inventory
    def _owners(self) -> list[tuple[str, object]]:
        """The objects walked for per-env tensors, in inventory-path order."""
        owners: list[tuple[str, object]] = [
            ("cmd", self.cmd),
            # NAMED explicitly because the walk does not recurse into plain objects: it
            # follows only tensors and containers, so a BatchedHistory attribute is a dead
            # end. And this is the single most important buffer in the inventory -- it IS
            # the autoregression, so a branch that does not carry it is conditioned on the
            # wrong past while every shape still agrees.
            ("cmd._history", getattr(self.cmd, "_history", None)),
            ("actions", self.env.action_manager),
        ]
        for name, term in self.env.action_manager._terms.items():
            owners.append((f"action:{name}", term))
        for noiser in _depth_noisers(self.env):
            owners.append((f"depth_noise:{id(noiser):x}", noiser))
        return owners

    def refresh(self) -> None:
        """Re-walk the inventory.  Cheap: no device work, only attribute reads."""
        slots: list[Slot] = []
        seen: set[int] = set()
        for path, owner in self._owners():
            if owner is None:
                continue
            _walk(owner, path, self.num_envs, _MAX_DEPTH, slots, seen)
        # ``episode_length_buf`` is reached explicitly rather than by walking the env, whose
        # __dict__ is the whole scene. It is the one env-level buffer that must travel: the
        # generator's re-plan cadence is ``episode_length_buf % fm_update_every``, so a
        # branch env whose counter is off by one re-plans on a different step than the
        # parent and every candidate is then sampled from a different phase of the plan.
        slots.append(Slot("env.episode_length_buf", self.env.episode_length_buf))
        self._slots = slots
        self._buffers = _history_buffers(self.env)
        # The DERIVED half: articulation caches and sensor readings. Separate from _slots
        # because these are not walked (warp arrays with struct dtypes) and, unlike a slot,
        # a stale one is invisible -- it has the right shape and a current timestamp.
        self._derived = (_derived_buffers(self.robot._data, "robot.data", self.num_envs)
                         + _sensor_buffers(self.env, self.num_envs))

    def describe(self) -> list[str]:
        """Inventory paths, sorted -- for printing at run start and for the pinning test."""
        return sorted([s.path for s in self._slots]
                      + [p for p, _ in getattr(self, "_derived", ())]
                      + [f"{p}(circular)" for p, _ in self._buffers])

    def missing(self, required: list[str]) -> list[str]:
        """Required inventory paths that the walk did not find.

        Args:
            required: Exact paths that must be present.

        Returns:
            The absent ones, in the order given.
        """
        have = set(self.describe())
        return [r for r in required if r not in have]

    # ------------------------------------------------------------------ copy
    def capture(self, src_ids: torch.Tensor) -> dict:
        """Clone the state of ``src_ids`` into a host-free snapshot.

        Args:
            src_ids: ``[m]`` long tensor of env indices.

        Returns:
            An opaque snapshot for :meth:`restore`, valid until the next
            :meth:`refresh` changes the inventory.
        """
        src = src_ids.to(self.device)
        snap = {
            "n": int(src.numel()),
            "root_pose": _t(self.robot.data.root_link_pose_w)[src].clone(),
            "root_vel": _t(self.robot.data.root_link_vel_w)[src].clone(),
            "joint_pos": _t(self.robot.data.joint_pos)[src].clone(),
            "joint_vel": _t(self.robot.data.joint_vel)[src].clone(),
            "slots": [s.tensor[src].clone() for s in self._slots],
            "paths": [s.path for s in self._slots],
            "buffers": [(buf._buffer[:, src, :].clone(), buf._num_pushes[src].clone())
                        if buf._buffer is not None else None
                        for _, buf in self._buffers],
            "derived": [t[src].clone() for _, t in self._derived],
            "derived_paths": [path for path, _ in self._derived],
        }
        return snap

    def restore(self, dst_ids: torch.Tensor, snap: dict, *,
                src_index: torch.Tensor | int | None = None) -> None:
        """Write a snapshot onto ``dst_ids``.

        Args:
            dst_ids: ``[k]`` long tensor of destination env indices.
            snap: A snapshot from :meth:`capture`.
            src_index: Which snapshot row each destination takes.  ``None`` requires
                ``len(dst_ids) == snap["n"]`` and pairs them in order; an int broadcasts one
                row to every destination; a ``[k]`` tensor selects per destination.

        Raises:
            ValueError: On a length mismatch, or if the inventory has changed since the
                snapshot was taken -- copying by position into a re-walked inventory would
                write the depth ring into the history ring with every shape agreeing.
        """
        if snap["paths"] != [s.path for s in self._slots]:
            raise ValueError(
                "the state inventory changed between capture and restore; call capture() "
                "again after refresh(). Paths differ at "
                f"{_first_difference(snap['paths'], [s.path for s in self._slots])}."
            )
        dst = dst_ids.to(self.device)
        k = int(dst.numel())
        if src_index is None:
            if k != snap["n"]:
                raise ValueError(f"restore: {k} destinations against {snap['n']} snapshot "
                                 "rows and no src_index")
            idx = torch.arange(k, device=self.device)
        elif isinstance(src_index, int):
            idx = torch.full((k,), int(src_index), device=self.device, dtype=torch.long)
        else:
            idx = src_index.to(self.device).long()
            if int(idx.numel()) != k:
                raise ValueError(f"restore: src_index has {idx.numel()} entries for {k} "
                                 "destinations")

        for slot, saved in zip(self._slots, snap["slots"]):
            slot.tensor[dst] = saved[idx]
        for (_, buf), saved in zip(self._buffers, snap["buffers"]):
            if saved is None or buf._buffer is None:
                continue
            buf._buffer[:, dst, :] = saved[0][:, idx, :]
            buf._num_pushes[dst] = saved[1][idx]

        # The sim last, and through the same four calls ``mdp.events.resets`` uses -- so the
        # quaternion convention (xyzw) and the LINK-frame velocity are whatever those
        # already agree on, rather than a second opinion stated here.
        self.robot.write_root_pose_to_sim_index(root_pose=snap["root_pose"][idx],
                                               env_ids=dst)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=snap["root_vel"][idx],
                                                        env_ids=dst)
        self.robot.write_joint_position_to_sim_index(position=snap["joint_pos"][idx],
                                                    env_ids=dst)
        self.robot.write_joint_velocity_to_sim_index(velocity=snap["joint_vel"][idx],
                                                    env_ids=dst)

        # DERIVED last, so the sim writes cannot clobber it. Those writes refresh the seven
        # pose/state caches they invalidate and leave every other cache -- projected
        # gravity, joint acceleration, the sensor hits -- holding the destination's own
        # values behind a current timestamp. See _derived_buffers.
        if snap["derived_paths"] != [p for p, _ in self._derived]:
            raise ValueError(
                "the derived inventory changed between capture and restore; call capture() "
                "again after refresh(). Paths differ at "
                f"{_first_difference(snap['derived_paths'], [p for p, _ in self._derived])}.")
        for (_, tensor), saved in zip(self._derived, snap["derived"]):
            tensor[dst] = saved[idx]

    def copy(self, src_id: int, dst_ids: torch.Tensor) -> None:
        """Broadcast one env's state onto several, the common case.

        Args:
            src_id: Source env index.
            dst_ids: Destination env indices.
        """
        snap = self.capture(torch.tensor([int(src_id)], device=self.device))
        self.restore(dst_ids, snap, src_index=0)



#: ``ArticulationData`` buffers that are RE-READ from the PhysX view on every access, and
#: whose source is solver OUTPUT rather than state.  Eight buffers rebind ``.data`` on read;
#: six of them re-read a quantity the four ``write_*_to_sim`` calls have already set (root
#: transforms, root velocities, link transforms, link velocities, dof positions, dof
#: velocities), so they come back correct on their own.  These two do not:
#:
#: * ``_body_com_acc_w`` -- ``get_link_accelerations()``, and there is no setter.
#: * ``_body_incoming_joint_wrench_b`` -- ``get_link_incoming_joint_force()``, likewise.
#:
#: Copying them would be a no-op DRESSED AS a restore: the write lands in the old array and
#: the next read rebinds past it.  So they are excluded rather than silently copied, and the
#: probe measures them separately as what they are -- unrestorable, alongside the contact
#: manifolds and friction anchors.  Nothing in ``robot_rl`` reads either (no ``body_acc_w``
#: or ``body_incoming_joint_wrench`` appears outside tests), so neither reaches the policy,
#: the generator, a reward or a termination.
SOLVER_OUTPUT: frozenset[str] = frozenset({
    "_body_com_acc_w", "_body_incoming_joint_wrench_b",
})

def _leading_env_dim(value, num_envs: int):
    """The torch view of ``value`` if it is a per-env buffer, else ``None``.

    Accepts warp arrays as well as tensors: ``ArticulationData`` stores ``wp.array`` with
    struct dtypes (``wp.transformf``, ``wp.vec3f``), which ``wp.to_torch`` views as a
    trailing-dimension float tensor sharing the same memory -- so a write through the view
    reaches the buffer the simulator reads.
    """
    import warp as wp  # noqa: PLC0415

    if isinstance(value, wp.array):
        if value.shape and value.shape[0] == num_envs:
            try:
                return wp.to_torch(value)
            except (RuntimeError, TypeError):
                return None
        return None
    if isinstance(value, torch.Tensor):
        return value if value.ndim and value.shape[0] == num_envs else None
    return None


def _derived_buffers(owner, path: str, num_envs: int) -> list[tuple[str, object]]:
    """Per-env buffers on a data object, both bare and ``TimestampedBuffer``-wrapped.

    **Why these are copied at all, when the four ``write_*_to_sim`` calls already put the
    state into PhysX.**  ``ArticulationData`` caches every DERIVED quantity behind a
    timestamp compared against ``_sim_timestamp``, and a sim write advances neither.
    ``write_root_link_pose_to_sim_index`` invalidates exactly seven buffers by hand --
    ``_root_link_state_w``, ``_root_state_w`` and the five body pose/state ones -- and
    ``_projected_gravity_b`` is NOT among them.  So a restored env keeps serving its OWN
    orientation, from a cache whose timestamp is already current, until something steps the
    sim.  That is a core observation term, and this repo has met the same buffer before: the
    eval harness's fall flag once read the previous episode's tilt for exactly this reason
    (see CLAUDE.md).

    Copying the cache CONTENTS rather than invalidating them is deliberate.  Invalidation
    would fix the pose-derived buffers but break ``joint_acc``, which finite-differences
    against ``_previous_joint_vel`` over ``_sim_timestamp - _joint_acc.timestamp`` -- a
    stamp forced to ``-1.0`` turns that elapsed time into garbage.  Copying leaves every
    timestamp alone: a current cache is now the source's value, and a stale one recomputes
    from state that is already the source's.

    Discovery is by SHAPE, as the main inventory is, so a buffer added by a future IsaacLab
    cannot be silently missed; ``describe()`` prints what was found.
    """
    out: list[tuple[str, object]] = []
    for name, value in sorted(vars(owner).items()):
        if name.startswith("__") or name in SOLVER_OUTPUT:
            continue
        _collect(value, f"{path}.{name}", num_envs, 2, out)
    return out


def _collect(value, path: str, num_envs: int, depth: int, out: list) -> None:
    """Record ``value`` if it is a per-env buffer, recursing one level into containers.

    The container arm is not cosmetic.  ``RayCasterCameraData.output`` is a DICT of images
    keyed by data type, so a walk that only accepted bare tensors copied each depth camera's
    pose, orientation and intrinsics -- everything that describes where the camera is --
    while leaving the image itself behind.  The restored env then reported its own view of
    the world from the parent's position, and only the two depth terms of the observation
    were wrong, which looks far more like a subtle normalisation bug than like a buffer that
    was never copied at all.
    """
    view = _leading_env_dim(value, num_envs)
    if view is not None:
        out.append((path, view))
        return
    if depth <= 0:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect(item, f"{path}[{key!r}]", num_envs, depth - 1, out)
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _collect(item, f"{path}[{i}]", num_envs, depth - 1, out)
        return
    # A TimestampedBuffer: the payload is .data and the stamp is left untouched.
    inner = getattr(value, "data", None)
    if inner is not None and not isinstance(value, torch.Tensor):
        _collect(inner, f"{path}.data", num_envs, depth - 1, out)


def _sensor_buffers(env, num_envs: int) -> list[tuple[str, object]]:
    """Per-env buffers on every scene sensor's ``_data``.

    The ray casters hold the hits from their last cast.  After a pose copy those are the
    DESTINATION env's view of the world, not the source's, and nothing recomputes them until
    the sensor's own update period next elapses -- so the first observation a restored env
    produces carries someone else's depth image.  The correct reading after a restore IS the
    source's, because the two robots now stand in the same place.
    """
    out: list[tuple[str, object]] = []
    for name, sensor in getattr(env.scene, "sensors", {}).items():
        data = getattr(sensor, "_data", None)
        if data is None:
            continue
        out.extend(_derived_buffers(data, f"sensor.{name}", num_envs))
    return out

def _t(buf) -> torch.Tensor:
    """Convert a warp-or-torch articulation buffer to torch, as ``accumulate._to_torch``."""
    if isinstance(buf, torch.Tensor):
        return buf
    import warp as wp  # noqa: PLC0415

    return wp.to_torch(buf)


def _first_difference(a: list[str], b: list[str]) -> str:
    """The first index where two path lists differ, rendered for an error message."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return f"index {i}: {x!r} vs {y!r}"
    return f"length {len(a)} vs {len(b)}"


def _history_buffers(env) -> list[tuple[str, object]]:
    """Every observation-term history ring, as ``(path, CircularBuffer)``.

    ``CircularBuffer._pointer`` is a SCALAR shared by the whole batch, so a per-env copy of
    ``_buffer`` and ``_num_pushes`` is complete -- provided every env is at the same pointer,
    which holds because the manager appends to all of them on every step.  There is no
    per-env pointer to get wrong.
    """
    out: list[tuple[str, object]] = []
    per_group = getattr(env.observation_manager, "_group_obs_term_history_buffer", {})
    for group, terms in per_group.items():
        for term, buf in terms.items():
            out.append((f"obs.{group}.{term}", buf))
    return out


def _depth_noisers(env) -> list:
    """The live ``DepthNoiser`` instances, or an empty list when none is registered.

    Under ``dr_preset='none'`` the resample events are nulled and no noiser exists at all,
    which is the configuration this harness runs by default: its per-episode plane tilt and
    blur sigma are per-env draws, and a per-frame additive draw is not per-env addressable,
    so leaving them live would be an uncontrolled difference between candidates.  Collected
    anyway so that a ``dr_preset='train'`` arm copies what it can and the inventory says so.
    """
    from robot_rl.tasks.manager_based.robot_rl.mdp.sensors.noise import (  # noqa: PLC0415
        _ENV_ATTR,
    )

    registry = getattr(env, _ENV_ATTR, None)
    return list(registry.values()) if isinstance(registry, dict) else []
