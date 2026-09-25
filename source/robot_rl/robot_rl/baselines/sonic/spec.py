# Copyright 2026 Zachary Olkin. All rights reserved.

"""The SONIC observation contract: names -> dimensions -> offsets, for both ONNX graphs.

NVIDIA ship the released policy as two ONNX graphs plus an ``observation_config.yaml`` that
lists observation names in concatenation order but carries **no dimensions** -- those live in
the C++ deploy reference's ``GetObservationRegistry()``
(``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp``, ~line 1704).
This module ports that registry, resolves the YAML into concrete ``[start, stop)`` slices, and
refuses to run unless the totals match what the graphs declare.

Dimensions are *derived*, not tabulated: every multi-frame name follows
``{base}_{N}frame_step{S}`` and is ``N * dim(base)`` wide, which reproduces every number in the
upstream C++ table from a short per-base table. That keeps a variant (e.g. the low-latency
checkpoint, which uses ``step1`` and 4-frame SMPL) working without a new hand-written column,
and the ONNX cross-check in :meth:`SonicSpec.validate_dims` is what actually guarantees it.

This module is deliberately free of torch, isaaclab and onnxruntime so the layout can be
unit-tested headless and without the third-party weights present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

# Per-base observation widths, ported from the upstream C++ registry. A ``{base}_{N}frame_step{S}``
# name is ``N *`` the base width; the bare name is the single-frame case. ``token_state`` is the
# one exception -- its width is the encoder's latent dimension, read from the YAML.
_BASE_DIM: dict[str, int] = {
    # encoder mode selector: the mode id sits in element 0, the rest is zero padding
    "encoder_mode": 3,
    "encoder_mode_4": 4,
    # G1 motion reference
    "motion_joint_positions": 29,
    "motion_joint_velocities": 29,
    "motion_joint_positions_lowerbody": 12,
    "motion_joint_velocities_lowerbody": 12,
    "motion_joint_positions_wrists": 6,
    "motion_joint_velocities_wrists": 6,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_heading": 6,
    "motion_anchor_orientation_refheading": 6,
    "heading_diff_robot_ref": 6,
    # SMPL reference
    "smpl_joints": 72,
    "smpl_joints_lower": 27,
    "smpl_pose": 63,
    "smpl_root_z": 1,
    "smpl_anchor_orientation": 6,
    "smpl_anchor_orientation_heading": 6,
    "smpl_anchor_orientation_refheading": 6,
    # VR teleop
    "vr_3point_local_target": 9,
    "vr_3point_local_target_compliant": 9,
    "vr_3point_local_orn_target": 12,
    "vr_3point_compliance": 3,
    "vr_5point_local_target": 15,
    "vr_5point_local_orn_target": 20,
    # robot state history (the ``his_`` prefix is upstream's; the bare names are single-frame)
    "base_angular_velocity": 3,
    "body_joint_positions": 29,
    "body_joint_velocities": 29,
    "last_actions": 29,
    "gravity_dir": 3,
    "his_base_angular_velocity": 3,
    "his_body_joint_positions": 29,
    "his_body_joint_velocities": 29,
    "his_last_actions": 29,
    "his_gravity_dir": 3,
}

_MULTI_FRAME = re.compile(r"^(?P<base>.+)_(?P<frames>\d+)frame_step(?P<step>\d+)$")

TOKEN_STATE = "token_state"


@dataclass(frozen=True)
class Block:
    """One observation term's slice of a concatenated ONNX input vector.

    Attributes:
        name: The upstream observation name, verbatim from ``observation_config.yaml``.
        offset: Start index into the flat input vector.
        dim: Width in scalars.
        frames: Number of stacked frames (1 for a single-frame term).
        step: Frame stride in control steps (1 for a single-frame term).
    """

    name: str
    offset: int
    dim: int
    frames: int
    step: int

    @property
    def stop(self) -> int:
        """One past the last index of this block."""
        return self.offset + self.dim

    @property
    def base(self) -> str:
        """The name with any ``_{N}frame_step{S}`` suffix removed."""
        m = _MULTI_FRAME.match(self.name)
        return m.group("base") if m else self.name

    @property
    def per_frame(self) -> int:
        """Width of a single frame of this block."""
        return self.dim // self.frames


@dataclass(frozen=True)
class Mode:
    """An encoder mode: which id selects it and which observations it actually consumes.

    Everything not in :attr:`required` is zero-filled. That is what the upstream C++ does, and
    it is verified empirically -- perturbing a non-required slice with the mode id pinned leaves
    the 64-d token bit-identical (see ``tests/test_sonic_obs.py::test_unused_slices_are_inert``).
    """

    name: str
    mode_id: int
    required: frozenset[str]


def _resolve_dim(name: str, token_dim: int) -> tuple[int, int, int]:
    """Resolve one observation name to ``(dim, frames, step)``.

    Args:
        name: Observation name from ``observation_config.yaml``.
        token_dim: The encoder latent width, used for ``token_state``.

    Returns:
        Total width, number of stacked frames, and the frame stride.

    Raises:
        NotImplementedError: If the name is not in the ported registry. Failing loudly beats
            guessing a width and silently shifting every later block.
    """
    if name == TOKEN_STATE:
        return token_dim, 1, 1
    m = _MULTI_FRAME.match(name)
    base, frames, step = (m.group("base"), int(m.group("frames")), int(m.group("step"))) if m \
        else (name, 1, 1)
    if base not in _BASE_DIM:
        raise NotImplementedError(
            f"SONIC observation '{name}' (base '{base}') is not in the ported registry. Add its "
            "width to robot_rl.baselines.sonic.spec._BASE_DIM, taking the value from "
            "GetObservationRegistry() in gear_sonic_deploy/.../g1_deploy_onnx_ref.cpp."
        )
    return _BASE_DIM[base] * frames, frames, step


def _lay_out(names: list[str], token_dim: int) -> tuple[Block, ...]:
    """Turn an ordered list of observation names into contiguous blocks."""
    blocks, offset = [], 0
    for name in names:
        dim, frames, step = _resolve_dim(name, token_dim)
        blocks.append(Block(name=name, offset=offset, dim=dim, frames=frames, step=step))
        offset += dim
    return tuple(blocks)


@dataclass(frozen=True)
class SonicSpec:
    """The full input contract for one released SONIC checkpoint.

    Attributes:
        decoder_blocks: Layout of the decoder/policy input, ``token_state`` first.
        encoder_blocks: Layout of the encoder input. Element 0 of the first block is the mode id.
        modes: Encoder modes by name.
        token_dim: Encoder latent width (64 for every released checkpoint so far).
        path: The ``observation_config.yaml`` this came from, for error messages.
    """

    decoder_blocks: tuple[Block, ...]
    encoder_blocks: tuple[Block, ...]
    modes: dict[str, Mode]
    token_dim: int
    path: Path

    @property
    def encoder_dim(self) -> int:
        """Total encoder input width."""
        return self.encoder_blocks[-1].stop if self.encoder_blocks else 0

    @property
    def decoder_dim(self) -> int:
        """Total decoder input width."""
        return self.decoder_blocks[-1].stop if self.decoder_blocks else 0

    def encoder_block(self, name: str) -> Block | None:
        """The encoder block called ``name``, or None if this checkpoint has no such input."""
        return next((b for b in self.encoder_blocks if b.name == name), None)

    def decoder_block(self, name: str) -> Block | None:
        """The decoder block called ``name``, or None if this checkpoint has no such input."""
        return next((b for b in self.decoder_blocks if b.name == name), None)

    def mode_blocks(self, mode: str) -> tuple[Block, ...]:
        """Encoder blocks that ``mode`` consumes, in layout order.

        Every other encoder block must be left at zero.
        """
        required = self.modes[mode].required
        return tuple(b for b in self.encoder_blocks if b.name in required)

    def reference_frames(self, mode: str = "g1") -> tuple[int, int]:
        """The ``(frames, step)`` the mode's motion-reference blocks are sampled at.

        Returns:
            Number of future reference frames and their stride in control steps -- ``(10, 5)``
            for the default release, ``(10, 1)`` for the low-latency one.

        Raises:
            ValueError: If the mode's multi-frame reference blocks disagree on ``(frames, step)``.
                They are read from one shared future window, so a disagreement would silently
                sample some blocks at the wrong times.
        """
        shapes = {(b.frames, b.step) for b in self.mode_blocks(mode)
                  if b.frames > 1 and b.base.startswith("motion_")}
        if not shapes:
            raise ValueError(f"mode '{mode}' has no multi-frame motion-reference blocks in "
                             f"{self.path}; nothing to build a reference window from.")
        if len(shapes) > 1:
            raise ValueError(
                f"mode '{mode}' mixes reference samplings {sorted(shapes)} in {self.path}. One "
                "shared future window cannot serve both; teach the obs builder to request a "
                "window per sampling before using this checkpoint."
            )
        return next(iter(shapes))

    def validate_dims(self, encoder_dim: int, decoder_dim: int) -> None:
        """Check the resolved layout against what the ONNX graphs declare.

        This is the guard that makes the derived widths trustworthy: a wrong per-base width or a
        config listing terms the graph was not exported with shows up here as a total mismatch,
        rather than as a policy that silently reads the wrong slice and falls over.

        Args:
            encoder_dim: Encoder graph's declared input width.
            decoder_dim: Decoder graph's declared input width.

        Raises:
            ValueError: On any mismatch, with the per-block breakdown.
        """
        for what, want, got, blocks in (
            ("encoder", encoder_dim, self.encoder_dim, self.encoder_blocks),
            ("decoder", decoder_dim, self.decoder_dim, self.decoder_blocks),
        ):
            if want != got:
                table = "\n".join(f"    {b.offset:5d}:{b.stop:<5d} {b.dim:5d}  {b.name}"
                                  for b in blocks)
                raise ValueError(
                    f"SONIC {what} layout from {self.path} sums to {got} but the ONNX graph "
                    f"declares {want}. Either the config does not belong to this checkpoint or a "
                    f"width in spec._BASE_DIM is wrong. Resolved layout:\n{table}"
                )


def load_spec(config_path: str | Path) -> SonicSpec:
    """Parse a SONIC ``observation_config.yaml`` into a :class:`SonicSpec`.

    Args:
        config_path: Path to the ``observation_config.yaml`` shipped beside the ONNX pair.

    Returns:
        The resolved layout. Call :meth:`SonicSpec.validate_dims` against the loaded graphs
        before trusting it.

    Raises:
        FileNotFoundError: If the config is missing.
        ValueError: If the config has no enabled observations or no encoder section while
            ``token_state`` is enabled (upstream's own validation rule).
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"SONIC observation config not found at {path}. Fetch the release with "
            "'python download_from_hf.py' in the GR00T-WholeBodyControl checkout."
        )
    cfg = yaml.safe_load(path.read_text()) or {}
    dec_names = [o["name"] for o in cfg.get("observations", []) if o.get("enabled")]
    if not dec_names:
        raise ValueError(f"{path} enables no policy observations.")

    enc_cfg = cfg.get("encoder") or {}
    token_dim = int(enc_cfg.get("dimension", 0))
    if TOKEN_STATE in dec_names and token_dim <= 0:
        raise ValueError(f"{path} enables '{TOKEN_STATE}' but has no encoder.dimension > 0.")
    enc_names = [o["name"] for o in enc_cfg.get("encoder_observations", []) if o.get("enabled")]
    modes = {
        m["name"]: Mode(name=m["name"], mode_id=int(m["mode_id"]),
                        required=frozenset(m.get("required_observations", [])))
        for m in enc_cfg.get("encoder_modes", [])
    }
    return SonicSpec(
        decoder_blocks=_lay_out(dec_names, token_dim),
        encoder_blocks=_lay_out(enc_names, token_dim),
        modes=modes,
        token_dim=token_dim,
        path=path,
    )
