# Copyright 2026 Zachary Olkin. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
import warp as wp

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.envs.mdp.events import _validate_scale_range, _randomize_prop_by_op
from isaaclab.envs.mdp.events import randomize_rigid_body_material as _randomize_rigid_body_material_base
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv

class randomize_joint_parameters_multi_friction(ManagerTermBase):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            TypeError: If `params` is not a tuple of two numbers.
            ValueError: If the operation is not supported.
            ValueError: If the lower bound is negative or zero when not allowed.
            ValueError: If the upper bound is less than the lower bound.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]
        # check for valid operation
        if cfg.params["operation"] == "scale":
            if "friction_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["friction_distribution_params"], "friction_distribution_params")
            if "armature_distribution_params" in cfg.params:
                _validate_scale_range(cfg.params["armature_distribution_params"], "armature_distribution_params")
        elif cfg.params["operation"] not in ("abs", "add"):
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' does not support operation:"
                f" '{cfg.params['operation']}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        static_friction_distribution_params: tuple[float, float] | None = None,
        dynamic_friction_distribution_params: tuple[float, float] | None = None,
        viscous_friction_distribution_params: tuple[float, float] | None = None,
        armature_distribution_params: tuple[float, float] | None = None,
        lower_limit_distribution_params: tuple[float, float] | None = None,
        upper_limit_distribution_params: tuple[float, float] | None = None,
        operation: Literal["add", "scale", "abs"] = "abs",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.asset.device)

        # resolve joint indices
        if self.asset_cfg.joint_ids == slice(None):
            joint_ids = slice(None)  # for optimization purposes
        else:
            joint_ids = torch.tensor(self.asset_cfg.joint_ids, dtype=torch.int, device=self.asset.device)

        # sample joint properties from the given ranges and set into the physics simulation
        # joint friction coefficient
        if static_friction_distribution_params and dynamic_friction_distribution_params and viscous_friction_distribution_params is not None:
            friction_coeff = _randomize_prop_by_op(
                wp.to_torch(self.asset.data.default_joint_friction_coeff).clone(),
                static_friction_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

            # ensure the friction coefficient is non-negative
            friction_coeff = torch.clamp(friction_coeff, min=0.0)

            # Always set static friction (indexed once)
            static_friction_coeff = friction_coeff[env_ids[:, None], joint_ids]

            # # TODO: Add back in the dynamic and viscous friction when the feature is added back.
            # # Randomize raw tensors
            # dynamic_friction_coeff = _randomize_prop_by_op(
            #     wp.to_torch(self.asset.data.default_joint_dynamic_friction_coeff).clone(),
            #     dynamic_friction_distribution_params,
            #     env_ids,
            #     joint_ids,
            #     operation=operation,
            #     distribution=distribution,
            # )
            #
            # viscous_friction_coeff = _randomize_prop_by_op(
            #     wp.to_torch(self.asset.data.default_joint_viscous_friction_coeff).clone(),
            #     viscous_friction_distribution_params,
            #     env_ids,
            #     joint_ids,
            #     operation=operation,
            #     distribution=distribution,
            # )
            #
            # # Clamp to non-negative
            # dynamic_friction_coeff = torch.clamp(dynamic_friction_coeff, min=0.0)
            # viscous_friction_coeff = torch.clamp(viscous_friction_coeff, min=0.0)
            #
            # # Ensure dynamic ≤ static (same shape before indexing)
            # dynamic_friction_coeff = torch.minimum(dynamic_friction_coeff, friction_coeff)
            #
            # # Index once at the end
            # dynamic_friction_coeff = dynamic_friction_coeff[env_ids[:, None], joint_ids]
            # viscous_friction_coeff = viscous_friction_coeff[env_ids[:, None], joint_ids]

            # Single write call for all versions
            self.asset.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=static_friction_coeff,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )

        # joint armature
        if armature_distribution_params is not None:
            armature = _randomize_prop_by_op(
                wp.to_torch(self.asset.data.default_joint_armature).clone(),
                armature_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
            self.asset.write_joint_armature_to_sim_index(
                armature=armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
            )

        # joint position limits
        if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
            joint_pos_limits = wp.to_torch(self.asset.data.default_joint_pos_limits).clone()
            # -- randomize the lower limits
            if lower_limit_distribution_params is not None:
                joint_pos_limits[..., 0] = _randomize_prop_by_op(
                    joint_pos_limits[..., 0],
                    lower_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )
            # -- randomize the upper limits
            if upper_limit_distribution_params is not None:
                joint_pos_limits[..., 1] = _randomize_prop_by_op(
                    joint_pos_limits[..., 1],
                    upper_limit_distribution_params,
                    env_ids,
                    joint_ids,
                    operation=operation,
                    distribution=distribution,
                )

            # extract the position limits for the concerned joints
            joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
            if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
                raise ValueError(
                    "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater"
                    " than upper joint limits. Please check the distribution parameters for the joint position limits."
                )
            # set the position limits into the physics simulation
            self.asset.write_joint_position_limit_to_sim_index(
                joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
            )

def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = wp.to_torch(asset.data.default_joint_pos)[0].clone()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = wp.to_torch(asset.data.default_joint_pos).to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        wp.to_torch(asset.data.default_joint_pos)[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos

##
## NOTE: This is a fix for a bug in IsaacLab where they don't handle the types correctly.
##
class randomize_rigid_body_material(_randomize_rigid_body_material_base):
    """Wrapper around IsaacLab's randomize_rigid_body_material that fixes env_ids dtype.

    IsaacLab has a bug where env_ids is not cast to int32 before being passed to
    warp's from_torch, which fails when env_ids is int64.
    """

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        """Cast env_ids to int32 before calling the parent implementation."""
        if env_ids is not None:
            env_ids = env_ids.to(dtype=torch.int32)
        super().__call__(
            env, env_ids,
            static_friction_range=static_friction_range,
            dynamic_friction_range=dynamic_friction_range,
            restitution_range=restitution_range,
            num_buckets=num_buckets,
            asset_cfg=asset_cfg,
            make_consistent=make_consistent,
        )


class add_payload_mass(ManagerTermBase):
    """Attach a randomized payload (a point mass at a random position) to a body on a random
    subset of the environments.

    Models a rigid payload -- e.g. a load carried on the robot's back while it bends over --
    rigidly attached to ``asset_cfg``'s (single) body at a per-env random position ``r_p``
    (expressed in that body's link frame) with a per-env random mass ``m_p``.  Only a random
    ``fraction`` of environments receive a payload; the rest are left untouched, so the policy
    trains on both loaded and unloaded robots.

    For every selected env, given the body's current mass ``m_l`` and COM ``c_l`` (link frame),
    the attached point mass shifts the *combined* rigid-body properties to::

        m = m_l + m_p                                   # total mass
        c = (m_l * c_l + m_p * r_p) / (m_l + m_p)       # combined COM (link frame)

    The inertia is scaled by the mass ratio ``m / m_l`` (the same uniform-density approximation
    used by IsaacLab's :class:`randomize_rigid_body_mass`).  This intentionally does *not* add
    the payload's parallel-axis term: in this Newton backend ``body_inertia`` is stored in the
    per-body *principal-axes* frame, so a body-frame parallel-axis correction would be
    frame-inconsistent.  The dominant effect for a balance/bend task -- the COM shift -- is
    modelled exactly; the offset's contribution to rotational inertia is approximated.

    This is a ``startup`` term (apply once).  When ``debug_vis`` is enabled it also draws a red
    sphere at each payload's live world position (following the body as it moves), sized by the
    payload mass, so the placement is visible in playback.

    Args:
        asset_cfg: The articulation and the (single) body the payload attaches to.
        mass_range: ``(min, max)`` payload mass in kg, sampled uniformly per env.
        position_range: Dict of ``{"x"/"y"/"z": (min, max)}`` giving the box (link frame) the
            payload position is sampled uniformly within.  Missing axes default to ``(0, 0)``.
        fraction: Fraction of environments that receive a payload (``0.5`` == half).
        debug_vis: If True, draw a marker sphere at each payload position.
        marker_prim_path: USD prim path for the debug markers.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        fraction = float(cfg.params.get("fraction", 0.5))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"add_payload_mass 'fraction' must be in [0, 1], got {fraction}.")
        mass_range = cfg.params["mass_range"]
        if mass_range[0] < 0.0 or mass_range[1] < mass_range[0]:
            raise ValueError(f"add_payload_mass 'mass_range' must be a non-negative (min, max), got {mass_range}.")

        # Debug-visualization state.  The marker prim is created lazily on the first
        # render callback -- never during the startup event -- because mutating the USD
        # stage mid-initialization can deadlock the sim.
        self._payload_markers = None
        self._debug_vis_handle = None
        self._debug_vis_pending = False
        self._marker_prim_path: str | None = None
        self._marker_mass_ref: float = 1.0
        self._marker_base_radius: float = 0.05
        self._payload_env_ids: torch.Tensor | None = None
        self._payload_pos_b: torch.Tensor | None = None
        self._payload_mass: torch.Tensor | None = None
        self._payload_body_id: int | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        mass_range: tuple[float, float] = (5.0, 7.0),
        position_range: dict[str, tuple[float, float]] | None = None,
        fraction: float = 0.5,
        debug_vis: bool = False,
        marker_prim_path: str = "/Visuals/BackPayload",
    ):
        device = self.asset.device
        # startup term -> env_ids is normally all envs; be robust to None.
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=device)
        else:
            env_ids = env_ids.to(device)

        # single target body (the attachment link, e.g. the torso).
        if self.asset_cfg.body_ids == slice(None):
            raise ValueError("add_payload_mass requires asset_cfg to select a single body (e.g. the torso).")
        body_id = int(self.asset_cfg.body_ids[0])
        self._payload_body_id = body_id

        # pick the random subset of envs that carry a payload.
        n = env_ids.numel()
        n_sel = int(round(fraction * n))
        if n_sel == 0:
            return
        # int32 for the warp-backed setters (see randomize_rigid_body_material note above).
        sel = env_ids[torch.randperm(n, device=device)[:n_sel]].to(torch.int32)

        # sample payload mass and position (link frame) within the box.
        pr = position_range or {}
        lo = torch.tensor([pr.get(k, (0.0, 0.0))[0] for k in ("x", "y", "z")], device=device)
        hi = torch.tensor([pr.get(k, (0.0, 0.0))[1] for k in ("x", "y", "z")], device=device)
        r_p = lo + (hi - lo) * torch.rand((n_sel, 3), device=device)                    # (n_sel, 3)
        m_p = mass_range[0] + (mass_range[1] - mass_range[0]) * torch.rand(n_sel, device=device)  # (n_sel,)

        # current (live) rigid-body properties of the target body for the selected envs.
        masses = wp.to_torch(self.asset.data.body_mass).clone()                         # (num_envs, num_bodies)
        coms = wp.to_torch(self.asset.data.body_com_pose_b).clone()                      # (num_envs, num_bodies, >=7)
        inertias = wp.to_torch(self.asset.data.body_inertia).clone()                     # (num_envs, num_bodies, 9)

        m_l = masses[sel, body_id]                                                       # (n_sel,)
        c_l = coms[sel, body_id, :3]                                                     # (n_sel, 3)
        m_new = m_l + m_p                                                                # (n_sel,)
        c_new = (m_l.unsqueeze(-1) * c_l + m_p.unsqueeze(-1) * r_p) / m_new.unsqueeze(-1)

        # write back: mass, COM, and mass-ratio-scaled inertia.
        masses[sel, body_id] = m_new
        coms[sel, body_id, :3] = c_new
        inertias[sel, body_id] = inertias[sel, body_id] * (m_new / m_l).unsqueeze(-1)

        body_ids = torch.tensor([body_id], dtype=torch.int32, device=device)
        self.asset.set_masses_index(masses=masses[sel][:, [body_id]], body_ids=body_ids, env_ids=sel)
        self.asset.set_coms_index(coms=coms[sel], env_ids=sel)
        self.asset.set_inertias_index(inertias=inertias[sel][:, [body_id]], body_ids=body_ids, env_ids=sel)

        # remember placement so the debug markers can follow the body each frame.
        self._payload_env_ids = sel
        self._payload_pos_b = r_p
        self._payload_mass = m_p

        if debug_vis:
            self._marker_prim_path = marker_prim_path
            self._marker_mass_ref = 0.5 * (mass_range[0] + mass_range[1])
            self._enable_debug_vis()

    # ------------------------------------------------------------------
    # Debug visualization -- draw a sphere at each payload's world position.
    # ------------------------------------------------------------------
    def _enable_debug_vis(self) -> None:
        """Subscribe to the render loop.  The marker prim is created lazily on the first
        callback (during the running sim), NOT here: creating it now -- inside the startup
        event, mid env-init -- mutates the USD stage and can deadlock the sim.  Mirrors
        IsaacLab's ``CommandTerm`` debug-vis pattern.
        """
        if self._debug_vis_handle is not None:
            return
        try:
            import omni.kit.app
            import weakref
        except Exception as exc:  # headless / no app -> skip viz entirely.
            print(f"[WARN add_payload_mass] debug viz unavailable: {exc}")
            return
        self._debug_vis_pending = True
        app_interface = omni.kit.app.get_app_interface()
        self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
            lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
        )

    def _create_markers(self) -> None:
        """Build the sphere :class:`VisualizationMarkers` (lazy, called from the callback)."""
        try:
            import isaaclab.sim as sim_utils
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

            cfg = VisualizationMarkersCfg(
                prim_path=self._marker_prim_path,
                markers={
                    "payload": sim_utils.SphereCfg(
                        radius=self._marker_base_radius,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                    )
                },
            )
            self._payload_markers = VisualizationMarkers(cfg)
        except Exception as exc:
            print(f"[WARN add_payload_mass] could not create payload markers, disabling viz: {exc}")
            self._payload_markers = None
            self._disable_debug_vis()

    def _disable_debug_vis(self) -> None:
        """Tear down the render subscription so a failed / finished viz never keeps firing."""
        self._debug_vis_pending = False
        if self._debug_vis_handle is not None:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    def _debug_vis_callback(self, event) -> None:
        # Guard the whole callback: an exception here fires every render frame, so on any
        # failure we disable the viz rather than spam / stall the render loop.
        try:
            if self._payload_markers is None:
                if not self._debug_vis_pending:
                    return
                self._debug_vis_pending = False
                self._create_markers()
                if self._payload_markers is None:
                    return
            sel = self._payload_env_ids
            if sel is None or sel.numel() == 0:
                return
            body_id = self._payload_body_id
            # live world pose of the attachment body -> lift the link-frame payload offset to world.
            body_pos_w = wp.to_torch(self.asset.data.body_pos_w)[sel, body_id]          # (n_sel, 3)
            body_quat_w = wp.to_torch(self.asset.data.body_quat_w)[sel, body_id]        # (n_sel, 4) xyzw
            world_pos = body_pos_w + quat_apply(body_quat_w, self._payload_pos_b)
            # size each sphere by its payload mass (heavier -> bigger).
            scale = (self._payload_mass / self._marker_mass_ref).unsqueeze(-1).expand(-1, 3)
            self._payload_markers.visualize(translations=world_pos, scales=scale)
        except Exception as exc:
            print(f"[WARN add_payload_mass] debug viz callback failed, disabling: {exc}")
            self._payload_markers = None
            self._disable_debug_vis()