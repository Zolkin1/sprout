# Copyright 2026 Zachary Olkin. All rights reserved.

"""Shared locomotion motion-DB builder used by EVERY clip generator.

Flat, stairs, boxes, and distractor generators all match against the SAME library so their clips are
mutually consistent: one union DB = flat-locomotion bones + clean(library_v2) steady strides + jump
lead-ins, tagged by source CATEGORY (``flat`` / ``clean`` / ``leadin``), with a single candidate graph
built over the union.  Each generator restricts what it may match per-generate via
``matcher.generate(..., allowed_categories=...)``:

* flat / stairs / distractors -> ``{flat, clean}`` (no lead-ins, so no elevated on-box frame leaks into
  flat ground),
* boxes / jump-on/off -> ``{flat, clean, leadin}`` (a jump must synthesise a takeoff from the lead-ins).

The clean library (library_v2) carries the full steady range -- forward / backward / running / turning
plus SIDESTEP (vel_y +-0.6) and DIAGONAL (walk/run with lateral) -- so every generator can hit a
sideways / diagonal command, not just forward.
"""

from __future__ import annotations

from .bones_ingest import load_cache
from .clean_ingest import _resolve_dir, ingest_clean_library
from .dp import DPConfig, DPMatcher
from .graph import GraphConfig, build_candidate_graph
from .vanilla import POSE_POSTURAL_IDX, VanillaConfig

# FLAT-locomotion bones filter: keep walking/jogging/running/turning on the feet; DROP everything that
# leaves flat ground (``climbing box`` / jumps / stairs / obstacles).  Backward + sidestep + turning are
# tagged walking/jogging (direction in the filename), so they SURVIVE -- the DB covers the full range.
_FLAT_MOVEMENT = ["walking", "jogging", "running", "running, turning", "turning",
                  "standing idle", "transition"]
_FLAT_EXCLUDE = ["injured", "limp", "drunk", "jump", "roll",
                 "stair", "box", "climb", "obstacle", "curb", "ledge", "ramp"]

# Jump/step LEAD-INS (elevated takeoff/landing frames): only the box/jump families may match these.
_TERRAIN_INCLUDE = ["jump_over_obstacle", "jump_on", "jump_off", "come_up", "come_down"]
_TERRAIN_EXCLUDE = ["ladder", "wall", "roll", "monkey", "shoulder", "safety", "hook"]

# Per family/tile TYPE: which source categories that family's DP may match.
APPROACH_FILTERS = {
    "flat": frozenset({"flat", "clean"}),
    "stair_transition": frozenset({"flat", "clean"}),
    "distractor": frozenset({"flat", "clean"}),
    "box": frozenset({"flat", "clean", "leadin"}),
}


def build_locomotion_matcher(*, bones_cache: str, clean_source: str, clean_hf_repo: str | None = "zolkin/robot_rl",
                             bones_limit: int | None = None, leadin_limit: int | None = None,
                             include_leadins: bool = True, dt: float = 0.02, device="cpu",
                             cut_penalty: float = 3.0, w_traj: float = 1.0, w_track: float = 0.0,
                             ground_penalty: float = 12.0, max_pelvis_z: float = 0.95,
                             graph_k_min: int = 25, graph_k_max: int = 50, graph_k_target: int = 40,
                             log=print) -> DPMatcher:
    """Build the shared locomotion :class:`DPMatcher` (union DB + candidate graph, categorised).

    ``include_leadins=False`` skips the lead-in load entirely (flat/stairs/distractor never need it);
    when ``True`` the lead-ins are in the graph but a caller still filters them out per-generate.  The
    DP cost weights (``cut_penalty`` / ``w_traj`` / ...) do not affect the DB or graph -- only matching
    at generate time -- so passing family-specific weights keeps the library identical across families.
    """
    log(f"[locomotion_db] flat-locomotion bones (limit={bones_limit}) ...")
    flat, _ = load_cache(bones_cache, filters={"package": "Locomotion",
                         "content_type_of_movement": _FLAT_MOVEMENT, "content_body_position": "standing"},
                         exclude_name_substrings=_FLAT_EXCLUDE, limit=bones_limit, device=device)
    log(f"[locomotion_db]   + {len(flat)} flat bones")
    log(f"[locomotion_db] clean library {clean_source} ...")
    clean = ingest_clean_library(clean_source, hf_repo=clean_hf_repo, dt=dt, device=device)
    log(f"[locomotion_db]   + {len(clean)} clean steady strides (fwd/back/run + sidestep/diagonal/turn)")
    leadins = []
    if include_leadins:
        leadins, _ = load_cache(bones_cache, filters={"package": "Locomotion"},
                                include_name_substrings=_TERRAIN_INCLUDE, exclude_name_substrings=_TERRAIN_EXCLUDE,
                                limit=leadin_limit, device=device)
        log(f"[locomotion_db]   + {len(leadins)} jump lead-ins")
    sources = flat + clean + leadins
    categories = ["flat"] * len(flat) + ["clean"] * len(clean) + ["leadin"] * len(leadins)
    dcfg = DPConfig(dt=dt, cut_penalty=cut_penalty, w_traj=w_traj, w_track=w_track,
                    w_ground=ground_penalty, ground_z=0.0, max_pelvis_z=max_pelvis_z)
    log(f"[locomotion_db] DP matcher DB: {len(sources)} clips, {sum(s.n for s in sources)} frames")
    matcher = DPMatcher(sources, VanillaConfig(dt=dt), dcfg, device=device, categories=categories)
    log("[locomotion_db] building candidate graph ...")
    cost_feat = matcher._norm[:, matcher._traj_dim:]
    select_feat = cost_feat[:, POSE_POSTURAL_IDX]
    graph = build_candidate_graph(
        select_feat, cost_feat, matcher.feat_raw, matcher.owner, matcher.src_start, matcher.src_len,
        matcher.yaw_all, GraphConfig(k_min=graph_k_min, k_max=graph_k_max, k_target=graph_k_target, dt=dt),
        steady_owner_mask=matcher._src_steady, periodic_owner_mask=matcher._src_periodic)
    matcher.attach_graph(graph)
    return matcher


def clean_library_fingerprint(clean_source: str, clean_hf_repo: str | None = "zolkin/robot_rl") -> dict:
    """Content fingerprint of the clean library (the shared clip source of truth) -- recorded on every
    generated clip library as ``source_library_fingerprint`` for provenance / drift detection.

    The clean library is npz + ``manifest.csv`` (not the traj-opt yaml layout ``fingerprint_library``
    expects), so hash the manifest content + the sorted per-clip file names & sizes."""
    import hashlib

    root = _resolve_dir(clean_source, clean_hf_repo)
    h = hashlib.sha256()
    manifest = root / "manifest.csv"
    if manifest.is_file():
        h.update(manifest.read_bytes())
    npzs = sorted(root.rglob("*.npz"))
    for f in npzs:
        h.update(str(f.relative_to(root)).encode())
        h.update(str(f.stat().st_size).encode())
    digest = h.hexdigest()
    return {"hash": digest, "short": digest[:12], "num_clips": len(npzs), "source": str(clean_source)}
