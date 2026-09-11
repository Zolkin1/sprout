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

from ..ingest.bones_ingest import load_cache
from ..ingest.clean_ingest import _resolve_dir, ingest_clean_library
from .dp import DPConfig, DPMatcher
from .graph import _GRAPH_CACHE_VERSION, CandidateGraph, GraphConfig, build_candidate_graph, graph_cache_path
from .vanilla import POSE_POSTURAL_IDX, VanillaConfig, VanillaMatcher

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
                             use_graph_cache: bool = True, matcher_kind: str = "dp",
                             log=print) -> DPMatcher | VanillaMatcher:
    """Build the shared locomotion :class:`DPMatcher` (union DB + candidate graph, categorised).

    ``include_leadins=False`` skips the lead-in load entirely (flat/stairs/distractor never need it);
    when ``True`` the lead-ins are in the graph but a caller still filters them out per-generate.  The
    DP cost weights (``cut_penalty`` / ``w_traj`` / ...) do not affect the DB or graph -- only matching
    at generate time -- so passing family-specific weights keeps the library identical across families.

    ``matcher_kind`` picks the SEARCH over that one DB.  ``"dp"`` (default) is the exact backward
    dynamic program every live generator uses.  ``"vanilla"`` is the greedy local-lookahead matcher
    -- the standard real-time motion-matching baseline -- and is offered for the paper ablation, so
    both arms are scored over an identical DB and an identical feature space (both take the same
    ``VanillaConfig``) and differ ONLY in how the path through it is chosen.  Vanilla needs the
    full-feature KD-tree the DP skips, and uses NO candidate graph, so that build is skipped too.

    **Vanilla cannot honour a per-generate category filter** -- ``allowed_categories`` lives on
    :class:`DPMatcher`, not on the base class -- so it is refused whenever the DB contains lead-ins,
    i.e. whenever a filter would have anything to do.  On a ``{flat, clean}`` DB every
    :data:`APPROACH_FILTERS` entry that may be used is the whole DB, so dropping the filter there is
    provably a no-op rather than a silently unfiltered match.

    ``use_graph_cache`` (default on) content-addresses the candidate graph to
    ``graph_cache_path(...)``: the DB build is deterministic given (bones cache, filters, limits, clean
    lib, lead-in flag), so the first family to run a given DB variant builds the graph and every later
    family loads it instead of rebuilding -- the two variants (with / without lead-ins) get distinct
    keys and never collide.  The matcher itself is always rebuilt (cheap next to the graph).
    """
    if matcher_kind not in ("dp", "vanilla"):
        raise ValueError(f"matcher_kind must be 'dp' or 'vanilla', got {matcher_kind!r}.")
    if matcher_kind == "vanilla" and include_leadins:
        raise ValueError(
            "matcher_kind='vanilla' with include_leadins=True: the greedy matcher has no "
            "allowed_categories, so the lead-ins it must not match could not be filtered out and "
            "an elevated on-box frame would leak onto flat ground. Vanilla is only available on a "
            "{flat, clean} DB (include_leadins=False), where no filter has anything to do.")
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
    log(f"[locomotion_db] {matcher_kind} matcher DB: {len(sources)} clips, "
        f"{sum(s.n for s in sources)} frames")
    if matcher_kind == "vanilla":
        # Greedy arm: the same sources and the same VanillaConfig feature space as the DP arm, but
        # searched with the KD-tree instead of the graph + backward DP.  No graph is built at all.
        log(f"[locomotion_db] building vanilla full-feature KD-tree ({sum(s.n for s in sources)} frames) ...")
        return VanillaMatcher(sources, VanillaConfig(dt=dt), device=device, build_tree=True)

    dcfg = DPConfig(dt=dt, cut_penalty=cut_penalty, w_traj=w_traj, w_track=w_track,
                    w_ground=ground_penalty, ground_z=0.0, max_pelvis_z=max_pelvis_z)
    matcher = DPMatcher(sources, VanillaConfig(dt=dt), dcfg, device=device, categories=categories)

    cfg = GraphConfig(k_min=graph_k_min, k_max=graph_k_max, k_target=graph_k_target, dt=dt)
    graph, cache_key = None, None
    if use_graph_cache:
        cache_key = _graph_cache_key(bones_cache, bones_limit, include_leadins, leadin_limit,
                                     clean_library_fingerprint(clean_source, clean_hf_repo)["hash"], cfg)
        cache_file = graph_cache_path(cache_key)
        graph = CandidateGraph.load(cache_file, expected_key=cache_key)
        if graph is not None:
            log(f"[locomotion_db] loaded cached candidate graph {cache_file.name}")
    if graph is None:
        log("[locomotion_db] building candidate graph ...")
        cost_feat = matcher._norm[:, matcher._traj_dim:]
        select_feat = cost_feat[:, POSE_POSTURAL_IDX]
        graph = build_candidate_graph(
            select_feat, cost_feat, matcher.feat_raw, matcher.owner, matcher.src_start, matcher.src_len,
            matcher.yaw_all, cfg, steady_owner_mask=matcher._src_steady, periodic_owner_mask=matcher._src_periodic)
        if use_graph_cache:
            graph.key = cache_key
            graph.save(cache_file)
            log(f"[locomotion_db] cached candidate graph -> {cache_file}")
    matcher.attach_graph(graph)
    return matcher


def _graph_cache_key(bones_cache: str, bones_limit: int | None, include_leadins: bool,
                     leadin_limit: int | None, clean_hash: str, cfg: GraphConfig) -> str:
    """Content hash of EVERYTHING the candidate graph depends on: the bones cache file identity, the
    flat + lead-in filter sets and limits, the clean library content, and the graph params + version.

    The lead-in flag/filter/limit are folded in so the with-lead-ins graph (box/jump) and the
    without-lead-ins graph (flat/distractor/stairs) get DISTINCT keys and never load each other's."""
    import hashlib
    import json
    from pathlib import Path

    st = Path(bones_cache).stat()
    payload = {
        "version": _GRAPH_CACHE_VERSION,
        "bones": [str(Path(bones_cache)), st.st_size, st.st_mtime_ns],
        "flat_movement": sorted(_FLAT_MOVEMENT), "flat_exclude": sorted(_FLAT_EXCLUDE),
        "bones_limit": bones_limit,
        "clean_hash": clean_hash,
        "include_leadins": bool(include_leadins),
        "leadin_include": sorted(_TERRAIN_INCLUDE), "leadin_exclude": sorted(_TERRAIN_EXCLUDE),
        "leadin_limit": leadin_limit,
        "cfg": {"k_min": cfg.k_min, "k_max": cfg.k_max, "k_target": cfg.k_target,
                "same_clip_cap": cfg.same_clip_cap, "min_same_clip_gap": cfg.min_same_clip_gap,
                "dt": round(float(cfg.dt), 6), "tier_mode": cfg.tier_mode},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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
