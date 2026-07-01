# Copyright 2026 Zachary Olkin. All rights reserved.

"""Offline viser-based visualization for flow-matching recordings.

This subpackage is intentionally NOT re-exported from
``robot_rl.flow_matching`` so that importing the flow-matching core never
pulls in viser/yourdfpy. Import it explicitly::

    from robot_rl.flow_matching.viz import load_viz_recording, FlowMatchingViewer
"""

from .recording import (
    VIZ_DIR_DEFAULT,
    VizRecordingMeta,
    find_newest_recording,
    load_viz_recording,
    save_viz_recording,
    unique_output_path,
)

__all__ = [
    "VIZ_DIR_DEFAULT",
    "VizRecordingMeta",
    "find_newest_recording",
    "load_viz_recording",
    "save_viz_recording",
    "unique_output_path",
    "FlowMatchingViewer",
]


def __getattr__(name: str):
    """Lazily import the viewer so loading recordings never requires viser."""
    if name == "FlowMatchingViewer":
        from .viewer import FlowMatchingViewer

        return FlowMatchingViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
