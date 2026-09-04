# Copyright 2026 Zachary Olkin. All rights reserved.

"""Import shim for the export script, which lives under ``scripts/`` and is not a package.

``scripts/token_planner/export_planner_onnx.py`` is an entry point, not an installed module,
so a test cannot ``import`` it by name. The flow-matching tests solve the same problem the
same way (``test_train_cli_defaults`` loads its script by path); this keeps the path
derivation in one place instead of in every test that needs it.
"""

from __future__ import annotations

import importlib.util
import sys

from motion_gen.features import _REPO_ROOT

_PATH = _REPO_ROOT / "scripts" / "token_planner" / "export_planner_onnx.py"
_spec = importlib.util.spec_from_file_location("_tp_export_under_test", _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["_tp_export_under_test"] = _mod
_spec.loader.exec_module(_mod)

export = _mod.export
_PlannerSamplerOnnx = _mod._PlannerSamplerOnnx
_IO_NAMES = _mod._IO_NAMES
