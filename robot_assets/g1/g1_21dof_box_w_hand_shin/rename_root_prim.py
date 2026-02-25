"""Rename /g1_.../pelvis to /g1_.../pelvis_link in the USD file."""

from pxr import Usd, Sdf

USD_FILE = "g1_21dof_with_hand_shin.usd"

stage = Usd.Stage.Open(USD_FILE)
old_path = Sdf.Path("/g1_29dof_lock_waist_with_hand_rev_1_0/pelvis")
new_path = Sdf.Path("/g1_29dof_lock_waist_with_hand_rev_1_0/pelvis_link")

edit = Sdf.BatchNamespaceEdit()
edit.Add(old_path, new_path)

if not stage.GetRootLayer().Apply(edit):
    print("ERROR: Failed to apply namespace edit")
else:
    stage.GetRootLayer().Save()
    print(f"Successfully renamed pelvis -> pelvis_link in {USD_FILE}")
