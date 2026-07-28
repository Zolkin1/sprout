# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command configurations for the environment.

The only command family left is the offline clip tracker under ``traj_tracking`` --
:class:`~.traj_tracking.motion_sequence_cmd_cfg.MotionSequenceCommandCfg` (open-loop clip
playback) and its flow-matching subclass
:class:`~.traj_tracking.fm_tracking_cmd_cfg.FmTrackingCommandCfg`.  Both are imported
directly from their leaf module by the env cfgs, so nothing is re-exported here.

The velocity-conditioned command terms that used to live here (``VelocityTrackingCommandCfg``,
``MultiskillVelocityTrackingCommandCfg``, ``VelocityBucketCfg``) belonged to the online,
in-training reference generator and went with it.
"""
