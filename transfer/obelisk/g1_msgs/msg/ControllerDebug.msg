# Copyright 2026 Zachary Olkin. All rights reserved.
#
# One per 50 Hz control tick: everything the controller computes and then throws away.
#
# Deliberately NOT the assembled observation vector. Every term in it is either already on
# the ROS network (/obelisk/g1/est_state carries the proprioception, the two
# /depth/<cam>/image_processed topics the exteroception) or reconstructible from what is
# here, so republishing the concatenated vector would be ~3100 duplicated floats a tick.
# What IS here is only what has no other route out of the process -- plus `obs_checksum`,
# which lets an offline reconstruction PROVE it rebuilt the vector the network actually saw.
#
# NOTE: rosidl copies these comments VERBATIM into generated C block comments, so a literal
# star-slash anywhere in this file terminates that comment early and the rest of the text
# is compiled as code. Write topic globs as <cam>, never as a star. Pinned by
# tests/test_msg_comments.py, because the resulting error names a type you never wrote.

std_msgs/Header header

# ---- Timing -------------------------------------------------------------------------
uint64 ctrl_step             # absolute tick counter since arming; the FM cadence is % K of it
float64 latency_ms           # create_obs + get_action for this tick, against a 20 ms budget

# The two networks separately, because they are judged against DIFFERENT budgets and a single
# total cannot say which one moved. The tracker runs on the control thread inside the tick, so
# its budget is the control period. The generator runs on its own thread and is asked for a
# plan `lead_ticks` before the boundary it is for, so its budget is lead_ticks * dt -- several
# times larger. `fm_infer_ms` holds the SAME value on the K-1 ticks between inferences; a
# reader wanting per-plan statistics must deduplicate on `fm_plan_id`.
float64 fm_infer_ms
float64 trk_infer_ms
# Wall time since the previous tick. The ROS timer's own jitter, which nothing else records:
# a loop not actually running at 50 Hz rescales the FM cadence (counted in TICKS) and every
# reference offset with it, and would otherwise be invisible in a bag.
float64 tick_period_ms

# ---- Generator schedule health --------------------------------------------------------
# Whether the receding-horizon schedule is being met. `fm_late` is the instantaneous state --
# no plan arrived for this boundary, so the cursor advanced anyway and the far end of the
# window began repeating its last node. The rest are CUMULATIVE counters since arming; take
# differences to see a rate. `plans_dropped` counts plans the generator finished but that were
# superseded in the mailbox before a tick read them, and `requests_skipped` counts re-plan
# requests refused because the worker was still busy -- both mean the generator is running
# slower than its cadence, and both appear before lateness does.
bool fm_late
uint32 fm_late_events
uint32 fm_plans_dropped
uint32 fm_requests_skipped
uint32 fm_offsets_clamped

# ---- Input provenance ---------------------------------------------------------------
# WHICH sample each input was, not its value. The estimator publishes at 1 kHz against a
# 50 Hz tick, so without the consumed stamp there is no way to tell offline which of twenty
# candidate messages fed a given tick -- and that alignment is exactly what a sim-vs-robot
# comparison turns on. Ages are measured on LOCAL receipt time, because the cameras' clock
# is not synchronised to this machine.
builtin_interfaces/Time est_state_stamp
float64 est_state_age        # seconds between the estimate's receipt and this tick
float64[] depth_age          # per camera, in the artifact's contract order
# The same frames' age measured from CAPTURE -- the publisher's header stamp -- rather than
# from local receipt, so it includes exposure, render and transport as well as the wait for a
# tick. depth_age is the controller's own contribution; this is the whole pipeline.
#
# Only comparable when the two clocks are: in sim the world node stamps with the same ROS
# clock the controller reads, so this is exact. On hardware the cameras run through a Jetson
# whose clock is NOT synchronised, and this then carries the offset too -- which is why the
# watchdog still uses receipt time and this field is diagnostic only. <0 if a camera has never
# published, or published with an empty stamp.
float64[] depth_capture_age
builtin_interfaces/Time[] depth_stamp   # the consumed frame's own stamp, per camera

# ---- camera-pose / image time matching ----------------------------------------------
# The generator's xyz channels unproject each depth image with the camera's pose, and that
# is only correct when the two describe the same instant. The image is tens of ms old, so
# the pose is looked up from a history of estimator samples at the frame's capture stamp
# rather than taken from the current tick. These say whether that actually happened.
#
# `depth_pose_age` is how far back the pose used for each camera was measured; it should
# track depth_capture_age closely. -1 means the live pose was used instead, which is the
# pre-matching behaviour and means the stamp was unusable -- absent, in the future, or
# older than the history, all of which say the camera box's clock is not this one's.
float64[] depth_pose_age
uint64 depth_pose_fallbacks   # cumulative frames that fell back to the live pose
uint64 depth_pose_clamped     # cumulative lookups that ran off the end of the history
float64 vel_cmd_age          # <0 if no velocity command has ever arrived

# ---- Command ------------------------------------------------------------------------
# The command the policy was ACTUALLY given, after the watchdogs. It differs from
# /obelisk/g1/vel_cmd whenever depth or the command latches stale, and in debug-clip mode it
# is the clip's authored target_vel rather than anything on the network -- neither of which
# is visible from outside the node.
float64[3] command_effective

# ---- Joint ordering -----------------------------------------------------------------
# The POLICY's joint order, which is NOT the robot's: IsaacLab orders joints breadth-first
# (left_hip_pitch, right_hip_pitch, waist_yaw, ...) while the robot publishes them per limb
# (left_hip_pitch, left_hip_roll, left_hip_yaw, ...). 27 of 29 positions differ on the G1.
# `action_isaac` and the reference window's joint block are in THIS order; /obelisk/g1/ctrl
# and /obelisk/g1/est_state are in the robot's, and each carries its own `joint_names`.
# Repeated every tick so the bag is self-describing -- subtracting one order from the other
# produces a plausible-looking error signal that means nothing, which is exactly the mistake
# this field exists to prevent.
string[] joint_names_isaac

# ---- Policy output ------------------------------------------------------------------
# The raw network output, in the POLICY's joint order. /obelisk/g1/ctrl carries the scaled,
# offset and hardware-reordered target; this is the tensor the tracker emitted, which is
# also what feeds back in as the next tick's `actions` observation term.
float64[] action_isaac

# ---- What the tracker was told to do ------------------------------------------------
# The reference window as handed to the policy, flattened [offset0(D), offset1(D), ...] --
# the single most diagnostic quantity in the loop, and the only field here that duplicates
# something derivable. It is carried anyway because deriving it means re-running the
# re-anchoring offline, and a bug in that replay is indistinguishable from a bug in the
# controller. `reference_blocks` and `reference_offsets` name the layout so a reader does
# not have to restate the tracker metadata.
float64[] reference_window
string[] reference_blocks
int32[] reference_offsets
uint32 reference_node_dim

# ---- Generator state ----------------------------------------------------------------
# Which plan, and how far into it. The plan itself is on /obelisk/g1/debug/fm_plan, published
# only when it changes; these say which one this tick used and where the cursor sat, so a
# tick can be joined to its plan. All zero in debug-clip mode, where no generator is loaded.
uint64 fm_plan_id
int32 fm_cursor              # served_node_index; -1 when no plan is live
float64[3] fm_anchor_pos     # robot pose the live plan is expressed relative to
float64[4] fm_anchor_quat    # xyzw

# ---- Debug-clip state ---------------------------------------------------------------
bool clip_mode               # true when a baked clip is driving and the generator is absent
int32 clip_step              # -1 outside clip mode

# ---- Controller state ---------------------------------------------------------------
# Arming is otherwise observable only as a log line, and the watchdog latches only as
# edge-triggered warnings -- so a run that spent 90 s degraded looks identical to a healthy
# one after the fact.
bool armed
bool depth_stale
bool vel_cmd_stale
uint32 nonfinite_streak

# ---- IMU health ---------------------------------------------------------------------
# A flaky IMU on one robot returns an ALL-ZERO estimator message perhaps 1 in 10 samples.
# The controller detects it by the base quaternion's norm and reuses the last good
# orientation AND angular velocity, which at 1 kHz against a 50 Hz tick is invisible to the
# policy -- so without these counters a run on a failing IMU is indistinguishable from a
# healthy one after the fact, which is the whole reason the patch is easy to forget about.
# Cumulative since the last arm, so a rate is bad_quat_samples / quat_samples.
uint64 quat_samples          # base quaternions examined
uint64 bad_quat_samples      # of those, degenerate (norm below the threshold) and held
bool quat_held               # true if THIS tick ran on a held orientation/gyro

# ---- Depth health -------------------------------------------------------------------
# Per camera, contract order. A silently degraded camera -- all-invalid frame, wrong
# exposure, half the image saturated -- currently reads to this stack as perfectly healthy,
# because nothing looks at the pixels. `invalid_fraction` uses the same threshold the
# observation normalisation applies.
float64[] depth_hz           # measured, from inter-arrival times
float64[] depth_jitter       # stddev of inter-arrival, seconds
float64[] depth_min          # metres, over finite pixels
float64[] depth_max
float64[] depth_mean
float64[] depth_invalid_fraction

# ---- Integrity ----------------------------------------------------------------------
# blake2b over the assembled observation's float32 bytes. Rebuild the vector offline from the
# bagged inputs and compare: equal means the reconstruction is exact, and any analysis built
# on it is trustworthy. Without it a replay bug and a controller bug look the same.
uint64 obs_checksum
uint32 obs_size
