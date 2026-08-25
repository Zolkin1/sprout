# Copyright 2026 Zachary Olkin. All rights reserved.
#
# The flow-matching generator's inputs and output, published ONLY when it re-plans.
#
# The generator runs one tick in K (K=12 for the current pair), so publishing this per tick
# would repeat the same ~2700 floats eleven times out of twelve. `plan_id` joins it to the
# ControllerDebug ticks that consumed it, and `cursor` on those says how far into this plan
# each one had walked.
#
# Together with the ControllerDebug tick this is the complete generator picture: everything
# the FM was conditioned on, and everything it produced. Nothing else in the stack sees it.

std_msgs/Header header

uint64 plan_id               # increments once per re-plan; joins to ControllerDebug.fm_plan_id

# WHEN this plan happened, which needs three numbers rather than one: the generator is asked
# for a plan BEFORE the boundary that plan is for, and may then arrive AFTER it.
#   request_ctrl_step  the tick whose exteroception and velocity command CONDITIONED the plan
#   ctrl_step          the tick that ADOPTED it -- the first tick it drove the robot
#   lead_ticks         the schedule lead, so request + lead is the boundary it was built FOR
# On time, ctrl_step == request_ctrl_step + lead_ticks; late, ctrl_step is larger by exactly
# how late. Only `request_ctrl_step` can be joined against `depth_capture_age` to say how old
# the image the model actually saw was.
#
# `ctrl_step` was documented here as the sample tick and is NOT: it is filled at publish, and
# publish is triggered by ADOPTION. Reading it as the sample tick understates the age of the
# exteroception by the entire lead, silently -- it cost a wrong answer once, which is why the
# other two fields exist rather than being left derivable.
uint64 ctrl_step
uint64 request_ctrl_step
uint32 lead_ticks

# ---- Output -------------------------------------------------------------------------
# The sampled reference window, flattened [node0(D), node1(D), ...]. This is the raw graph
# output, in its own anchor frame -- NOT re-anchored onto the robot. The re-anchoring
# happens per tick on the way into the observation, and the result is what
# ControllerDebug.reference_window carries; keeping this one raw means a frame bug in the
# re-anchoring is visible as a difference between the two rather than baked into both.
float64[] trajectory
uint32 num_nodes             # F
uint32 node_dim              # D
string[] window_blocks       # block names, in order, that make up each node

# The FK target frames the joint_pos block IMPLIES. The slim window predicts joint angles and
# no ee_pos, so a viewer that wants the feet has to forward-kinematic them -- and it needs
# these names to do it. Carried here because there is nowhere else on the graph to get them,
# and a viewer with its own hardcoded list would be a second copy of a tracker contract.
string[] ee_frame_names

# ---- The frame the output lives in --------------------------------------------------
float64[3] anchor_pos        # robot base position at the re-plan
float64[4] anchor_quat       # xyzw

# ---- Conditioning -------------------------------------------------------------------
# The autoregressive history is the runtime's OWN previously served nodes, not measured
# state -- so if it goes wrong it goes wrong silently and compounds, and this is the only
# place it can be seen. `history_null` says the learned null-history embedding replaced the
# window's contents, which is normal for the first N steps after a reset and a bug at any
# other time.
float64[] history_nodes      # [N * D]
float64[] history_cmd        # [N * 3]
uint32 history_length        # N
bool history_null

# Exteroception as the generator saw it: already normalised and clipped, one token per
# camera (or one height-scan token). Comparing this against the raw camera stream on the
# bag is how a normalisation or camera-ordering error is caught.
float64[] extero_tokens
string extero_source         # "depth" or "height_scan"

# The future velocity command the sample was conditioned on, strided to the model's token
# steps. Normally the current command held flat; a clip->FM handoff supplies the real
# upcoming horizon instead, and the difference matters for how the sample bends.
float64[] cmd_tokens
uint32 cmd_dim

float64 infer_ms             # this sample's ONNX time; ~99% of the tick's cost
