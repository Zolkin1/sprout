# The shared torch-free deploy runtime

numpy + onnxruntime, no torch and no IsaacLab. Metadata rides IN the `.onnx` -- there is no
side-car YAML. `transfer/deploy_cpp` is the C++ port of the numeric half, tested against this as its
oracle with no ROS in the loop.

`quat_math_np.py` is parity-tested against `transfer/torch_math/quat_math.py`; `transforms_np.py`
against the real `robot_rl.flow_matching.transforms` (`importorskip`-guarded).
`deploy/tests/test_fk_parity.py` holds URDF vs MJCF at 4e-13 on the camera chain and 4.6e-7 m at the
wrists, with MuJoCo kept as a test-only oracle.

# One exported generator per FM run (`artifacts.py`)

Exports are named for the checkpoint they came from -- `<fm run>_fm.onnx`, plus the checkpoint stem
when it is not `best` -- so a teacher's `exported/` holds every generator trained against it instead
of each `play.py` silently replacing the last. `fm_artifact_name()` is the one rule (used by
`play.py` and `export_fm_onnx.py`); `resolve_fm_onnx()` is how every consumer finds one
(`gen_tracker_runner.py --fm_onnx`, `benchmark.py --fm`, `export_sensor_configs.py`,
`export_to_hardware.py --fm_onnx`).

- **With several in a directory the resolver RAISES and lists them.** Defaulting to the newest would
  reinstate, as a silent default, the ambiguity the naming removed.
- **`fm.onnx` is still the DEPLOYMENT name**: a robot installs one pair and the obelisk controller
  reads a fixed filename, so `export_to_hardware.py` uploads whichever local artifact it was given AS
  `fm.onnx`. The rename happens at that boundary and nowhere else; the resolver still accepts a bare
  `fm.onnx` as input.
- Pre-rename `fm.onnx` files (anything exported before `extero_tokens` -> `extero_sensor_count`) are
  refused at load and need one `export_fm_onnx.py` call; the `.pt` is unaffected.
- **The FM needs a GPU**: CPU 14-32 ms against CUDA 6.7 on a 20 ms budget, and CUDA falls back
  silently without `preload_dlls`.

# Spec agreement: `depth_obs.py` and `scan_obs.py`

The tracker's contract and the generator's must describe the same physical cameras. The rules that
are easy to get wrong:

- **Match by NAME, not index.** The policy reads the NEAR camera alone and the generator both, so the
  two contracts have DIFFERENT LENGTHS; `assert_depth_specs_agree` keys on `camera_key` (which
  strips the `fm_` sampling prefix). Pairing by index would compare the policy's near camera against
  the generator's far one and report two genuinely different cameras as a violation. A camera the
  tracker reads and the generator does not stays an error -- the pair cannot supply it.
- **Relaxing the resolution check is NOT "drop width/height".** `fx/fy/cx/cy` are linear in the
  resolution, so they go free at the same time and a narrower cone on a bigger grid would pass.
  `depth_obs.py` exchanges the sampling fields for a positive check on the four resolution-invariant
  pinhole half-angles -- all four, not their sum, since a compensating principal-point shift
  preserves the sum. `REQUIRE_SAME_DEPTH_RESOLUTION = False` carries the TODO to re-tighten.
- **`extero_sensor_dim` cannot tell 30x26 from 26x30**, and under a convolution a transposed camera
  is scrambled input rather than a reordered vector. The per-camera `(H, W)` is DERIVED from the
  recording's `depth_camera_spec`, travels into `fm.onnx`, and is checked at deploy.
- `parse_extero_spec` picks its parser from the contract's own `extero_source` (parsing a scan as
  depth failed for a missing `fx`). `assert_scan_specs_agree` compares the two artifacts over the
  fields they SHARE -- `fm.onnx` embeds the full `mdp.height_scan_specs` dict, `tracker.onnx` the six
  `export_parameters` emits -- with a floor so a sparse contract cannot agree by silence.
- A **depth tracker + scan generator** is a legitimate pair, so `_resolve_depth_cameras` only
  reconciles `extero_spec` against the tracker's cameras when the FM's source IS depth.
- `GenTrackerPolicy` defaults `--allow_privileged_extero` OFF and nothing on the robot path passes
  it, which is what keeps a privileged dense scan sim-only.

# The generator's ONNX graph runs entirely on the GPU (and is captured)

The graph is ~3000 kernels for one 62-node plan at **batch 1**, so onnxruntime is LAUNCH-bound there:
wall time tracks the kernel COUNT, not the FLOPs. Two changes follow, and the first is a precondition
for the second.

**1. The batch axis is FIXED at 1 in the export** (`export_fm_onnx._export_graph`; `--dynamic_batch`
restores the old symbolic axis for throughput experiments). With a symbolic batch, every head reshape
in the traced attention blocks computed its shape at RUN time, so the graph carried a
`Shape -> Gather -> Unsqueeze -> Concat` chain of int64 scalars per block -- and onnxruntime keeps
shape arithmetic on the HOST, so **2687 of 6606 kernel nodes ran on the CPU** with 71
`MemcpyFromHost` nodes between them (the "Memcpy nodes are added" warning at load). Fixing the axis
lets constant folding delete all of it, which also lets the fusion pass match patterns the shape ops
were splitting (`SkipLayerNormalization`, `QuickGelu`, a fused QKV `Split`). Result: **0 host nodes,
3052 kernel nodes.** `tracker.onnx` was never part of this -- 17 nodes, already all-CUDA, 0.18 ms.
- The GPU-free way to check it is ORT's OWN optimizer, not the file: the file keeps its `Shape` nodes
  either way and folding them is the work being measured. 0 surviving `Shape` against 358 lines up
  exactly with 3052 all-CUDA against 6606-with-2687-on-host (`test_fm_onnx_static_batch`).
- Numerics: ~1e-6 from the old graph (torch folds a few more constants once shapes are known), three
  orders below the ~2.6e-3 CPU-vs-CUDA fp32 spread the graph already had. Freezing an EXISTING graph
  instead (`onnxruntime.tools.make_dynamic_shape_fixed --dim_param V --dim_value 1`) is
  bit-identical, if a forensic comparison is ever needed.

**2. CUDA Graph capture, which the host nodes had been blocking.** onnxruntime refuses capture
outright while any node is on the host ("all compute graph nodes have not been partitioned to the
CUDAExecutionProvider") -- it fails to CREATE the session, so it cannot half-apply. On:
`fm_cuda_graph: true` in `g1_cpp.yml`, `--fm_cuda_graph` on `gen_tracker_runner.py`,
`load_fm_context(cuda_graph=True)`. Generator only; the tracker has nothing to win.

**Every wrong way of feeding a captured graph fails SILENTLY**, which is why there are tests either
side of the port (`tests/test_fm_cuda_graph.py`, `deploy_cpp/test/ort_sessions_test.cc`). A capture
holds raw DEVICE POINTERS, so:
- a plain `sess.run()` on a capture-enabled session returns an **empty output list** once the capture
  is recorded -- an `IndexError` far from the cause, or no plan at all;
- binding host memory, or rebinding fresh host buffers, returns the **first** answer forever,
  bit-identical, with no error;
- even mutating one bound host buffer IN PLACE is not picked up, because the host-to-device copy is
  itself inside the capture.

That last shape is the dangerous one: "the plan stopped responding to the cameras" is
indistinguishable on a walking robot from conditioning that has not moved. So the inputs live in
device buffers written in place -- `update_inplace` in python, `Ort::Env::CopyTensors` in C++ -- and
the tests assert the replay TRACKS ITS INPUT, not merely that it is finite.
- **`CopyTensors` resolves its copy from the providers registered on the `Ort::Env`, which is a
  DIFFERENT registry from the one `SessionOptions` appends to.** A process can hold a perfectly good
  CUDA session and still get "Data transfer implementation between source and destination device was
  not found" on the first copy. `EnsureCudaDataTransfer()` registers the provider library on the Env
  once, by bare soname (it sits beside `libonnxruntime.so`, so an absolute `/opt` path would just be
  a second statement of where ORT is installed). Needs ORT >= 1.23; the images ship 1.24.4.
- Linking CUDA to do the copy ourselves was the alternative and was rejected: the images carry
  `libcudart.so.13` but no CUDA headers, so it would mean growing them.
- The C++ warm-up must go through the binding under capture (a plain warm-up would both mis-warm the
  session and leave the capture cost on the first live tick), and profiling is skipped there entirely
  -- placement is established by construction.

**Measured, per re-plan** (dynamic -> fixed batch -> fixed + capture): `11.6 -> 9.8 -> 8.3 ms` on an
RTX 4070, `22.4 -> 19.3 -> 14-15 ms` on the Jetson Thor. Quote Thor numbers as a RANGE: repeated
passes on one graph vary ~8% there (clock/thermal) where the desktop is stable to ~1%. **The Thor
baseline sat ABOVE the 20 ms control period**, so this is aimed at the chronically-late plan the
two-box bring-up traced its falls to, not just at throughput. The capture path is verified on the
Thor through the PYTHON runtime on the host (3/3 tests, 19.33 -> 15.45 ms); the C++ half compiles
clean on aarch64 and passes 12/12 on the CPU there, but its GPU behaviour in the Thor CONTAINER is
unverified (`docker run --runtime nvidia` gives that container no working CUDA, and the untouched
`ort_smoke` fails identically, so it is the launch, not the change).

# The clip pack

`tools/dump_clip_pack.py` writes a clip library into the `manifest.json` + raw-float64-blob format
`fixture_io.py` already writes and `g1_deploy::BlobPack` reads. It exists because the deploy image
ships no HDF5 and the libraries are `lzf`-compressed, which the plain HDF5 C library cannot decode
without h5py's filter plugin. The pack carries the channel NAMES, so a pack built for another robot
fails loudly instead of reading the right columns of the wrong motion, and it is committed beside
the `.h5`.

# No PD warm-up, and no way to add one

The stand hold, its stiffened gains, the start-button gate, `warm_start()` and
`BatchedHistory.fill()` are all DELETED. Every one of them put the robot in a state no training
episode started from and fed the generator history rows of ZEROS: `_make_snapshot` reads the
runtime's own previously-served node, which does not exist yet, which flipped `null_mask` off and
denied the model the learned null-history embedding meant to cover exactly those steps. What survives
is `advance_dormant_step()`, which moves the cadence counter and touches nothing else -- the clip path
needs it so the first re-plan after a clip->FM handoff lands on the right phase. Pinned by
`tests/test_history_null.py` on both sides of the port.

# The cadence is not a knob

Under `window_anchor="plan"` the clip's ego re-anchor and the reference window's anchor latch must
fire on the SAME step, or up to `K-1` steps of the robot's own egomotion leak into `base_pose` -- a
window matching neither training nor an odometry-free serve, and exactly as wide as a correct one.
Both configs DERIVE it from the tracker's `fm_update_every` (`0` = derive) and both runtimes refuse a
mismatch. That cancellation is what makes C++ clip playback odometry-free: measured 8.3e-17 m between
a walking and a stalled robot, against 0.132 m at a mismatched cadence.
