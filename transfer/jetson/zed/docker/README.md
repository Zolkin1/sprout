# Jetson ZED depth container

The ZED SDK needs a GPU and the onboard minipc has none, so the depth camera runs here and
reaches the control stack over the shared DDS graph (host networking, same `ROS_DOMAIN_ID`).

The node publishes a downsampled, FOV-clipped depth image on **`/depth/image_processed`** --
the same topic and the same 32FC1 flat-float32 layout the Obelisk MuJoCo ray-caster
publishes in sim. That is the whole sim2real trick: the controller cannot tell them apart,
so what you tested in simulation is what runs on the robot.

## What the camera must be configured to produce

Not a free choice. The resolution, field of view and range have to match what the policy
trained on, and those numbers are embedded in the policy artifact. Generate them:

```bash
python scripts/rsl_rl/export_sensor_configs.py --run_dir logs/rsl_rl/g1/<run>
cat transfer/obelisk/g1_control/config/scan/zed_targets.yml
```

Then map those onto `zed_depth_publisher`'s own parameters
(`$(ros2 pkg prefix zed_depth_publisher)/share/zed_depth_publisher/config/default_params.yaml`).
The field names differ; the numbers must not.

## First-time setup, on the Jetson

```bash
# once per device: make the nvidia runtime the default in /etc/docker/daemon.json
#   { "default-runtime": "nvidia", "runtimes": { "nvidia": { "path": "nvidia-container-runtime" } } }
cd transfer/jetson/zed/docker
cp .env_example .env && $EDITOR .env        # USER/UID/GID, ZED_IMAGE, ROS_DOMAIN_ID
docker compose build
docker compose up -d                         # idle container; the node is launched into it
```

Then, **while the machine still has internet**, from the dev box:

```bash
./seed_zed_calib.sh <camera-serial>          # per-camera factory calibration
ssh <jetson> docker exec -it zed_depth run_node.sh optimize    # TensorRT engines
```

Both exist because of the same constraint: the robot LAN has no DNS. The SDK would
otherwise fetch the calibration and build the engine lazily on the first camera open --
which needs internet, and occupies the camera on the Jetson's only USB-C port. Skipping
these turns into a camera that hangs ~20 s and then fails to open, on the robot.

Both are persisted in named volumes (`zed_settings`, `zed_resources`) and survive
`down`/`up`. A `docker volume rm` means doing them again.

## Running

Normally you do not: `transfer/obelisk/scripts/run_hardware_low.sh` launches the node over
ssh in its own pane. By hand:

```bash
docker exec -it zed_depth run_node.sh zed
```

`run_node.sh` puts the node in a **detached container-side tmux**, so it survives the ssh
link dropping, and re-running it re-attaches to the same session instead of starting a
second node that would fight over the camera.

To stop it: `transfer/obelisk/scripts/stop.sh --zed` from the onboard machine. Plain
`stop.sh` deliberately leaves the camera running -- it is slow and flaky to bring back up,
and the next launch just re-attaches.

## Troubleshooting

| symptom | cause |
|---|---|
| controller reports a depth dropout, no error here | `ROS_DOMAIN_ID` mismatch -- it must come from `.env` via `env_file`, since a shell `export` beats `${}` interpolation |
| camera open hangs ~20 s then fails | missing `SN<serial>.conf`; run `seed_zed_calib.sh` |
| first open takes minutes | TensorRT engine not built; run `run_node.sh optimize` |
| `cuInit` / `NvRmMemInit` failure | container not running as root (the tegra device nodes are `root:video 660`) |
| image is the wrong shape | node params do not match `zed_targets.yml`; regenerate and re-check |
