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

Three host prerequisites, none of them optional and each of which fails somewhere else:

```bash
# 1. the nvidia runtime as the DEFAULT. nvidia-ctk writes daemon.json correctly; hand-editing
#    it is how you end up with a container that starts and has no GPU.
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default && sudo systemctl restart docker

# 2. the `raw` iptables table. Docker >= 29 installs a direct-access-filtering DROP rule into
#    it, and on an L4T kernel iptable_raw.ko SHIPS but is missing from modules.dep (which is
#    older than the .ko), so modprobe cannot find it until depmod runs. Without this every
#    build dies at the FIRST RUN step with "can't initialize iptables table `raw'" -- an error
#    that names iptables and looks nothing like a missing kernel module.
sudo depmod -a && sudo modprobe iptable_raw
echo iptable_raw | sudo tee /etc/modules-load.d/docker-iptables.conf   # or it is gone on reboot

# 3. ONLY if you also want the host-side ZED tools (ZED_Explorer, ZED_Depth_Viewer, the
#    tools/ scripts). The container runs privileged as root and needs none of this.
sudo usermod -aG zed,imu $USER      # then log out and back in
```

Then:

```bash
cd transfer/jetson/zed/docker
cp .env_example .env && $EDITOR .env        # ZED_USER/UID/GID, ZED_IMAGE, ROS_DOMAIN_ID
docker compose build
docker compose up -d                         # idle container; the node is launched into it
```

`ZED_IMAGE` must name **the SDK version installed on the host**, not merely a version that
exists. The GMSL ZED X path is split across the container boundary -- the kernel driver and
`nvargus-daemon` are the host's, only the SDK is the container's -- so the two have to agree.
Read the host's with `cat /usr/local/zed/zed-config-version.cmake` and its L4T revision with
`cat /etc/nv_tegra_release`.

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
| no `/dev/video*` and `ZED_Diagnostic` finds no camera | the cameras were not plugged in AT BOOT. GMSL is not USB: the deserializer probes once, and `sl_max96712_probe: No Camera connected to this deserializer` in `journalctl -b` is the proof. You do NOT need a reboot -- `sudo systemctl restart driver_zed_loader.service` re-runs `/usr/sbin/ZEDX_Driver`, which rmmods and reloads the `sl_*` modules and restarts nvargus |
| `Cannot open device /dev/spsc_bmi0: Permission denied`, then `CAMERA MOTION SENSORS NOT DETECTED` | that is the ZED X's onboard IMU, which `/etc/udev/rules.d/99-imu.rules` gives to group `imu`. The SDK treats a missing IMU as a FATAL open failure, not a degraded one. Host tools need the group; the container does not |
| `ZED_Depth_Viewer: command not found` right after `usermod` | the symlink in `/usr/local/bin` points into `/usr/local/zed`, which is mode 770 `user:zed`. A shell started before the group change cannot traverse it, and bash reports the EACCES as "command not found". Log out and back in |
| `docker compose build` fails at the first `RUN` with an iptables error | `iptable_raw` -- see prerequisite 2 above |
| `Connecting to nvargus-daemon failed: Connection refused` inside the container, and it never recovers | the container is pinned to a DELETED socket inode. nvargus recreates `/tmp/argus_socket` on every restart, so bind-mounting the socket FILE breaks permanently the first time the daemon restarts (it took a SIGBUS here during bring-up). The compose file mounts the DIRECTORY for this reason; if you see this, compare `stat -c %i /tmp/argus_socket` inside and out |
| the node will not reopen the camera right after being stopped | `sl::Camera::close()` takes **~25-30 s** to release a ZED X. The process outlives its own "Grab loop exited" log line by that much, and prints "ZED camera closed." only at the end. Wait for that line before relaunching |
| a camera reads `NOT AVAILABLE` with nothing holding `/dev/video*` | it is wedged below the SDK -- every open returns an Argus socket timeout. Same recovery as the hot-plug row: `sudo systemctl restart driver_zed_loader.service` |
| first camera open sits for ~6 min per depth mode | the TensorRT engine for that mode is not built. `ZED_Diagnostic -nrlo` / `-nrlo_light` / `-nrlo_plus` build them on the HOST; `run_node.sh optimize` does it in the container. They are per mode, and building one does not build the others |
| controller reports a depth dropout, no error here | `ROS_DOMAIN_ID` mismatch -- it must come from `.env` via `env_file`, since a shell `export` beats `${}` interpolation |
| camera open hangs ~20 s then fails | missing `SN<serial>.conf`; run `seed_zed_calib.sh` |
| first open takes minutes | TensorRT engine not built; run `run_node.sh optimize` |
| `cuInit` / `NvRmMemInit` failure | container not running as root (the tegra device nodes are `root:video 660`) |
| image is the wrong shape | node params do not match `zed_targets.yml`; regenerate and re-check |
