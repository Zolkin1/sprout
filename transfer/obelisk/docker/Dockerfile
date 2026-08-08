# syntax=docker/dockerfile:1

# The prebuilt onnxruntime C++ install tree. An ARG used in a FROM must be declared before
# the first FROM (global scope). Built separately, once per pin bump per architecture, by
# docker/Dockerfile.ort -- see that file for why it is not built here. If the tag is missing
# this fails immediately with a pull error rather than silently starting a ~50 min compile.
ARG ORT_DIST_IMAGE=g1-ort:1.24.4-cu130
FROM ${ORT_DIST_IMAGE} AS ortdist


# ============================================
# Stage 0: CUDA runtime
# ============================================
# The C++ deploy runtime does its inference on the GPU, so the image needs a CUDA userspace.
# On JetPack 7 that is not optional and not inherited: the nvidia container runtime mounts
# only devices.csv and drivers.csv here (checked on the Thor -- there is no cuda.csv or
# cudnn.csv as there was on JetPack 6), so the HOST supplies libcuda.so.1 and everything
# above it must be in the image.
#
# Its own stage, and the FIRST one, so these heavy rarely-changing layers sit below
# OBELISK_REF and are not invalidated by an obelisk bump.
#
# Measured cost: ~3.5 GB. That is a visible jump for an image that shipped no CUDA at all.
# The lever if it ever matters is onnxruntime_CUDA_MINIMAL, which drops cuDNN and cuBLAS in
# favour of ORT's own kernels -- not taken here because the effect on a transformer is
# unmeasured.
FROM ubuntu:22.04 AS cuda-base
SHELL ["/bin/bash", "-c"]

# RUNTIME CUDA only -- no nvcc, no -dev packages; nothing in this image compiles CUDA.
# NEVER `cuda`, `cuda-drivers` or `nvidia-open`: a driver package would either fail to
# install or shadow the host's with a mismatched libcuda.
#
# aarch64 -> sbsa. JetPack 7 ships the arm64-SBSA CUDA packaging, so Thor is served by the
# same repo as Grace; there is no ubuntu2204/tegra repo. This `case` is the ONLY place in the
# whole build that branches on architecture -- the install prefix is identical on both, so no
# CMakeLists ever has to know. That is the direct lesson of obelisk's hardcoded
# COMP_ARCH "x86_64" (see the MuJoCo patch below).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg && \
    case "$(uname -m)" in \
      aarch64) REPO_ARCH=sbsa ;; \
      x86_64)  REPO_ARCH=x86_64 ;; \
      *) echo "unsupported arch $(uname -m)" && false ;; \
    esac && \
    curl -fsSLO "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${REPO_ARCH}/cuda-keyring_1.1-1_all.deb" && \
    dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-13-0 cuda-nvrtc-13-0 \
        libcublas-13-0 libcufft-13-0 libcurand-13-0 libcusparse-13-0 libcusolver-13-0 \
        libcudnn9-cuda-13 && \
    rm -rf /var/lib/apt/lists/* && \
    echo "/usr/local/cuda-13.0/lib64" > /etc/ld.so.conf.d/000_cuda13.conf && ldconfig && \
    ldconfig -p | grep -q 'libcudart\.so\.13' && \
    ldconfig -p | grep -q 'libcudnn\.so\.9'
# The two ldconfig greps are the assert: the CUDA debs do not reliably drop an ld.so.conf.d
# entry for every sub-package, and a missing one does not fail here -- it surfaces much later
# as onnxruntime silently falling back to the CPU, which for this pair is not a degraded mode
# but an unusable one (~103 ms per generator tick against a 20 ms period).


# ============================================
# Stage 1: Base / Simulation
# ============================================
FROM cuda-base AS sim
SHELL ["/bin/bash", "-c"]

# The onnxruntime C++ runtime, built by Dockerfile.ort. Only the install tree comes across --
# none of the toolchain, CUDA dev packages or ORT sources that produced it.
COPY --from=ortdist /opt/onnxruntime /opt/onnxruntime
RUN echo "/opt/onnxruntime/lib" > /etc/ld.so.conf.d/001_onnxruntime.conf && ldconfig && \
    ldconfig -p | grep -q 'libonnxruntime\.so' && \
    ldd /opt/onnxruntime/lib/libonnxruntime_providers_cuda.so > /tmp/ldd.txt && \
    ! grep -q 'not found' /tmp/ldd.txt && rm /tmp/ldd.txt && \
    echo "[image] onnxruntime CUDA provider resolves against the runtime-only CUDA set"
# That ldd is not redundant with the one in Dockerfile.ort. There the provider was checked
# against the BUILDER's package set, which had every -dev package; here it is checked against
# the runtime-only set this image actually ships. A library that exists only in the builder is
# exactly the "libcublasLt.so.13: cannot open shared object file" failure that cost us the
# ==1.22.0 pin in requirements.txt, and this catches it at build time.

# username, uid, gid
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG OBELISK_ROOT=/
ENV USER=$USER
# ENV UID=1000
# ENV GID=1000
ENV UID=$UID
ENV GID=$GID
ENV OBELISK_ROOT=$OBELISK_ROOT

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# basic dependencies from docker_setup.sh (up until sudo and below)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang-tools-12 \
    nano \
    vim \
    git \
    libeigen3-dev \
    x11-apps \
    locales \
    iputils-ping \
    evtest \
    net-tools \
    tmux \
    sudo && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# create non-root user with sudo privileges for certain commands
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash && \
    echo "${USER}:password" | chpasswd && \
    usermod -aG sudo ${USER} && \
    echo "%${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# switch to new user and workdir
USER ${UID}


RUN echo "Pulling Obelisk..."
# Obelisk is pinned by SHA, not by branch: a branch clone silently changes the config
# schema and the node API from one image build to the next, and this stack's configs are
# written against one schema. Bumping OBELISK_REF busts this layer's cache (and so the
# obelisk_ws build below); OBELISK_REFRESH is the manual bust for re-pulling the same ref:
#     docker compose build --build-arg OBELISK_REFRESH=$(date +%s)
#
# Current pin: main. Relative to the old feb_hot_fix_scan pin this brings in the config
# rewrite this stack now depends on -- no top-level `onboard:`, no `device=` launch arg,
# one `obelisk_settings` YAML parameter per node, composable `include:` with KEYED merge
# (list entries merged by `name:`), `name:`-derived node names, `${VAR}` expansion in
# config values, inline `scan_config:`, ObkImu removed in favour of sensor_msgs/Imu, and
# optional control/estimation/robot sections so a config can launch any subset.
#
# Bumped 9f52216 -> abf37fe for TWO DepthImage sensors. At 9f52216 obelisk held ONE
# `depth_scan_interface_`, reassigned per sensor entry, so g1_world.yml's near+far pair
# published two topics at the right rates that both carried the FAR camera's image -- no
# error, just a policy fed a wrong observation. abf37fe keys the caster by sensor_key.
# Also picks up f97a2c9 (lidar per-direction occlusion mask), the only other change in
# between; it touches the ObkScan/PointCloud2 path we use for /livox/lidar.
#
# TODO(before hardware testing): replace this with an Obelisk RELEASE tag. Tracking a
# moving ref is fine while developing against the sim; it is not fine on the robot.
ARG OBELISK_REF=abf37fe71aa42db15266a12c5a7b9e9729c21d7a
ARG OBELISK_REFRESH=0
RUN echo "obelisk refresh token: ${OBELISK_REFRESH}" && \
    git clone https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk && \
    cd /home/${USER}/obelisk && git checkout ${OBELISK_REF}
ENV OBELISK_ROOT=/home/${USER}/obelisk

# Point obelisk's MuJoCo fetch at the arch this image is being BUILT for.
#
# obelisk_cpp/CMakeLists.txt hardcodes `set(COMP_ARCH "x86_64" ...)` and interpolates it
# into the MuJoCo release URL, so on aarch64 (a Jetson) FetchContent happily downloads the
# x86_64 tarball and the failure surfaces ~40 s later, deep in colcon, as
#
#   /usr/bin/ld: _deps/mujoco-src/lib/libmujoco.so: error adding symbols: file in wrong format
#
# which then aborts obelisk_unitree_cpp and everything else that links ObkCore. MuJoCo does
# publish mujoco-${VERSION}-linux-aarch64.tar.gz, so the arch is the only thing wrong.
# COMP_ARCH is a CACHE variable, i.e. overridable with -D, but the obelisk_ws build runs
# through obelisk's own build script and there is no seam to pass cmake args in -- patching
# the default is the smaller change.
#
# `$(uname -m)` prints exactly MuJoCo's asset spelling on both platforms (x86_64 / aarch64),
# so this is a no-op on the x86 dev box rather than a Jetson-only branch. Do NOT write it as
# \$(uname -m): docker leaves that backslash in place and bash then treats it as a literal.
# The grep first is the assert -- if obelisk ever fixes this upstream the pattern stops
# matching, and failing here is better than silently building the wrong MuJoCo again.
RUN grep -q 'set(COMP_ARCH "x86_64"' /home/${USER}/obelisk/obelisk/cpp/obelisk_cpp/CMakeLists.txt && \
    sed -i "/set(COMP_ARCH/s/x86_64/$(uname -m)/" \
        /home/${USER}/obelisk/obelisk/cpp/obelisk_cpp/CMakeLists.txt && \
    grep -n 'set(COMP_ARCH' /home/${USER}/obelisk/obelisk/cpp/obelisk_cpp/CMakeLists.txt

# Install other dependencies here...

# you must run the setup script from a directory where the user has permissions
# run docker setup script in Dockerfile
WORKDIR /home/${USER}
RUN source /home/${USER}/obelisk/setup.sh --downstream-setup --mujoco --unitree

# Pre-build the Obelisk ROS workspace INTO THE IMAGE so it survives `docker compose
# down/up` and a first launch does not pay for it. Build through `obk`, NOT a bare
# build_obelisk.sh: obelisk_unitree_cpp's CMakeLists only compiles its executables when
# OBELISK_BUILD_UNITREE=true is in the environment, which `obk` sets. Without it the image
# ships an obelisk_ws with no unitree executables and the failure surfaces much later, as
# "package 'obelisk_unitree_cpp' not found" at launch -- so assert it here instead.
RUN source /opt/ros/humble/setup.bash && \
    source /home/${USER}/.bash_aliases && \
    obk && \
    echo "[image] verifying the unitree interface built..." && \
    ros2 pkg executables obelisk_unitree_cpp | grep -q obelisk_unitree_sim
WORKDIR /home/${USER}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# The python onnxruntime, installed SEPARATELY from requirements.txt because which wheel is
# correct depends on the architecture, and on aarch64 on what the python side is even for.
#
#   x86_64  -- the GPU wheel, pinned. The dev box runs the python runtime against a GPU.
#   aarch64 -- the CPU wheel. There is NO aarch64 onnxruntime-gpu wheel on PyPI at any
#              version, and the only one that works on Thor is cp312 while ROS Humble pins
#              this image to cp310. That is the whole reason the deploy runtime is being
#              ported to C++: on this architecture THE GPU BELONGS TO THE C++ PATH
#              (/opt/onnxruntime, installed above), and python here is the sim2sim driver and
#              the parity oracle only. A CPU wheel is the right answer, not a fallback.
#
# Keep the pin in step with transfer/deploy/docker/Dockerfile.bench and the rationale block
# in requirements.txt when bumping either.
RUN case "$(uname -m)" in \
      x86_64)  pip install --no-cache-dir "onnxruntime-gpu[cuda,cudnn]==1.22.0" ;; \
      aarch64) pip install --no-cache-dir "onnxruntime==1.22.0" ;; \
      *) echo "unsupported arch $(uname -m)" && false ;; \
    esac

# Assert the installed wheel is the one this architecture expects. The regression this
# catches is real: someone edits the line above and the failure otherwise surfaces on the
# robot as resolve_providers raising during bring-up.
#
# get_available_providers() reports what the wheel was COMPILED with -- a static list. It
# cannot tell you the provider will LOAD, which is the other half (the "libcublasLt.so.13
# on a CUDA 12 host" trap that pinning ==1.22.0 avoids).
#
# That second half deliberately is NOT checked here: dlopen'ing
# libonnxruntime_providers_cuda.so SEGFAULTS (exit 139) without a GPU, and `docker build`
# has no GPU -- measured both ways on this image. The .so does not LINK libcuda.so.1, but
# loading it still reaches the driver. The check therefore lives at runtime, where a GPU
# exists and where it matters, and on aarch64 it lives in the C++ controller rather than
# here: it compares the requested provider against what each ONNX session actually
# registered, and refuses to run on a fallback.
RUN python3 -c "\
import platform, onnxruntime as ort; \
p = ort.get_available_providers(); \
want_gpu = platform.machine() == 'x86_64'; \
assert (not want_gpu) or 'CUDAExecutionProvider' in p, \
    'x86_64 expects the GPU wheel, got: %s' % p; \
assert want_gpu or 'CUDAExecutionProvider' not in p, \
    'aarch64 expects the CPU wheel (the GPU belongs to the C++ path), got: %s' % p; \
print('[image] python onnxruntime', ort.__version__, platform.machine(), 'providers:', p)"

# MCAP bag storage. rosbag2's default plugin is sqlite3, whose .db3 stores only message
# TYPE NAMES -- reading one back needs obelisk_control_msgs/obelisk_estimator_msgs/g1_msgs
# built and sourced, i.e. all analysis happens inside this container. MCAP embeds each
# schema in the file, so a run can be read on any machine with `pip install mcap` and no ROS
# at all, and Foxglove opens it directly (it dropped ROS 2 .db3 support). It is also chunked
# and append-only, so a bag killed mid-run -- which is every run that ends in a fall -- is
# still readable. scripts/lib/obk_launch.sh records with `-s mcap`.
RUN sudo apt-get update && sudo apt-get install -y \
    ros-humble-rosbag2-storage-mcap && \
    sudo rm -rf /var/lib/apt/lists/*

# ============================================
# Stage 2: Hardware (adds lidar stack)
# ============================================
FROM sim as hardware
# Lidar odometry
# Install PCL and other dependencies for the lidar
RUN sudo apt-get update && sudo apt-get install -y \
    libpcl-dev \
    pcl-tools \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions

# Livox SDK
WORKDIR /home/${USER}
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
 cd Livox-SDK2 && \
 mkdir build && cd build && \
 cmake .. && \
 make -j && \
 sudo make install && \
 cd /home/${USER} && \
 mkdir -p livox_driver_ws/src && \
 cd livox_driver_ws/src && \
 git clone https://github.com/Livox-SDK/livox_ros_driver2.git && cd livox_ros_driver2

RUN cd /home/${USER}/livox_driver_ws/src/livox_ros_driver2 && \
    source /opt/ros/humble/setup.sh && ./build.sh humble && source ../../install/setup.sh

RUN sudo rosdep init || true
RUN rosdep update

# LIO-Localization: FAST-LIO mapping PLUS scan_lock (ICP relocalisation against a prior
# map) and pose_prompt, in one workspace. This replaces the bare hku-mars/FAST_LIO clone:
# FAST-LIO alone provides odom->imu but never map->odom, so the robot's pose drifts in an
# unanchored frame. We need the map frame because the generator's history tokens are built
# from base POSITION and world velocity -- localisation is load-bearing for the POLICY here,
# not just for navigation.
#
# Pinned like OBELISK_REF: at this ref scan_lock/pose_prompt accept absolute and ${ENV}
# pcd_file_name paths, so the relocalisation configs can point straight at the bind-mounted
# resources/pcd/ instead of needing a push step after every image rebuild.
ARG LIO_LOC_REF=39451d2d6c0fdaf9aa059775fabee250647d72d7
RUN git clone https://github.com/wdc3iii/LIO-Localization.git /home/${USER}/LIO-Localization && \
    cd /home/${USER}/LIO-Localization && git checkout ${LIO_LOC_REF} && \
    git submodule update --init --recursive -j 8 && \
    cd ws_lio_loc && \
    rosdep install --from-paths src --ignore-src -r -y && \
    source /opt/ros/humble/setup.sh && \
    source /home/${USER}/livox_driver_ws/install/setup.bash && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

RUN echo "source /home/${USER}/livox_driver_ws/install/setup.bash" >> /home/${USER}/.bashrc && \
    echo "export LD_LIBRARY_PATH=/home/${USER}/livox_driver_ws/install/livox_ros_driver2/lib:\$LD_LIBRARY_PATH" >> /home/${USER}/.bashrc && \
    echo "source /home/${USER}/LIO-Localization/ws_lio_loc/install/setup.bash" >> /home/${USER}/.bashrc


# cat > config/MID360_config.json << 'EOF'
#{
#"lidar_summary_info" : {
# "lidar_type": 8
#},
#"MID360": {
# "lidar_net_info" : {
#   "cmd_data_port": 56100,
#   "push_msg_port": 56200,
#   "point_data_port": 56300,
#   "imu_data_port": 56400,
#   "log_data_port": 56500
# },
# "host_net_info" : {
#   "cmd_data_ip" : "192.168.123.222",
#   "cmd_data_port": 56101,
#   "push_msg_ip": "192.168.123.222",
#   "push_msg_port": 56201,
#   "point_data_ip": "192.168.123.222",
#   "point_data_port": 56301,
#   "imu_data_ip" : "192.168.123.222",
#   "imu_data_port": 56401,
#   "log_data_ip" : "",
#   "log_data_port": 56501
# }
#},
#"lidar_configs" : [
# {
#   "ip" : "192.168.123.120",
#   "pcl_data_type" : 1,
#   "pattern_mode" : 0,
#   "extrinsic_parameter" : {
#     "roll": 0.0,
#     "pitch": 0.0,
#     "yaw": 0.0,
#     "x": 0,
#     "y": 0,
#     "z": 0
#   }
# }
#]
#}
#EOF