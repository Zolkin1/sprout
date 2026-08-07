# syntax=docker/dockerfile:1


# ============================================
# Stage 1: Base / Simulation
# ============================================
FROM ubuntu:22.04 as sim
SHELL ["/bin/bash", "-c"]

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

# Assert the installed wheel is the GPU one. This is the only CUDA question answerable at
# BUILD time, and it is worth asserting because it is the regression that actually happens:
# someone edits requirements.txt back to plain `onnxruntime` and the failure then surfaces
# on the robot as resolve_providers raising during bring-up.
#
# get_available_providers() reports what the wheel was COMPILED with -- a static list. It
# cannot tell you the provider will LOAD, which is the other half (the "libcublasLt.so.13
# on a CUDA 12 host" trap that pinning ==1.22.0 avoids).
#
# That second half deliberately is NOT checked here: dlopen'ing
# libonnxruntime_providers_cuda.so SEGFAULTS (exit 139) without a GPU, and `docker build`
# has no GPU -- measured both ways on this image. The .so does not LINK libcuda.so.1, but
# loading it still reaches the driver. The check therefore lives at runtime, where a GPU
# exists and where it matters: G1Controller._require_cuda compares the requested provider
# against what each ONNX session actually registered, and refuses to run on a fallback.
RUN python3 -c "\
import onnxruntime as ort; \
p = ort.get_available_providers(); \
assert 'CUDAExecutionProvider' in p, 'CPU-only onnxruntime wheel installed: %s' % p; \
print('[image] onnxruntime', ort.__version__, 'providers:', p)"

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