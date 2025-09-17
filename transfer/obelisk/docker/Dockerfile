# syntax=docker/dockerfile:1

# base image
FROM ubuntu:22.04 as base
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
# clone Obelisk into this docker image at the $OBELISK_ROOT path
# RUN git clone -b v0.3.0-alpha https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk
RUN git clone -b unitree-debug https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk
ENV OBELISK_ROOT=/home/${USER}/obelisk

# Install other dependencies here...

# you must run the setup script from a directory where the user has permissions
# run docker setup script in Dockerfile
WORKDIR /home/${USER}
RUN source /home/${USER}/obelisk/setup.sh --install-sys-deps --cyclone-perf --obk-aliases --basic --unitree

WORKDIR /home/${USER}

COPY requirements.txt .
RUN pip install --retries 5 --timeout=1000 -r requirements.txt
# RUN pip install -r requirements.txt
RUN rm requirements.txt

# Lidar odometry
# Install PCL and other dependencies for the lidar
RUN sudo apt-get update && sudo apt-get install -y \
    libpcl-dev \
    pcl-tools \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions
    
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
    git clone https://github.com/Livox-SDK/livox_ros_driver2.git && \
    cd livox_ros_driver2 && source /opt/ros/humble/setup.sh && ./build.sh humble && source ../../install/setup.sh && \
    cd /home/${USER} && mkdir -p lidar_odom_ws/src && cd lidar_odom_ws/src && \
    git clone -b ROS2 https://github.com/wdc3iii/FAST_LIO_vel.git --recursive && cd .. && \
    sudo rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install && \
    echo "source /home/${USER}/lidar_odom_ws/install/setup.bash" >> /home/${USER}/.bashrc
