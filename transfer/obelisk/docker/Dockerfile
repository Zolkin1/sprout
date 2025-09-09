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

# # Optitrack
# RUN sudo apt install -y \
#     ros-humble-tf2 \
#     ros-humble-tf2-bullet \
#     ros-humble-tf2-eigen \
#     ros-humble-tf2-eigen-kdl \
#     ros-humble-tf2-geometry-msgs \
#     ros-humble-tf2-kdl \
#     ros-humble-tf2-msgs \
#     ros-humble-tf2-py \
#     ros-humble-tf2-ros \
#     ros-humble-tf2-ros-py \
#     ros-humble-tf2-sensor-msgs \
#     ros-humble-tf2-tools \
#     wget && \
#     cd /home/${USER} && git clone https://github.com/L2S-lab/natnet_ros2

# RUN sudo apt update && sudo apt install -y python3-pyqt5 lshw
