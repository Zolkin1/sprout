#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Stop whichever obk launcher is running (sim, hardware, world -- they share one container
# and one tmux server, so this stops all of them).
#
# If the last launch put the ZED node on the Jetson, that lives in its OWN tmux inside the
# Jetson container and does NOT die with this one. By default it is LEFT RUNNING (slow and
# flaky to restart, and the next launch re-attaches to it); pass --zed to reap it too.
#
#   scripts/stop.sh          kill the container's tmux + reap orphaned launches;
#                            leave the container and the Jetson ZED node up.
#   scripts/stop.sh --zed    also stop the Jetson ZED node.
#   scripts/stop.sh --stop   also `docker compose stop` (container KEPT).
#   scripts/stop.sh --down   also `docker compose down` (container REMOVED).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"

obk_stop "$@"
