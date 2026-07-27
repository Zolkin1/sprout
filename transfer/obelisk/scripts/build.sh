#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Build the colcon workspace inside the container (obelisk itself is already built into the
# image). Run this once after a fresh clone or image rebuild, and after changing any C++;
# the Python nodes are symlink-installed, so editing them needs no rebuild.
#
#   scripts/build.sh           incremental
#   scripts/build.sh --clean   remove build/ install/ log/ first
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"

obk_build "$@"
