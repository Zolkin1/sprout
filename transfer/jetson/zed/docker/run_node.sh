#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Launch a node into a DETACHED container-side tmux, then attach. Baked into the image at
# /usr/local/bin/run_node.sh, so the onboard stack can reach it with a plain
# `docker exec -it zed_depth run_node.sh zed` -- no nested quoting, nothing to copy over.
#
# The detached tmux is the point: the node survives the ssh session (or the whole onboard
# tmux) dropping, and re-running this is IDEMPOTENT -- it re-attaches to the SAME session
# rather than starting a second node that would fight over the camera.
#
# Inside the session: Ctrl-C -> container prompt with the command one Up-arrow away;
# Ctrl-b Ctrl-b d detaches (nested tmux).
#
#   run_node.sh zed        launch the ZED depth node
#   run_node.sh optimize   build the TensorRT engines (no camera needed)
set -euo pipefail

node="${1:-}"
case "$node" in
    zed)      cmd="ros2 launch zed_depth_publisher zed_depth.launch.py" ;;
    optimize) cmd="optimize_ai_models.sh ${*:2}" ;;
    *) echo "run_node.sh: usage: run_node.sh <zed|optimize> [args...]" >&2; exit 2 ;;
esac

sess="obk_${node}"
rc="/tmp/obk_${node}_rc"
cat > "$rc" <<RC
[ -f ~/.bashrc ] && source ~/.bashrc
history -s '$cmd'
$cmd
RC

tmux has-session -t "$sess" 2>/dev/null || tmux new-session -d -s "$sess" "bash --rcfile $rc -i"
exec tmux attach -t "$sess"
