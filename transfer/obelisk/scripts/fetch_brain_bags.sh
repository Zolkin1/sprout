#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Copy the brain box's bags here, into log/bags_thor/ and NOT log/bags/.
#
# Separate directory on purpose. Bags are named by timestamp only, so a Thor bag dropped into
# log/bags/ is indistinguishable from a local one afterwards -- which already happened, and
# cost a comparison table with two mislabelled rows. The directory is the label.
#
#   scripts/fetch_brain_bags.sh            # the newest bag
#   scripts/fetch_brain_bags.sh 5          # the newest 5
#
# Never copy a bag from a LIVE run: the recorder is still writing and the mcap comes back torn
# at message ~0. Stop the run first (scripts/stop.sh finalises it), then fetch.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(dirname "$HERE")/log/bags_thor"
HOST="${OBK_THOR_HOST:-zolkin@192.168.50.20}"
RDIR="${OBK_THOR_DIR:-/home/zolkin/sprout/transfer/obelisk}/log/bags"
N="${1:-1}"
mkdir -p "$DEST"
BAGS=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$HOST" "ls -1dt $RDIR/*/ 2>/dev/null | head -$N" | tr -d '\r')
[ -z "$BAGS" ] && { echo "no bags on $HOST:$RDIR"; exit 1; }
for b in $BAGS; do
    name=$(basename "$b")
    if [ -d "$DEST/$name" ]; then echo "  have $name"; continue; fi
    # The bracket keeps pgrep from matching its OWN command line, which arrives on the remote
    # side containing the very pattern being searched for. Without it this always reported a
    # live recorder and refused every bag.
    if ssh -o BatchMode=yes "$HOST" "pgrep -f '[r]os2 bag record' >/dev/null"; then
        echo "  REFUSING $name: a recorder is running on $HOST, the bag would come back torn"; continue
    fi
    scp -qr -o BatchMode=yes "$HOST:$b" "$DEST/" && echo "  fetched $name"
done
