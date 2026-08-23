#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# One-time setup of the brain -> ZED box ssh link, which is how the depth cameras come up.
#
# run_hardware_low.sh's depth pane does not launch a camera node locally: it ssh's to the ZED
# box and runs `run_node.sh pair` inside the container there (see obk_jetson_zed_cmd). That
# ssh happens from INSIDE the brain's container, non-interactively, in a tmux pane -- so it
# has to work with no password, no passphrase and no prompt of any kind. When it does not,
# the pane sits empty and no depth arrives, which on a robot reads as a camera fault.
#
# Two things this sets up, and both are deliberate:
#
#   1. A DEDICATED key, ~/.ssh/id_obk_zed, with NO passphrase. The obvious move is to reuse
#      ~/.ssh/id_ed25519 -- but on this box that key HAS a passphrase (it is the GitHub one,
#      and CLAUDE.md already records that non-interactive ssh with it fails). An interactive
#      pane would block at a passphrase prompt indefinitely, showing nothing about why.
#      A separate key also means the camera link does not inherit whatever that key can reach.
#
#   2. A ~/.ssh/config block with `IdentitiesOnly yes`, so ssh offers ONLY that key. Without
#      it ssh still tries the default identities first and can land back on the passphrase
#      prompt even though the right key exists.
#
# ~/.ssh is bind-mounted into the container (docker-compose.yml), so a key made here on the
# host IS the key the pane uses. That is also why this script can run on either side.
#
# Usage:
#   scripts/setup_zed_link.sh            set the link up (asks for the ZED box's password ONCE)
#   scripts/setup_zed_link.sh --check    read-only: is the link there, from host AND container
#   scripts/setup_zed_link.sh --key-only make the key + config, do not copy it over
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"

CHECK="" ; KEY_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check)    CHECK=1; shift ;;
        --key-only) KEY_ONLY=1; shift ;;
        -h|--help)  sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup_zed_link.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

HOST="${OBK_JETSON_HOST:-$OBK_JETSON_HOST_FALLBACK}"
ADDR="${HOST#*@}"
KEY="$HOME/.ssh/id_obk_zed"
CFG="$HOME/.ssh/config"
BEGIN="# >>> obk zed link (scripts/setup_zed_link.sh) >>>"
END="# <<< obk zed link <<<"

say() { printf '%s\n' "$*"; }

# ----------------------------------------------------------------------------------------
# Check the link the way the depth pane will use it: BatchMode, so a prompt is a failure
# rather than a question. Run in the CONTAINER when one is up, because that is where the
# pane's ssh actually runs -- it has its own /etc/ssh/ssh_config and its own idea of
# OBK_JETSON_HOST (compose puts docker/.env there, and that value wins).
# ----------------------------------------------------------------------------------------
check_container() {
    cd "$OBK_DOCKER_DIR" || return 1
    if ! docker compose -f "$OBK_COMPOSE_FILE" ps --status running --services 2>/dev/null \
         | grep -qx "$OBK_SERVICE"; then
        say ""
        say "  (the '$OBK_SERVICE' container is not running, so the check above is the HOST's"
        say "   ssh only. The pane runs it inside the container -- start the stack, or"
        say "   \`cd docker && docker compose up -d\`, and re-run --check.)"
        return 0
    fi
    say ""
    say "--- from inside the '$OBK_SERVICE' container (where the pane's ssh runs) ---"
    # A RUNNING container carries the OBK_JETSON_HOST it was CREATED with. Editing
    # docker/.env does not reach it until compose recreates it, so a long-lived container
    # can be aimed at a box the file stopped naming hours ago. The launch scripts' `up -d`
    # does recreate it -- but a --check run against the live one has to say which value it
    # is actually testing, or it is testing the wrong link and reporting the right one.
    local in_ctr
    in_ctr="$(docker compose -f "$OBK_COMPOSE_FILE" exec -T "$OBK_SERVICE" \
              printenv OBK_JETSON_HOST 2>/dev/null | tr -d '\r\n')"
    say "  container's OBK_JETSON_HOST: ${in_ctr:-<unset>}"
    if [ -n "$in_ctr" ] && [ "$in_ctr" != "$HOST" ]; then
        say "  MISMATCH with docker/.env ($HOST) -- this container predates that edit."
        say "  \`cd docker && docker compose up -d\` recreates it (the run scripts do this)."
    fi
    # OBK_IN_CONTAINER=1 stops the lib re-exporting docker/.env over the container's own
    # environment, which would hide exactly the mismatch above.
    docker compose -f "$OBK_COMPOSE_FILE" exec -T -e OBK_IN_CONTAINER=1 "$OBK_SERVICE" bash -c '
        source "$ROBOT_RL_ROOT/scripts/lib/obk_launch.sh"
        obk_jetson_preflight && obk_jetson_clock_report' || return 1
}

if [ -n "$CHECK" ]; then
    say "=== brain -> ZED box link: $HOST ==="
    say ""
    say "--- from this host ---"
    if [ -f "$KEY" ]; then
        if ssh-keygen -y -P "" -f "$KEY" >/dev/null 2>&1; then
            say "  key      $KEY (no passphrase -- usable from a pane)"
        else
            say "  key      $KEY EXISTS BUT IS PASSPHRASE-PROTECTED -- a pane will hang on it."
            say "           Delete it and re-run this script without --check."
        fi
    else
        say "  key      MISSING ($KEY) -- run this script without --check"
    fi
    if grep -qF "$BEGIN" "$CFG" 2>/dev/null; then
        say "  config   ~/.ssh/config has the obk block (IdentitiesOnly -> only that key)"
    else
        say "  config   ~/.ssh/config has NO obk block -- ssh may offer the passphrase key first"
    fi
    obk_jetson_preflight || true
    obk_jetson_clock_report || true
    check_container
    exit 0
fi

# ----------------------------------------------------------------------------------------
# 1. The key.
# ----------------------------------------------------------------------------------------
if [ -f "$KEY" ]; then
    if ! ssh-keygen -y -P "" -f "$KEY" >/dev/null 2>&1; then
        say "setup_zed_link.sh: $KEY exists but has a passphrase, which is exactly what this"
        say "  key exists to avoid (a tmux pane cannot answer a prompt). Move it aside and"
        say "  re-run:  mv $KEY{,.bak} && mv $KEY.pub{,.bak}"
        exit 1
    fi
    say "[zed-link] key present: $KEY"
else
    say "[zed-link] creating $KEY (ed25519, no passphrase)..."
    ssh-keygen -t ed25519 -N "" -C "obk-zed-link@$(hostname)" -f "$KEY" >/dev/null || exit 1
fi
chmod 600 "$KEY" 2>/dev/null || true

# ----------------------------------------------------------------------------------------
# 2. ~/.ssh/config -- a managed block, rewritten in place so re-running is idempotent.
#
# ConnectTimeout keeps a dead box from holding the pane; ServerAlive* makes a link that
# drops mid-run fail instead of hanging forever with the camera "running" on the far side.
# ----------------------------------------------------------------------------------------
umask 077
touch "$CFG"
BLOCK="$BEGIN
Host $ADDR zed-box
    HostName $ADDR
    User ${HOST%@*}
    IdentityFile $KEY
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
    ConnectTimeout 5
    ServerAliveInterval 5
    ServerAliveCountMax 3
$END"
python3 - "$CFG" "$BEGIN" "$END" "$BLOCK" <<'PY'
import re, sys
path, begin, end, block = sys.argv[1:5]
with open(path) as f:
    text = f.read()
pat = re.compile(re.escape(begin) + r".*?" + re.escape(end) + r"\n?", re.S)
if pat.search(text):
    text = pat.sub(block + "\n", text)
else:
    # PREPEND: ssh applies the FIRST matching value for each keyword, so a broad `Host *`
    # already in the file would otherwise win on IdentityFile and undo the point of this.
    text = block + "\n\n" + text
with open(path, "w") as f:
    f.write(text)
PY
chmod 600 "$CFG"
say "[zed-link] ~/.ssh/config block written (Host $ADDR zed-box -> $KEY only)"

[ -n "$KEY_ONLY" ] && { say "[zed-link] --key-only: not copying the key."; exit 0; }

# ----------------------------------------------------------------------------------------
# 3. Reachability, before anything that can hang.
# ----------------------------------------------------------------------------------------
if ! ping -c1 -W2 "$ADDR" >/dev/null 2>&1; then
    say "setup_zed_link.sh: $ADDR does not answer a ping."
    say "  Check the wired link (this box is 192.168.123.220 on it) and that OBK_JETSON_HOST"
    say "  in docker/.env names the ZED box and not the onboard minipc."
    exit 1
fi

# ----------------------------------------------------------------------------------------
# 4. Copy the key, unless it is already there.
# ----------------------------------------------------------------------------------------
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST" true 2>/dev/null; then
    say "[zed-link] pubkey auth to $HOST already works -- nothing to copy."
else
    say "[zed-link] copying the public key to $HOST."
    say "[zed-link] This is the ONE interactive step: it will ask for that box's password."
    ssh-copy-id -i "$KEY.pub" "$HOST" || {
        say "setup_zed_link.sh: ssh-copy-id failed."
        say "  If it refused the password, that box may have PasswordAuthentication off; then"
        say "  append this line to ~/.ssh/authorized_keys there by hand (from a console):"
        say ""
        say "    $(cat "$KEY.pub")"
        exit 1
    }
fi

# ----------------------------------------------------------------------------------------
# 5. Verify, on both sides of the container boundary.
# ----------------------------------------------------------------------------------------
say ""
say "--- verifying ---"
obk_jetson_preflight || exit 1
obk_jetson_clock_report || true
check_container || exit 1
say ""
say "[zed-link] done. run_hardware_low.sh will now bring the ZED pair up in its depth pane."
