#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# One time base for the robot: the BRAIN serves it, every other box on 192.168.123.0/24
# follows. Run with --master here, --client there.
#
# WHY THE BRAIN, and not the ZED box (which has the better clock today, being the one with
# internet over wifi):
#
#   1. The brain is where the numbers are CONSUMED. `depth_capture_age` is a receipt time
#      made here minus a capture stamp made there, and every other stamp in the bag -- ticks,
#      plans, est_state, ctrl -- is this box's. Making this box the reference turns a
#      difference between two independently steered clocks into one client's offset, which
#      that client reports in one line (`chronyc tracking`).
#   2. The ZED box's UTC comes over WIFI. The robot has to work with no wifi at all, and a
#      time base whose existence depends on the lab AP is not one. With `local stratum 10
#      orphan` here, the LAN keeps ONE self-consistent time base either way, and still tracks
#      real UTC whenever this box's own uplink is there.
#   3. If the brain's clock is going to be steered by someone, it should be steered by the
#      box that can be watched from the brain -- not the other way round. A step on the
#      camera box shifts capture stamps; a step here shifts everything in the recording.
#
# The honest cost, since the ZED box's suggestion was the cheaper one: making the brain the
# server means installing chrony HERE (systemd-timesyncd is an SNTP client and cannot serve
# time at all), and on Ubuntu chrony CONFLICTS with systemd-timesyncd, so apt removes it.
# Pointing timesyncd at the ZED box instead would have been zero-install. Worth the package:
# timesyncd also cannot report an offset worth reading, cannot serve the onboard PC later,
# and steps where chrony slews.
#
# NOT what this fixes: the depth watchdog. It times frames by LOCAL RECEIPT (see
# g1_controller.cc), deliberately, so a skewed camera clock can never stale-latch a healthy
# camera. What skew corrupts is `depth_capture_age` -- the only number that says how old a
# frame was when the policy used it -- and every cross-box comparison in a bag. Diagnostics,
# not safety, which is why this is a bring-up task and not a blocker.
#
# Usage:
#   scripts/setup_clock_sync.sh              # --check: read-only, both boxes
#   scripts/setup_clock_sync.sh --master     # THIS box serves the robot LAN (needs sudo)
#   scripts/setup_clock_sync.sh --client     # this box follows the brain  (needs sudo)
#   scripts/setup_clock_sync.sh --client --master-addr 192.168.123.220
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/obk_launch.sh"

MODE=check
MASTER="$OBK_TIME_MASTER"
SUBNET="${OBK_TIME_SUBNET:-192.168.123.0/24}"
CONF=/etc/chrony/conf.d/obk-robot-lan.conf

while [ $# -gt 0 ]; do
    case "$1" in
        --master)      MODE=master; shift ;;
        --client)      MODE=client; shift ;;
        --check)       MODE=check; shift ;;
        --master-addr) MASTER="${2:?}"; shift 2 ;;
        -h|--help)     sed -n '3,44p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "setup_clock_sync.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

say() { printf '%s\n' "$*"; }

# ----------------------------------------------------------------------------------------
# What a CLIENT actually gets from an NTP server, asked over the wire.
#
# `chronyc tracking` is not that answer. While the LOCAL reference is active it reports
# 7F7F0101 (127.127.1.1) / stratum 10 / zero root delay -- and it is active from startup
# until the root distance of a real source falls below the `local distance` threshold (1 s
# by default), so for the first minutes after `--master` this box reads as unsynchronised
# while clients are already being served the real upstream. Measured here: tracking said
# stratum 10 / 7F7F0101 for ~2 min after install, while a query to 192.168.123.220 was
# answered stratum 3 / 51.81.226.229 / 108 ms root delay. It also goes back to the local
# reference if every upstream is lost, which is the whole point of `local` -- and clients
# keep getting time, just marked stratum 10.
#
# The probe also proves `allow` is SCOPED: the same query to 127.0.0.1 must time out,
# because only the robot LAN is permitted to ask.
# ----------------------------------------------------------------------------------------
served() {
    python3 - "$1" <<'PYEOF'
import socket, struct, sys
target = sys.argv[1]
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)
try:
    sock.sendto(b"\x23" + b"\0" * 47, (target, 123))
    data, _ = sock.recvfrom(96)
except OSError as exc:
    print(f"no answer from {target} ({type(exc).__name__})")
else:
    stratum = data[1]
    delay, disp = struct.unpack("!II", data[4:12])
    refid = ".".join(str(b) for b in data[12:16]) if stratum > 1 else data[12:16].decode(errors="replace")
    print(f"stratum {stratum}, reference {refid}, root delay {delay / 65536 * 1e3:.1f} ms, "
          f"dispersion {disp / 65536 * 1e3:.1f} ms")
finally:
    sock.close()
PYEOF
}

# ----------------------------------------------------------------------------------------
# Read-only status of THIS box, in the terms that matter: who steers it, how far off it is,
# and -- when it is the master -- who is actually asking it for time.
# ----------------------------------------------------------------------------------------
report_local() {
    say "--- this box ($(hostname)) ---"
    if command -v chronyc >/dev/null 2>&1 && systemctl is-active --quiet chrony 2>/dev/null; then
        say "  daemon      chrony (can serve time)"
        chronyc tracking 2>/dev/null | sed -n 's/^\(Reference ID\|Stratum\|Last offset\|RMS offset\|Leap status\) *:/  \1 :/p'
        # Only while it IS active -- printing this unconditionally would claim the numbers
        # above are fake on a box that is perfectly well synchronised to an upstream.
        if chronyc tracking 2>/dev/null | grep -q '^Reference ID *: *7F7F0101'; then
            say "              (that is the LOCAL reference: no upstream selected yet, or all of"
            say "               them lost. Clients are still served -- marked stratum 10.)"
        fi
        if [ -f "$CONF" ]; then
            if grep -q '^allow ' "$CONF"; then say "  role        MASTER for the robot LAN"
            else say "  role        client of $(sed -n 's/^server *\([^ ]*\).*/\1/p' "$CONF")"
            fi
            say "  config      $CONF:"
            sed -n 's/^[^#]/    &/p' "$CONF" | sed '/^$/d'
            # `chronyc clients` needs root (it reads chronyd's server stats), so a plain
            # user run says nothing rather than looking empty and wrong.
            if [ "$(id -u)" = 0 ]; then
                say "  clients:"
                chronyc clients 2>/dev/null | sed 's/^/    /' | head -8
            else
                say "  clients     (re-run with sudo to list them)"
            fi
        else
            say "  role        NEITHER -- $CONF is not there, so this box neither serves the"
            say "              robot LAN nor follows it; it is on its own uplink"
        fi
    elif systemctl is-active --quiet systemd-timesyncd 2>/dev/null; then
        say "  daemon      systemd-timesyncd -- an SNTP CLIENT. It cannot serve time, so this"
        say "              box cannot be the LAN's master until chrony replaces it."
        timedatectl show-timesync 2>/dev/null | sed -n 's/^\(ServerName\|ServerAddress\)=/  \1: /p'
        say "  synced      $(timedatectl show -p NTPSynchronized --value 2>/dev/null)"
    else
        say "  daemon      NONE ACTIVE -- this box's clock is free-running off its RTC"
    fi
}

if [ "$MODE" = check ]; then
    say "=== robot LAN time base (master: $MASTER) ==="
    say ""
    report_local
    say "  master      $MASTER answers: $(served "$MASTER")"
    say ""
    # The remote half is a BRAIN-side view. Run on the ZED box itself it would be an ssh to
    # this box's own address, which fails for a confusing reason (its private key lives on
    # the brain, not here) and reads as a broken link.
    ZED_ADDR="${OBK_JETSON_HOST:-$OBK_JETSON_HOST_FALLBACK}"
    ZED_ADDR="${ZED_ADDR#*@}"
    case " $(hostname -I 2>/dev/null) " in
        *" $ZED_ADDR "*)
            say "--- this IS the ZED box ($ZED_ADDR) ---"
            say "  Its offset from the master is above, as \`Last offset\`. The cross-box view"
            say "  belongs on the brain: run this there." ;;
        *)
            say "--- the ZED box, over ssh ---"
            obk_jetson_clock_report || say "  (needs the ssh link: scripts/setup_zed_link.sh)" ;;
    esac
    exit 0
fi

# ----------------------------------------------------------------------------------------
# Both write modes need root. Re-exec under sudo rather than failing on the first write --
# but only when there is a terminal to type a password into, so a scripted call fails loudly
# instead of blocking on an invisible prompt.
# ----------------------------------------------------------------------------------------
if [ "$(id -u)" != 0 ]; then
    if [ -t 0 ]; then
        say "[clock] re-running under sudo (it will ask for your password)..."
        exec sudo -- "${BASH_SOURCE[0]}" "--$MODE" --master-addr "$MASTER"
    fi
    say "setup_clock_sync.sh: --$MODE writes $CONF and needs root, and there is no terminal"
    say "  here to enter a password. Run it yourself:"
    say "    sudo transfer/obelisk/scripts/setup_clock_sync.sh --$MODE"
    exit 1
fi

# ----------------------------------------------------------------------------------------
# chrony, which is the part that needs the network up. Installing it REMOVES
# systemd-timesyncd (they both Provide/Conflict `time-daemon`), which is intended: two
# daemons steering one clock is worse than either.
# ----------------------------------------------------------------------------------------
if ! command -v chronyd >/dev/null 2>&1; then
    say "[clock] installing chrony (this removes systemd-timesyncd -- see the header)..."
    if ! DEBIAN_FRONTEND=noninteractive apt-get install -y chrony; then
        say "setup_clock_sync.sh: apt could not install chrony."
        say "  This box reaches the internet over wifi, not the robot link -- check that the"
        say "  wifi is up (the robot LAN has no route out) and try again."
        exit 1
    fi
fi

mkdir -p "$(dirname "$CONF")"

# A drop-in is only a config if the main file reads it. Ubuntu's chrony.conf does, via
# `confdir` -- but VERIFY rather than assume: on a box whose chrony.conf predates that, the
# file below is one nothing ever parses, and every check afterwards would keep passing
# against the old configuration while reporting the new one.
MAIN=/etc/chrony/chrony.conf
if [ -f "$MAIN" ] && ! grep -qE '^[[:space:]]*(confdir|include)[[:space:]]+.*conf\.d' "$MAIN"; then
    say "[clock] $MAIN does not read conf.d -- adding the include, or the drop-in is inert"
    printf '\n# Added by transfer/obelisk/scripts/setup_clock_sync.sh\nconfdir /etc/chrony/conf.d\n' \
        >>"$MAIN"
fi
if [ "$MODE" = master ]; then
    cat >"$CONF" <<EOF
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Written by transfer/obelisk/scripts/setup_clock_sync.sh --master. This box is the time
# master for the robot LAN; every other box there runs the --client half.

# Serve time to the robot LAN, and ONLY to it. The wifi side stays a client of the internet.
allow $SUBNET

# Keep serving when this box has no uplink of its own. \`orphan\` makes the local reference
# active only while no real source is reachable, so internet time still wins whenever the
# wifi has it -- and a robot on a bench with no AP still has one agreed time base instead of
# every box free-running from its own RTC.
local stratum 10 orphan
EOF
    say "[clock] wrote $CONF (serving $SUBNET)"
else
    cat >"$CONF" <<EOF
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Written by transfer/obelisk/scripts/setup_clock_sync.sh --client. This box follows the
# brain's clock, because the brain is where cross-box ages are computed and bags recorded.

# minpoll/maxpoll 4/6 = every 16-64 s: cheap on a wired LAN and keeps the offset in the
# tens of microseconds. \`prefer\` puts this ahead of any internet pool still configured
# here, and \`trust\` stops a majority of pool servers outvoting it -- being CLOSE TO THE
# BRAIN matters more here than being close to UTC.
server $MASTER iburst minpoll 4 maxpoll 6 prefer trust
EOF
    say "[clock] wrote $CONF (following $MASTER)"
fi

systemctl enable --now chrony >/dev/null 2>&1 || true
systemctl restart chrony || { say "setup_clock_sync.sh: chrony failed to restart."; exit 1; }

# chrony needs a few polls before `tracking` says anything real; iburst makes that seconds,
# not minutes, but it is not instant.
say "[clock] waiting for chrony to take its first samples..."
for _ in $(seq 15); do
    chronyc tracking 2>/dev/null | grep -q "^Reference ID.*[0-9A-F]\{8\}" && break
    sleep 1
done
say ""
report_local
say ""
# Prove the written half is live, rather than merely written.
if [ "$MODE" = master ]; then
    # Ask this box for time the way a client will, over its robot-LAN address. This is the
    # `allow` line taking effect rather than merely existing in a file -- and it is the only
    # check that reports what clients are actually served (see the `served` header).
    say "[clock] a client on $MASTER is answered: $(served "$MASTER")"
    say "[clock] and off-LAN (127.0.0.1) it should refuse: $(served 127.0.0.1)"
else
    if chronyc sources -n 2>/dev/null | grep -q "$MASTER"; then
        say "[clock] $MASTER is in this box's source list."
    else
        say "[clock] WARNING: $MASTER is NOT in \`chronyc sources\`. Either the drop-in is not"
        say "        being read, or the brain is not serving yet (--master over there first)."
    fi
fi
say ""
if [ "$MODE" = master ]; then
    say "[clock] next: on each other box, from its own checkout,"
    say "        sudo transfer/obelisk/scripts/setup_clock_sync.sh --client"
    say "        then back here:  scripts/setup_clock_sync.sh --check"
else
    say "[clock] verify from the brain:  scripts/setup_clock_sync.sh --check"
fi
