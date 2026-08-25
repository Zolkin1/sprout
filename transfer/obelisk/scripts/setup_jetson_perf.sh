#!/usr/bin/env bash
# Copyright 2026 Zachary Olkin. All rights reserved.
#
# Pin the brain's clocks and power budget, on the HOST. Nothing in the launch path does
# this, so by default a Jetson runs the controller under DYNAMIC governors -- which is the
# wrong setting for this particular workload, for a reason that is measurable rather than
# folklore.
#
# WHY IT MATTERS HERE, specifically:
#   The generator's ONNX graph is ~3052 kernels for one plan at batch 1, so onnxruntime is
#   LAUNCH-bound: wall time tracks the kernel COUNT. It runs once per `fm_update_every`
#   ticks -- 12 at 50 Hz, so every 240 ms -- and between re-plans the GPU is near idle.
#   Under `nvhost_podgov` it therefore clocks DOWN (measured here: 315 MHz against a
#   1386 MHz ceiling), and every re-plan starts on a cold clock and ramps while it runs.
#   That is a direct candidate for both the ~8% pass-to-pass variance this box shows where
#   the desktop is stable to ~1%, and for the chronically-late plan the two-box bring-up
#   traced its falls to. Pinning removes the ramp; it does not make the graph cheaper.
#
# MAXN (mode 0) vs 120W (mode 1), measured from /etc/nvpmodel.conf and
# /sys/class/devfreq/gpu-gpc-0/available_frequencies rather than assumed:
#   * CPU gains NOTHING. 120W already allows 2601 MHz, which is cpuinfo_max_freq.
#   * GPU ceiling goes 1386 MHz -> 1575 MHz (+13.6%), and the SoC-level 120W budget cap
#     comes off entirely. The cap coming off is the part to think about, not the extra
#     189 MHz: this is a battery-powered robot carrying a 2.64 kg compute backpack, and
#     MAXN is by definition unbudgeted.
#
# ============================ WHY THE UNIT SETS THE MODE TOO ============================
# The obvious version of this script sets nvpmodel once and lets `nvpmodel.service` replay
# it, because a mode is widely described as sticky -- it is written to
# /var/lib/nvpmodel/status and that service is enabled. DO NOT RELY ON THAT HERE. The boot
# script is:
#
#     /etc/systemd/nvpmodel.sh  ->  /usr/sbin/nvpmodel -f /etc/nvpmodel.conf
#
# with no `-m`, and `nvpmodel --help` says of exactly that form: "If it is the only option,
# then it sets the power mode as default mode configured in the file." The file says
# `< PM_CONFIG DEFAULT=1 >` -- 120W. So whether a saved MAXN survives a reboot depends on
# whether the status file or the conf DEFAULT wins, and ON THIS BOX THE TWO CANNOT BE
# DISTINGUISHED BY OBSERVATION: the saved status is 1 and the conf default is 1, the same
# number. A wrong answer here is silent and expensive -- the robot runs a whole session at
# 120W while the operator believes it is on MAXN, and the only symptom is a generator that
# is intermittently late, which looks like every other cause of that.
#
# So the mode is not trusted to persist, it is RE-ASSERTED: the installed unit runs
# `nvpmodel -m <mode>` and then `jetson_clocks`, ordered After=nvpmodel.service (which is
# Type=oneshot, so the ordering is a real barrier and not a race). Persistence is then a
# property of this unit rather than of undocumented nvpmodel behaviour, and --check reads
# the result back so the claim is verifiable after any reboot.
#
# --force is deliberately NOT passed to nvpmodel: it means "automatically reboot if
# required when switching the power mode", which inside a boot-time unit is a boot loop.
# MAXN and 120W online the same 14 cores and share GPU_PG_MASK, so no reboot is required
# between them anyway.
#
# jetson_clocks itself is a one-shot write to sysfs and has never been sticky, which is the
# other half of what the unit is for.
#
# MUST RUN ON THE HOST. The obk container has neither nvpmodel nor write access to the
# sysfs nodes involved, so this cannot be folded into obk_launch.sh's container half.
#
# Usage:
#   scripts/setup_jetson_perf.sh                          # --check: read-only, no sudo
#   sudo scripts/setup_jetson_perf.sh --apply --mode 0 --install    # MAXN, every boot
#   sudo scripts/setup_jetson_perf.sh --apply             # pin at the current model only
#   sudo scripts/setup_jetson_perf.sh --restore           # governors back, unit removed
set -uo pipefail

MODE=check
PMODE=""
INSTALL=""
UNIT_NAME=obk-jetson-perf.service
UNIT=/etc/systemd/system/$UNIT_NAME
GPU_DEVFREQ=/sys/class/devfreq/gpu-gpc-0
# An explicit, PERSISTENT store path. jetson_clocks defaults --store/--restore to
# ${HOME}/l4t_dfs.conf, which under sudo is whatever HOME survived the escalation -- so a
# --restore can silently find no file, or the wrong one. Named here so both halves agree.
STORE=/var/lib/obk-jetson-clocks.conf

usage() { sed -n '3,66p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [ $# -gt 0 ]; do
    case "$1" in
        --check)   MODE=check; shift ;;
        --apply)   MODE=apply; shift ;;
        --restore) MODE=restore; shift ;;
        --mode)    PMODE="${2:?--mode needs a power-model id (0=MAXN, 1=120W)}"; shift 2 ;;
        --install) INSTALL=1; shift ;;
        -h|--help) usage ;;
        *) echo "setup_jetson_perf.sh: unknown arg '$1'" >&2; exit 2 ;;
    esac
done

if [ ! -x /usr/sbin/nvpmodel ] || [ ! -x /usr/bin/jetson_clocks ]; then
    echo "[perf] nvpmodel/jetson_clocks not found -- this is not a Jetson host, or you are" >&2
    echo "[perf] inside the container. Run it on the brain's HOST." >&2
    exit 1
fi

# ---------------------------------------------------------------------------------------
# report: the numbers that actually say whether this box is pinned, read from sysfs rather
# than from `nvpmodel -q` (which needs root) so --check works as a plain user and is
# therefore usable as a post-reboot verification step.
# ---------------------------------------------------------------------------------------
report() {
    local pmode name gov gpu_gov gpu_min gpu_max gpu_cur cpu_max cpu_min unit_mode
    pmode="$(sed -n 's/^pmode:0*\([0-9]\+\).*/\1/p' /var/lib/nvpmodel/status 2>/dev/null)"
    pmode="${pmode:-0}"
    name="$(sed -n "s/^< POWER_MODEL ID=${pmode} NAME=\([^ ]*\) >/\1/p" /etc/nvpmodel.conf 2>/dev/null)"
    gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"
    cpu_min="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_min_freq 2>/dev/null)"
    cpu_max="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null)"
    gpu_gov="$(cat $GPU_DEVFREQ/governor 2>/dev/null)"
    gpu_min="$(cat $GPU_DEVFREQ/min_freq 2>/dev/null)"
    gpu_max="$(cat $GPU_DEVFREQ/max_freq 2>/dev/null)"
    gpu_cur="$(cat $GPU_DEVFREQ/cur_freq 2>/dev/null)"

    echo "[perf] power model : ${pmode} (${name:-unknown})"
    echo "[perf] cpu         : governor=${gov} $((cpu_min/1000))-$((cpu_max/1000)) MHz"
    echo "[perf] gpu         : governor=${gpu_gov} $((gpu_min/1000000))-$((gpu_max/1000000)) MHz, now $((gpu_cur/1000000)) MHz"

    # The verdict is min==max on BOTH domains, NOT the governor's name. jetson_clocks on
    # this SoC pins by writing scaling_min_freq = scaling_max_freq and leaves the governor
    # called `schedutil`, so testing for `performance` would report NOT PINNED after a
    # perfectly successful apply -- a false alarm that would send you to re-run it forever.
    # A governor with no range left to move in is pinned whatever it is named.
    if [ -n "$gpu_min" ] && [ "$gpu_min" = "$gpu_max" ] && [ "$cpu_min" = "$cpu_max" ]; then
        echo "[perf] PINNED     : both domains min==max; a re-plan pays no clock ramp."
    else
        echo "[perf] NOT PINNED : the GPU scales between re-plans, so each plan starts on a"
        echo "[perf]              cold clock. Fix: sudo $0 --apply --mode 0 --install"
    fi

    # The reboot verdict names the mode the UNIT will assert, not the mode that happens to
    # be live -- those differ exactly when someone changed the mode by hand after install.
    if systemctl is-enabled --quiet "$UNIT_NAME" 2>/dev/null; then
        unit_mode="$(sed -n 's|^ExecStart=/usr/sbin/nvpmodel -m \([0-9]\+\)$|\1|p' "$UNIT" 2>/dev/null)"
        echo "[perf] reboot     : $UNIT_NAME enabled -- asserts mode ${unit_mode:-<none>} then jetson_clocks."
        [ -n "$unit_mode" ] && [ "$unit_mode" != "$pmode" ] && \
            echo "[perf] WARNING    : live mode $pmode != the mode the unit asserts ($unit_mode)."
    else
        echo "[perf] reboot     : NOT persistent. Neither the mode nor the clocks are re-asserted"
        echo "[perf]              at boot (nvpmodel.sh runs \`nvpmodel -f\` with no -m, and the"
        echo "[perf]              conf default is $(sed -n 's/^< PM_CONFIG DEFAULT=\([0-9]\+\) >/\1/p' /etc/nvpmodel.conf)). Add --install."
    fi
}

case "$MODE" in
check)   report; exit 0 ;;
restore)
    [ "$(id -u)" = 0 ] || { echo "[perf] needs sudo." >&2; exit 1; }
    if systemctl is-enabled --quiet "$UNIT_NAME" 2>/dev/null; then
        systemctl disable "$UNIT_NAME" >/dev/null 2>&1
        rm -f "$UNIT"; systemctl daemon-reload
        echo "[perf] removed $UNIT_NAME"
    fi
    if [ -f "$STORE" ]; then
        jetson_clocks --restore "$STORE" && echo "[perf] governors restored from $STORE"
    else
        echo "[perf] no $STORE -- governors left as they are (reboot restores the defaults)." >&2
    fi
    report; exit 0 ;;
esac

[ "$(id -u)" = 0 ] || { echo "[perf] --apply needs sudo." >&2; exit 1; }

# Snapshot the PRE-PINNED governors once, and only once: a --store taken after the first
# apply would record the maxed state, which makes --restore a no-op that reports success.
if [ ! -f "$STORE" ]; then
    jetson_clocks --store "$STORE" >/dev/null 2>&1 \
        && echo "[perf] saved pre-pin governor state to $STORE"
fi

# Power model FIRST: jetson_clocks pins min=max at the ACTIVE model's ceiling, so a mode
# change after it would leave the pin at the old ceiling.
if [ -n "$PMODE" ]; then
    echo "[perf] nvpmodel -m $PMODE"
    nvpmodel -m "$PMODE" </dev/null || {
        echo "[perf] nvpmodel failed. The clocks below are NOT applied." >&2; exit 1; }
fi

echo "[perf] jetson_clocks"
jetson_clocks || { echo "[perf] jetson_clocks failed." >&2; exit 1; }

if [ -n "$INSTALL" ]; then
    # The mode baked in here is the one asserted at every boot. Default to whatever is live
    # now, so `--apply --install` with no --mode installs a unit consistent with the box.
    inst_mode="${PMODE:-$(sed -n 's/^pmode:0*\([0-9]\+\).*/\1/p' /var/lib/nvpmodel/status)}"
    inst_mode="${inst_mode:-0}"
    # Two ExecStart lines, in order: assert the mode, then pin at its ceiling. See the
    # header for why the mode is re-asserted rather than trusted to persist.
    # TimeoutStartSec bounds the boot cost if nvpmodel ever decides to ask a question --
    # a oneshot with no tty would otherwise be able to stall the boot.
    cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Assert the obelisk brain's power model and pin its clocks
Documentation=file://$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/setup_jetson_perf.sh
After=nvpmodel.service
Wants=nvpmodel.service

[Service]
Type=oneshot
RemainAfterExit=yes
TimeoutStartSec=60
StandardInput=null
ExecStart=/usr/sbin/nvpmodel -m $inst_mode
ExecStart=/usr/bin/jetson_clocks

[Install]
WantedBy=multi-user.target
UNIT_EOF
    systemctl daemon-reload
    systemctl enable "$UNIT_NAME" >/dev/null
    echo "[perf] installed $UNIT (asserts mode $inst_mode + jetson_clocks at every boot)"
fi

report
