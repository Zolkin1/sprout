# Copyright 2026 Zachary Olkin. All rights reserved.

"""Benchmark the ONNX deploy path, to pick the hardware the policy runs on.

Torch-free (numpy + onnxruntime only), so this runs unchanged in the deployment
container and on a Jetson. Needs nothing but the two ``.onnx`` files.

Five legs, each reported as mean / p50 / p95 / p99 / max:

  1. ``tracker_net``    -- the tracker graph alone (one MLP pass, every control step)
  2. ``fm_net``         -- the FM graph alone. Runs only every ``fm_update_every`` steps,
                           but it unrolls ``num_steps`` Euler steps through a transformer,
                           so it dominates the tick it fires on. This is the number that
                           decides the device.
  3. ``obs_build``      -- the numpy observation assembly. Identical on every target, so
                           it is a floor no accelerator buys back.
  4. ``step_amortized`` -- a full control step, averaged over the FM cadence
  5. ``step_worst``     -- the step the FM fires on. THE BINDING CONSTRAINT: the loop has
                           to hit ``dt`` (20 ms at 50 Hz) on its worst tick, not on average.

Usage::

    # everything available, swept over thread counts, saved for cross-machine diffing
    python transfer/deploy/benchmark.py --run_dir logs/rsl_rl/g1/<run> \\
        --threads 1,2,4,0 --json bench_$(hostname).json

    # graphs only -- no obs assembly, so it needs just the two files
    python transfer/deploy/benchmark.py --tracker tracker.onnx --fm fm.onnx --synthetic

    # measure under GPU contention (the question on a Jetson sharing with the ZED)
    python transfer/deploy/benchmark.py --run_dir <run> --providers cuda --sample-gpu

Reading the results:

* ``step_worst`` p99 against the ``budget`` column is the go/no-go. Everything else
  explains it.
* On a Jetson, **pin the clocks first** -- ``sudo nvpmodel -q`` to see the power mode,
  then ``sudo jetson_clocks``. Without that the governor makes runs unrepeatable and a
  CPU-vs-GPU comparison measures thermal state instead of the graphs.
* A GPU row is only meaningful if ``providers`` in the header actually lists the
  accelerator. A wheel mismatch reports CPU-only rather than failing.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # transfer/

from deploy.fm_session import load_fm_context  # noqa: E402
from deploy.gen_tracker import GenTrackerPolicy  # noqa: E402
from deploy.tracker_session import PROVIDER_ALIASES, TrackerSession, resolve_providers  # noqa: E402

#: Warmup iterations before timing. A GPU needs far more than a CPU: the first calls pay
#: cuDNN/cuBLAS algorithm selection and memory-arena growth, and TensorRT may build an
#: engine outright. Timing those measures the setup, not the steady state.
WARMUP = {"CPUExecutionProvider": 20, "CUDAExecutionProvider": 100,
          "TensorrtExecutionProvider": 100}


def _percentiles(samples: list[float]) -> dict[str, float]:
    """Latency summary in ms from per-call wall times in seconds.

    Percentiles rather than a mean alone: the FM cadence makes the cost bimodal, and a
    mean hides exactly the spikes that break a fixed-rate control loop.

    Args:
        samples: Per-call durations, seconds.

    Returns:
        ``{mean, p50, p95, p99, max, n}`` in milliseconds (``n`` is a count).
    """
    a = np.asarray(samples, dtype=np.float64) * 1e3
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"mean": float(a.mean()), "p50": float(p50), "p95": float(p95),
            "p99": float(p99), "max": float(a.max()), "n": int(a.size)}


def _time_each(fn, iters: int, warmup: int) -> list[float]:
    """Run ``fn`` ``iters`` times after ``warmup``, returning EVERY call's duration.

    Per-call rather than a total/iters mean, because the percentiles are the point.
    """
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


class _GpuSampler:
    """Background sampler for GPU utilization and memory during a run.

    Answers the question that actually decides a shared-GPU deployment: what the policy
    costs while something else (the ZED depth pipeline) is using the same device. Uses
    ``nvidia-smi`` on discrete GPUs and ``tegrastats`` on Jetson, where the integrated GPU
    does not appear in ``nvidia-smi``.
    """

    def __init__(self, period_s: float = 0.25):
        """Pick a backend; become inert if neither tool is present."""
        self.period_s = period_s
        self._util: list[float] = []
        self._mem: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self.error: str | None = None
        self.backend = None
        # tegrastats first: on Jetson the integrated GPU does not appear in nvidia-smi,
        # which is present but reports nothing useful for it.
        if _which("tegrastats"):
            self.backend = "tegrastats"
        elif _which("nvidia-smi"):
            self.backend = "nvidia-smi"

    def _poll_loop(self) -> None:
        """nvidia-smi backend: it exits per call, so poll it on a timer."""
        while not self._stop.wait(self.period_s):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2.0).stdout.strip().splitlines()[0]
                util, mem = (float(x) for x in out.split(","))
                self._util.append(util)
                self._mem.append(mem)
            except Exception:                                 # noqa: BLE001 -- best effort
                pass

    def _stream_loop(self) -> None:
        """tegrastats backend: it STREAMS until killed, so read its lines.

        It never exits on its own, so a ``subprocess.run(..., timeout=...)`` per sample
        times out every time and silently yields nothing -- on the one platform where this
        measurement is the point. Spawn it once and consume stdout instead.
        """
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(max(int(self.period_s * 1000), 100))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception as exc:                              # noqa: BLE001
            self.error = f"could not start tegrastats ({exc})"
            return
        for line in self._proc.stdout:                        # blocks until terminate()
            if self._stop.is_set():
                break
            util, mem, ok = _parse_tegrastats(line)
            if ok:
                self._util.append(util)
                self._mem.append(mem)
            elif self.error is None:
                self.error = f"tegrastats line not understood: {line.strip()[:120]}"

    def start(self) -> None:
        """Begin sampling in the background (no-op without a backend)."""
        if self.backend is None:
            return
        self._stop.clear()
        target = self._stream_loop if self.backend == "tegrastats" else self._poll_loop
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> dict | None:
        """Stop sampling and return ``{backend, util_mean, util_max, mem_max_mib, n}``."""
        if self._thread is None:
            return None
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()                            # unblocks the stdout read
            try:
                self._proc.wait(timeout=3.0)
            except Exception:                                 # noqa: BLE001
                self._proc.kill()
        self._thread.join(timeout=3.0)
        if not self._util:
            return {"backend": self.backend, "n": 0, "error": self.error}
        return {"backend": self.backend, "util_mean": float(np.mean(self._util)),
                "util_max": float(np.max(self._util)),
                "mem_max_mib": float(np.max(self._mem)), "n": len(self._util),
                "error": self.error}


def _which(prog: str) -> bool:
    """Whether ``prog`` is on PATH."""
    from shutil import which  # noqa: PLC0415
    return which(prog) is not None


def _parse_tegrastats(line: str) -> tuple[float, float, bool]:
    """Pull ``(gpu_util_pct, ram_used_mib, parsed_ok)`` out of one tegrastats line.

    Lines look like ``RAM 4096/30536MB ... GR3D_FREQ 45%@1300 ...``; GR3D is the
    integrated GPU. The exact field set varies across JetPack releases, so report whether
    the GPU field was actually found rather than silently returning 0% -- an unparsed
    line and a genuinely idle GPU must not look the same.

    Args:
        line: One line of tegrastats output.

    Returns:
        ``(gpu_util_pct, ram_used_mib, parsed_ok)``; ``parsed_ok`` is False when the GPU
        field was not present.
    """
    import re  # noqa: PLC0415
    # JP5 prints "GR3D_FREQ 45%@1300"; some JP6 builds print "GR3D_FREQ 45%" or "GR3D 45%".
    gpu = re.search(r"GR3D(?:_FREQ)?\s+(\d+)%", line)
    ram = re.search(r"RAM\s+(\d+)/(\d+)MB", line)
    if gpu is None:
        return 0.0, 0.0, False
    return float(gpu.group(1)), float(ram.group(1)) if ram else 0.0, True


def _synthetic_state(policy: GenTrackerPolicy, rng) -> dict:
    """A plausible robot state for the observation-assembly and full-step legs.

    Content does not matter for timing (the graphs are dense), but the SHAPES must be
    exactly what the real runner supplies, or the leg measures a different code path.
    """
    nj = len(policy.get_joint_names())
    k = policy.ctx.num_extra_frames
    state = {
        "root_pos": np.array([0.0, 0.0, 0.79]), "root_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "root_lin_vel_w": rng.standard_normal(3) * 0.1,
        "root_ang_vel_w": rng.standard_normal(3) * 0.1,
        "base_ang_vel": rng.standard_normal(3) * 0.1,
        "joint_pos": rng.standard_normal(nj) * 0.1, "joint_vel": rng.standard_normal(nj) * 0.1,
        "command": np.array([0.5, 0.0, 0.0]),
    }
    if k > 0:
        q = rng.standard_normal((k, 4))
        state.update(frame_pos=rng.standard_normal((k, 3)) * 0.1,
                     frame_quat=q / np.linalg.norm(q, axis=-1, keepdims=True),
                     frame_lin=np.zeros((k, 3)), frame_ang=np.zeros((k, 3)))
    terms = {t["name"]: t["shape"] for t in policy.get_obs_terms()}
    if "height_scan" in terms:
        state["height_scan"] = rng.uniform(-0.3, 0.3, terms["height_scan"])
    spec = policy.get_depth_camera_spec()
    if spec:
        state["depth"] = np.stack([
            rng.uniform(-1.0, 1.0, int(c["width"]) * int(c["height"])) for c in spec])
    return state


def _fm_feed(ctx, rng) -> dict:
    """Synthetic inputs for the FM graph, in its declared shapes."""
    feed = {
        "history": rng.standard_normal((1, ctx.N, ctx.D)).astype(np.float32),
        "extero": rng.standard_normal((1, ctx.extero_tokens * ctx.height_dim)).astype(np.float32),
        "cmd_tokens": rng.standard_normal((1, ctx.num_cmd_tokens, ctx.cmd_dim)).astype(np.float32),
        "noise": rng.standard_normal((1, ctx.F, ctx.D)).astype(np.float32),
    }
    if ctx.has_history_null:
        feed["history_null"] = np.zeros((1, 1), dtype=np.float32)
    return feed


def _bench_graphs(tracker_path, fm_path, providers, threads, iters, rng) -> dict:
    """Time the two ONNX graphs in isolation (no observation assembly)."""
    res = {}
    trk = TrackerSession(tracker_path, threads, providers)
    obs = rng.standard_normal((1, trk.get_num_obs())).astype(np.float32)
    warm = max(WARMUP.get(p, 20) for p in trk.sess.get_providers())
    res["tracker_net"] = _percentiles(_time_each(lambda: trk.infer(obs), iters, warm))
    res["_providers"] = trk.sess.get_providers()
    if fm_path:
        ctx = load_fm_context(fm_path, threads, providers)
        feed = _fm_feed(ctx, rng)
        res["fm_net"] = _percentiles(
            _time_each(lambda: ctx.sess.run(["trajectory"], feed), max(iters // 5, 20), warm))
        res["_fm"] = {"F": ctx.F, "D": ctx.D, "N": ctx.N, "num_steps": ctx.num_steps}
    return res


def _bench_steps(tracker_path, fm_path, providers, threads, iters, rng) -> dict:
    """Time observation assembly and the full control step, split by FM cadence."""
    policy = GenTrackerPolicy(tracker_path, fm_path, num_threads=threads,
                              zero_noise=True, providers=providers)
    policy.load()
    every = policy.fm_update_every
    state = _synthetic_state(policy, rng)
    names = policy.get_joint_names()
    policy.reset()

    warm = max(WARMUP.get(p, 20) for p in policy._trk.sess.get_providers())
    # Enough steps to see many FM ticks even at a slow cadence.
    n = max(iters, every * 20)
    for _ in range(warm):
        policy.get_action(policy.create_obs(state), names)

    # Split by whether the FM fires. _update_generator increments _ctrl_step first, so the
    # tick that re-plans is the one where (step + 1) % every == 0.
    fm_ticks, plain_ticks, obs_only = [], [], []
    for _ in range(n):
        fires = ((policy._ctrl_step + 1) % every) == 0
        t0 = time.perf_counter()
        obs = policy.create_obs(state)
        t1 = time.perf_counter()
        policy.get_action(obs, names)
        t2 = time.perf_counter()
        (fm_ticks if fires else plain_ticks).append(t2 - t0)
        if not fires:
            obs_only.append(t1 - t0)          # assembly cost without the FM sample in it

    out = {"step_amortized": _percentiles(fm_ticks + plain_ticks)}
    if obs_only:
        out["obs_build"] = _percentiles(obs_only)
    if fm_ticks:
        out["step_worst"] = _percentiles(fm_ticks)
    out["_dt_ms"] = policy.get_dt() * 1e3
    out["_fm_update_every"] = every
    return out


def _machine_info() -> dict:
    """Host + runtime identification, so a saved result says what produced it."""
    info = {
        "hostname": platform.node(), "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "ort_device": ort.get_device(),
        "available_providers": ort.get_available_providers(),
    }
    try:
        info["cpu_model"] = next(
            l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo")           # noqa: PTH123
            if l.startswith(("model name", "Model")))
    except Exception:                                          # noqa: BLE001 -- cosmetic
        pass
    if _which("nvidia-smi"):
        try:
            info["gpu"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5.0).stdout.strip()
        except Exception:                                      # noqa: BLE001 -- cosmetic
            pass
    if os.path.exists("/etc/nv_tegra_release"):                # noqa: PTH110
        info["jetson"] = open("/etc/nv_tegra_release").readline().strip()       # noqa: PTH123
    # The scaling governor silently rescales every number here. A 'powersave' box can read
    # 2-3x slower than the same box under 'performance', which would send the device
    # comparison the wrong way -- so record it and warn rather than let it stay invisible.
    gov = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov):                                    # noqa: PTH110
        info["cpu_governor"] = open(gov).read().strip()                         # noqa: PTH123
    return info


def _print_table(rows: list[dict]) -> None:
    """Print one line per (provider, threads, leg), with the budget verdict."""
    hdr = f"{'provider':<14}{'thr':>4}  {'leg':<16}{'mean':>8}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}   verdict"
    print("\n" + hdr)
    print("-" * len(hdr))
    for row in rows:
        dt = row.get("dt_ms")
        for leg in ("tracker_net", "fm_net", "obs_build", "step_amortized", "step_worst"):
            s = row.get(leg)
            if s is None:
                continue
            verdict = ""
            if leg == "step_worst" and dt:
                head = dt - s["p99"]
                verdict = (f"{'OK  ' if head > 0 else 'OVER'} budget {dt:.0f} ms, "
                           f"headroom {head:+.1f} ms")
            print(f"{row['provider']:<14}{row['threads']:>4}  {leg:<16}"
                  f"{s['mean']:>8.2f}{s['p50']:>8.2f}{s['p95']:>8.2f}{s['p99']:>8.2f}"
                  f"{s['max']:>8.2f}   {verdict}")


def main() -> None:
    """Run the benchmark over every requested provider and thread count."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", help="Run directory (its exported/ holds the two .onnx files).")
    p.add_argument("--tracker", help="Path to tracker.onnx (overrides --run_dir).")
    p.add_argument("--fm", help="Path to fm.onnx (overrides --run_dir).")
    p.add_argument("--iters", type=int, default=300, help="Timed iterations per leg.")
    p.add_argument("--threads", default="0",
                   help="Comma-separated intra-op thread counts to sweep; 0 = ORT auto.")
    p.add_argument("--providers", default="",
                   help="Comma-separated providers (cpu,cuda,trt). Default: every one "
                        "onnxruntime reports as available.")
    p.add_argument("--synthetic", action="store_true",
                   help="Graphs only -- skip observation assembly and the full-step legs. "
                        "Needs nothing but the .onnx files.")
    p.add_argument("--sample-gpu", action="store_true", dest="sample_gpu",
                   help="Sample GPU utilization/memory during the run (nvidia-smi or tegrastats).")
    p.add_argument("--json", help="Write the full result to this path.")
    args = p.parse_args()

    if args.tracker:
        tracker_path, fm_path = args.tracker, args.fm
    elif args.run_dir:
        exp = args.run_dir if os.path.basename(args.run_dir) == "exported" else \
            os.path.join(args.run_dir, "exported")
        tracker_path = os.path.join(exp, "tracker.onnx")
        fm_path = os.path.join(exp, "fm.onnx")
        fm_path = fm_path if os.path.exists(fm_path) else None
    else:
        p.error("pass --run_dir or --tracker")
    if not os.path.exists(tracker_path):
        p.error(f"no tracker.onnx at {tracker_path}")

    info = _machine_info()
    print("=" * 78)
    print(f"host      {info['hostname']}  ({info['machine']}, {info['cpu_count']} cpus)")
    print(f"cpu       {info.get('cpu_model', info['processor'])}")
    if "gpu" in info:
        print(f"gpu       {info['gpu']}")
    if "jetson" in info:
        print(f"jetson    {info['jetson']}")
    print(f"ort       {info['onnxruntime']}  device={info['ort_device']}")
    print(f"providers {info['available_providers']}")
    if "cpu_governor" in info:
        note = "" if info["cpu_governor"] == "performance" else \
            "   <-- NOT 'performance': these numbers are throttled and not comparable " \
            "across machines"
        print(f"governor  {info['cpu_governor']}{note}")
    print(f"tracker   {tracker_path}")
    print(f"fm        {fm_path or '(none)'}")
    print("=" * 78)

    available = ort.get_available_providers()
    if args.providers:
        wanted = [PROVIDER_ALIASES.get(x.strip().lower(), x.strip())
                  for x in args.providers.split(",") if x.strip()]
        for w in wanted:
            if w not in available:
                p.error(f"provider '{w}' is unavailable; onnxruntime has {available}. "
                        "A CUDA/TensorRT provider needs the onnxruntime-gpu wheel (and on "
                        "Jetson, NVIDIA's JetPack-matched build, not the PyPI aarch64 one).")
    else:
        # Azure EP is a remote-inference shim, never a compute target here.
        wanted = [x for x in available if x != "AzureExecutionProvider"]
    thread_counts = [int(t) for t in args.threads.split(",") if t.strip() != ""]

    sampler = _GpuSampler() if args.sample_gpu else None
    if sampler is not None and sampler.backend is None:
        print("[WARN] --sample-gpu: neither tegrastats nor nvidia-smi found; not sampling.")
    if sampler is not None:
        sampler.start()

    rows, meta = [], {}
    for prov in wanted:
        for threads in thread_counts:
            rng = np.random.default_rng(0)
            row = {"provider": prov.replace("ExecutionProvider", ""), "threads": threads}
            g = _bench_graphs(tracker_path, fm_path, [prov], threads, args.iters, rng)
            meta.setdefault("fm", g.pop("_fm", None))
            row["actual_providers"] = g.pop("_providers")
            row.update(g)
            if not args.synthetic:
                s = _bench_steps(tracker_path, fm_path, [prov], threads, args.iters, rng)
                row["dt_ms"] = s.pop("_dt_ms")
                meta["fm_update_every"] = s.pop("_fm_update_every")
                row.update(s)
            rows.append(row)
            # A provider that silently fell back to CPU makes its row a duplicate, not a
            # measurement -- say so rather than letting it read as a GPU result.
            if prov not in row["actual_providers"]:
                print(f"[WARN] requested {prov} but the session runs on "
                      f"{row['actual_providers']} -- that row is NOT a {prov} measurement.")

    gpu = sampler.stop() if sampler is not None else None
    _print_table(rows)
    if gpu:
        if gpu.get("n"):
            print(f"\ngpu load ({gpu['backend']}): util mean {gpu['util_mean']:.0f}% "
                  f"max {gpu['util_max']:.0f}%, mem max {gpu['mem_max_mib']:.0f} MiB, "
                  f"{gpu['n']} samples")
        else:
            print(f"\n[WARN] --sample-gpu collected NO samples from {gpu['backend']}"
                  + (f": {gpu['error']}" if gpu.get("error") else "")
                  + ". Treat the GPU-load line as missing, not as an idle GPU.")
        if gpu.get("error") and gpu.get("n"):
            print(f"       (sampler note: {gpu['error']})")
    if meta.get("fm"):
        f = meta["fm"]
        print(f"\nfm graph: F={f['F']} D={f['D']} N={f['N']}, {f['num_steps']} Euler steps "
              f"per call, fired every {meta.get('fm_update_every', '?')} control steps")
    print("\nstep_worst is the tick the FM fires on -- the loop must hit dt on THAT tick.")

    if args.json:
        with open(args.json, "w") as fh:                                        # noqa: PTH123
            json.dump({"machine": info, "meta": meta, "gpu_load": gpu,
                       "tracker": os.path.abspath(tracker_path),
                       "fm": os.path.abspath(fm_path) if fm_path else None,
                       "iters": args.iters, "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
