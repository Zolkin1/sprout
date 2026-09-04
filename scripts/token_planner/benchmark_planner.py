# Copyright 2026 Zachary Olkin. All rights reserved.

"""Latency of one plan from the autoregressive planner -- spec section B.5.

The question this answers is whether an AR planner can hold a 20 ms control period at
deployment batch 1, where the flow-matching generator it replaces takes 8.3-15 ms per
re-plan (see CLAUDE.md's "The generator's ONNX graph runs entirely on the GPU"). The shapes
of the two problems are different and it is worth being explicit about why:

* the FM runs ``num_sampling_steps`` Euler steps over the WHOLE window, each a full forward
  pass over ``prefix_len + future_len`` tokens;
* the AR planner runs ``N`` steps, each a forward pass over ONE query row against a growing
  key set -- far fewer FLOPs per step, but ``N`` is 16 where the FM's is 8, and at batch 1
  onnxruntime and torch are both LAUNCH-bound rather than FLOP-bound. So the comparison
  cannot be made on arithmetic and has to be measured.

The prefix is encoded ONCE per plan and reported separately, because it is the part that does
not scale with ``N``: a larger token count costs decode steps, not conditioning.

Reported at batch 1 (the deployment case) and at a simulator's env count (the training case),
which are different regimes -- the first is launch-bound and the second is not.

**Torch, not ONNX.** No export exists yet, and the FM's own numbers say the two differ
substantially at batch 1 (its exported graph measured ~0.82x of torch there, being
launch-bound with ~27% more nodes). So these numbers bound the ARCHITECTURE, and an export is
what would bound the deployment.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from robot_rl.token_planner.config import PlannerConfig, TokenizerConfig
from robot_rl.token_planner.planner import TokenPlanner
from robot_rl.token_planner.tokenizer import TrajectoryTokenizer


def parse_args() -> argparse.Namespace:
    """Build the CLI.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(description="Benchmark one AR plan.")
    p.add_argument("--planner", default="",
                   help="A planner.pt to measure. Omit to build an untrained model from the "
                        "shipped defaults -- the timing is architecture, not weights, so an "
                        "untrained model measures the same thing.")
    p.add_argument("--batches", default="1,256,1024",
                   help="Comma-separated batch sizes. 1 is the deployment case.")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true",
                   help="Autocast the decode. Off by default: the deploy path is fp32.")
    p.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True,
                   help="Greedy decoding, as a deployment would use.")
    p.add_argument("--downsample", default="",
                   help="Comma-separated tokenizer downsample factors D to sweep, e.g. "
                        "'2,4,8,16'. Only meaningful without --planner (it changes the "
                        "architecture). D sets the token count N = H / D, which is the "
                        "number of DECODE STEPS -- the quantity per-plan latency is "
                        "expected to scale with at batch 1, where the model is launch-bound "
                        "rather than FLOP-bound. This is the knob that decides whether the "
                        "architecture can hold a control period at all.")
    p.add_argument("--out", default="", help="Write the numbers to this JSON path.")
    return p.parse_args()


def build_model(path: str, device: torch.device,
                downsample: int | None = None) -> TokenPlanner:
    """Load a planner, or build the shipped architecture untrained.

    Args:
        path: A ``planner.pt``, or ``""``.
        device: Target device.
        downsample: Override the tokenizer's ``D`` (hence the token count). Only valid
            without ``path`` -- changing it under a trained planner would build a model its
            weights do not fit.

    Returns:
        The planner in eval mode.

    Raises:
        ValueError: If ``downsample`` is given alongside a checkpoint.
    """
    if path and downsample is not None:
        raise ValueError(
            "--downsample changes the architecture, so it cannot be combined with "
            "--planner; drop --planner to sweep it on the shipped defaults.")
    if path:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PlannerConfig.from_dict(ckpt["config"])
        model = TokenPlanner(cfg, TrajectoryTokenizer(cfg.tokenizer_cfg))
        model.load_state_dict(ckpt.get("model_ema") or ckpt["model"])
        return model.to(device).eval()
    tok = TokenizerConfig() if downsample is None else TokenizerConfig(downsample=int(downsample))
    cfg = PlannerConfig(
        tokenizer=tok.to_dict(), extero_channels=5, extero_image_hw=[[26, 30], [26, 30]],
        extero_camera_intrinsics=[[12.586494467659199, 18.56592408764749, 15.0, 13.0],
                                  [21.42222010113172, 33.86615784101942, 15.0, 13.0]],
    )
    return TokenPlanner(cfg, TrajectoryTokenizer(tok)).to(device).eval()


def timed(fn, iters: int, warmup: int, device: torch.device) -> dict[str, float]:
    """Median and spread of ``fn``'s wall time, in milliseconds.

    Synchronises around every iteration, so a CUDA launch queue cannot make an
    asynchronous call look free. The MEDIAN is the headline and the 90th percentile is
    reported beside it, because a control loop misses its deadline on the tail rather than on
    the average -- the two-box bring-up traced its falls to a chronically-late plan.

    Args:
        fn: Zero-argument callable to time.
        iters: Timed iterations.
        warmup: Untimed iterations first.
        device: For the synchronisation.

    Returns:
        ``{"median_ms", "p90_ms", "min_ms", "max_ms"}``.
    """
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median_ms": statistics.median(times),
        "p90_ms": times[min(int(0.9 * len(times)), len(times) - 1)],
        "min_ms": times[0],
        "max_ms": times[-1],
    }


def main() -> None:
    """Measure and print the per-plan latency."""
    args = parse_args()
    device = torch.device(args.device)
    print(f"[bench] {args.planner or 'untrained defaults'} on {device} "
          f"({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    sweep = [int(v) for v in args.downsample.split(",") if v.strip()] or [None]
    all_results: dict[str, dict] = {}
    for d_factor in sweep:
        model = build_model(args.planner, device, d_factor).eval()
        all_results[str(d_factor or "default")] = _measure(model, args, device)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(all_results, fh, indent=2)
        print(f"[bench] wrote {args.out}")


def _measure(model: TokenPlanner, args: argparse.Namespace,
             device: torch.device) -> dict[str, dict]:
    """Time one architecture at every requested batch size.

    Args:
        model: The planner.
        args: Runtime args.
        device: Where to run.

    Returns:
        Batch size (as a string) -> timings.
    """
    cfg = model.cfg
    print(f"[bench] prefix {model.prefix_len} + {model.num_tokens} output slots "
          f"(D={cfg.tokenizer_cfg.downsample}), d={cfg.d_model} x{cfg.num_layers}, "
          f"V={model.vocab_size}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    results: dict[str, dict] = {}
    for b in [int(v) for v in args.batches.split(",") if v.strip()]:
        hist = torch.randn(b, cfg.num_history_states, cfg.history_dim, device=device)
        extero = torch.randn(b, cfg.extero_sensor_count, cfg.extero_sensor_dim, device=device)
        cmd = (torch.randn(b, len(range(0, cfg.future_len, cfg.cmd_stride)), cfg.cmd_dim,
                           device=device) if cfg.cmd_dim else None)
        gen = torch.Generator(device="cpu").manual_seed(0)

        def _prefix():
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.bf16 and device.type == "cuda"):
                model.encode_prefix(hist, extero, cmd)

        toks = torch.zeros(b, model.num_tokens, dtype=torch.long, device=device)

        def _decode():
            model.decode(toks)

        def _plan():
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=args.bf16 and device.type == "cuda"):
                t, _, _ = model.sample(hist, extero, cmd, greedy=args.greedy,
                                       generator=None if args.greedy else gen)
                model.decode(t)

        with torch.no_grad():
            pre = timed(_prefix, args.iters, args.warmup, device)
            dec = timed(_decode, args.iters, args.warmup, device)
            whole = timed(_plan, args.iters, args.warmup, device)
        # BOTH constants are subtracted before dividing. The prefix encode and the tokenizer
        # DECODE are each once per plan, so folding either into a per-token figure charges a
        # constant to N -- which manufactures exactly the falling curve a `--downsample`
        # sweep exists to look for. Only `sample`'s N decode steps are per-token.
        per_token = ((whole["median_ms"] - pre["median_ms"] - dec["median_ms"])
                     / max(model.num_tokens, 1))
        results[str(b)] = {"prefix": pre, "decode": dec, "plan": whole,
                           "per_token_ms": per_token,
                           "tokens_per_plan": model.num_tokens}
        print(f"[bench] batch {b:>5d}: plan {whole['median_ms']:7.2f} ms "
              f"(p90 {whole['p90_ms']:7.2f})  = prefix {pre['median_ms']:6.2f} "
              f"+ decode {dec['median_ms']:5.2f} + {model.num_tokens} x "
              f"{per_token:5.2f} ms/token")

    one = results.get("1")
    if one:
        m = one["plan"]["median_ms"]
        print(f"[bench] deployment (batch 1): {m:.2f} ms/plan against a 20 ms control "
              f"period -- {'INSIDE' if m < 20.0 else 'OVER'} budget "
              f"({100 * m / 20.0:.0f}% of it). The FM it replaces measures 8.3 ms on an "
              "RTX 4070 and 14-15 ms on the Jetson Thor, both as an exported CUDA-graph "
              "ONNX; this is unexported torch, so it is an architecture bound rather than a "
              "deployment one.\n")
    return results


if __name__ == "__main__":
    main()
