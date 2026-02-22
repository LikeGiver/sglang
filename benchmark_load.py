"""
Benchmark script to compare serial vs parallel module loading in sglang diffusion pipelines.

Usage:
    python benchmark_load.py --mode serial
    python benchmark_load.py --mode parallel
    python benchmark_load.py --mode both
"""

import argparse
import gc
import json
import os
import sys
import time

# Must be set before importing sglang / torch internals
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

import torch  # noqa: E402

_DEFAULT_MODEL_PATH = "/data/codes/2026_02/sglang/models/FLUX.2-klein-4B"


def get_gpu_memory_gb():
    """Return (allocated_GiB, reserved_GiB) or None if CUDA not available."""
    if not torch.cuda.is_available():
        return None
    alloc = torch.cuda.memory_allocated() / (1024**3)
    resv = torch.cuda.memory_reserved() / (1024**3)
    return alloc, resv


def fmt_mem(mem):
    if mem is None:
        return "CUDA not available"
    return f"allocated={mem[0]:.2f} GiB, reserved={mem[1]:.2f} GiB"


def ensure_distributed():
    """Initialise the distributed environment required by the loaders."""
    torch.cuda.set_device(0)

    from sglang.multimodal_gen.runtime.distributed import (
        maybe_init_distributed_environment_and_model_parallel,
    )

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=1,
        enable_cfg_parallel=False,
    )


def make_server_args(model_path: str, parallel: bool):
    """Build ServerArgs with the given parallel_loading flag."""
    sglang_argv = [
        "--model-path", model_path,
        "--port", "30099",
        "--num-gpus", "1",
    ]
    if parallel:
        sglang_argv.extend(["--parallel-loading", "true"])
    else:
        sglang_argv.extend(["--parallel-loading", "false"])

    # prepare_server_args inspects sys.argv for provided-arg detection
    original_argv = sys.argv
    sys.argv = ["benchmark_load.py"] + sglang_argv
    try:
        from sglang.multimodal_gen.runtime.server_args import prepare_server_args
        return prepare_server_args(sglang_argv)
    finally:
        sys.argv = original_argv


def print_timeline(timings: dict[str, tuple[float, float]], total_time: float, label: str):
    """Print an ASCII timeline showing when each component started and finished.

    timings: {component_name: (start_seconds, end_seconds)} relative to load start
    """
    if not timings:
        return

    # Sort by start time
    sorted_items = sorted(timings.items(), key=lambda kv: kv[1][0])
    max_end = max(end for _, end in timings.values())
    bar_width = 60

    print(f"\n  Timeline: {label}")
    print(f"  {'Component':<20} {'Start':>6} {'End':>6} {'Dur':>6}  {'|' + '-' * bar_width + '|'}")

    for name, (t_start, t_end) in sorted_items:
        duration = t_end - t_start
        # Map to bar positions
        if max_end > 0:
            col_start = int(t_start / max_end * bar_width)
            col_end = int(t_end / max_end * bar_width)
        else:
            col_start = 0
            col_end = bar_width

        col_end = max(col_end, col_start + 1)  # at least 1 char wide

        bar = "." * col_start + "#" * (col_end - col_start) + "." * (bar_width - col_end)
        print(f"  {name:<20} {t_start:>5.2f}s {t_end:>5.2f}s {duration:>5.2f}s  |{bar}|")

    # Print time axis
    axis_labels = f"  {'':20} {'0.00s':>6} {'':>6} {'':>6}  |"
    mid_label = f"{max_end / 2:.1f}s"
    end_label = f"{max_end:.1f}s"
    mid_pos = bar_width // 2
    axis_line = " " * (mid_pos - len(mid_label) // 2) + mid_label
    axis_line = axis_line + " " * (bar_width - len(axis_line) - len(end_label)) + end_label + "|"
    print(f"  {'':20} {'':>6} {'':>6} {'':>6}  |{axis_line}")


def run_benchmark(model_path: str, mode_label: str, parallel: bool):
    """Run a single benchmark and return a results dict."""
    server_args = make_server_args(model_path, parallel=parallel)

    # Must set global server_args so component loaders can access it
    from sglang.multimodal_gen.runtime.server_args import set_global_server_args
    set_global_server_args(server_args)

    print(f"\n{'=' * 70}")
    print(f"  Benchmark: {mode_label.upper()} module loading")
    print(f"  parallel_loading = {server_args.parallel_loading}")
    print(f"  Model: {model_path}")
    print(f"  GPU memory before: {fmt_mem(get_gpu_memory_gb())}")
    print(f"{'=' * 70}\n")

    from sglang.multimodal_gen.runtime.pipelines.flux_2_klein import (
        Flux2KleinPipeline,
    )

    torch.cuda.synchronize()
    start = time.perf_counter()

    pipeline = Flux2KleinPipeline(
        model_path=model_path,
        server_args=server_args,
    )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    mem_after = get_gpu_memory_gb()
    modules = list(pipeline.modules.keys())
    mem_usages = dict(pipeline.memory_usages)
    load_timings = dict(pipeline.load_timings)
    phase_timings = dict(pipeline.phase_timings)

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {mode_label.upper()} loading took {elapsed:.2f}s")
    print(f"  GPU memory after: {fmt_mem(mem_after)}")
    print(f"  Modules loaded: {modules}")
    print(f"  Per-module memory (GiB): {mem_usages}")

    # Per-component timing table
    if load_timings:
        print(f"\n  Per-component timing:")
        print(f"  {'Component':<20} {'Start':>8} {'End':>8} {'Duration':>10}")
        print(f"  {'-' * 48}")
        for name, (t_start, t_end) in sorted(load_timings.items(), key=lambda kv: kv[1][0]):
            print(f"  {name:<20} {t_start:>7.2f}s {t_end:>7.2f}s {t_end - t_start:>9.2f}s")

    # Per-component phase breakdown
    if phase_timings:
        print(f"\n  Phase breakdown:")
        print(f"  {'Component.Phase':<35} {'Start':>8} {'End':>8} {'Duration':>10}")
        print(f"  {'-' * 63}")
        for comp_name in sorted(phase_timings.keys()):
            for phase_name, ps, pe in phase_timings[comp_name]:
                label = f"{comp_name}.{phase_name}"
                print(f"  {label:<35} {ps:>7.2f}s {pe:>7.2f}s {pe - ps:>9.2f}s")

    # ASCII timeline
    print_timeline(load_timings, elapsed, mode_label.upper())

    # Phase-level timeline
    if phase_timings:
        # Build flat dict for print_timeline: "component.phase" -> (start, end)
        flat_phases = {}
        for comp_name in sorted(phase_timings.keys()):
            for phase_name, ps, pe in phase_timings[comp_name]:
                flat_phases[f"{comp_name}.{phase_name}"] = (ps, pe)
        print_timeline(flat_phases, elapsed, f"{mode_label.upper()} PHASES")

    print(f"{'=' * 70}\n")

    # Cleanup
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "mode": mode_label,
        "elapsed": elapsed,
        "mem_after": mem_after,
        "modules": modules,
        "mem_usages": mem_usages,
        "load_timings": load_timings,
        "phase_timings": phase_timings,
    }


def print_comparison(serial_result, parallel_result):
    """Print a side-by-side comparison table."""
    s = serial_result
    p = parallel_result
    speedup = s["elapsed"] / p["elapsed"] if p["elapsed"] > 0 else float("inf")

    print(f"\n{'=' * 70}")
    print("  COMPARISON: Serial vs Parallel Module Loading")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<30} {'Serial':>15} {'Parallel':>15}")
    print(f"  {'-' * 60}")
    print(f"  {'Wall time (s)':<30} {s['elapsed']:>15.2f} {p['elapsed']:>15.2f}")
    print(f"  {'Speedup':<30} {'1.00x':>15} {speedup:>14.2f}x")

    if s["mem_after"] and p["mem_after"]:
        print(
            f"  {'GPU allocated (GiB)':<30} {s['mem_after'][0]:>15.2f} {p['mem_after'][0]:>15.2f}"
        )
        print(
            f"  {'GPU reserved (GiB)':<30} {s['mem_after'][1]:>15.2f} {p['mem_after'][1]:>15.2f}"
        )

    print(f"  {'Modules loaded':<30} {len(s['modules']):>15} {len(p['modules']):>15}")

    # Per-component memory
    all_keys = sorted(set(list(s["mem_usages"].keys()) + list(p["mem_usages"].keys())))
    if all_keys:
        print(f"\n  {'Per-component memory (GiB)':<30} {'Serial':>15} {'Parallel':>15}")
        print(f"  {'-' * 60}")
        for k in all_keys:
            sv = s["mem_usages"].get(k, 0.0)
            pv = p["mem_usages"].get(k, 0.0)
            print(f"  {k:<30} {sv:>15.3f} {pv:>15.3f}")

    # Per-component duration comparison
    s_timings = s.get("load_timings", {})
    p_timings = p.get("load_timings", {})
    timing_keys = sorted(set(list(s_timings.keys()) + list(p_timings.keys())))
    if timing_keys:
        print(f"\n  {'Per-component duration (s)':<30} {'Serial':>15} {'Parallel':>15}")
        print(f"  {'-' * 60}")
        for k in timing_keys:
            s_dur = (s_timings[k][1] - s_timings[k][0]) if k in s_timings else 0.0
            p_dur = (p_timings[k][1] - p_timings[k][0]) if k in p_timings else 0.0
            print(f"  {k:<30} {s_dur:>15.2f} {p_dur:>15.2f}")

    # Side-by-side timelines
    print(f"\n  --- Serial Timeline ---")
    if s_timings:
        s_max = max(end for _, end in s_timings.values())
        _print_compact_timeline(s_timings, s_max)

    print(f"\n  --- Parallel Timeline ---")
    if p_timings:
        p_max = max(end for _, end in p_timings.values())
        _print_compact_timeline(p_timings, p_max)

    print(f"\n{'=' * 70}\n")

    # Overlap analysis for parallel
    if p_timings and len(p_timings) > 1:
        _print_overlap_analysis(p_timings)

    # Correctness check
    if sorted(s["modules"]) == sorted(p["modules"]):
        print("  Correctness: modules match between serial and parallel")
    else:
        print("  WARNING: module lists differ!")
        print(f"    Serial:   {sorted(s['modules'])}")
        print(f"    Parallel: {sorted(p['modules'])}")


def _print_compact_timeline(timings, max_end):
    """Print a compact ASCII timeline."""
    bar_width = 50
    for name, (t_start, t_end) in sorted(timings.items(), key=lambda kv: kv[1][0]):
        duration = t_end - t_start
        if max_end > 0:
            col_start = int(t_start / max_end * bar_width)
            col_end = int(t_end / max_end * bar_width)
        else:
            col_start = 0
            col_end = bar_width
        col_end = max(col_end, col_start + 1)
        bar = "." * col_start + "#" * (col_end - col_start) + "." * (bar_width - col_end)
        print(f"    {name:<18} {t_start:>5.2f}-{t_end:>5.2f}s ({duration:>5.2f}s) |{bar}|")

    print(f"    {'':18} {'':>13}         |{'0':1}{'':>{bar_width // 2 - 1}}{max_end / 2:.0f}{'':>{bar_width - bar_width // 2 - len(f'{max_end:.0f}')}}{max_end:.0f}|")


def _print_overlap_analysis(timings):
    """Analyze and print which components had overlapping execution."""
    items = sorted(timings.items(), key=lambda kv: kv[1][0])
    overlaps = []
    for i, (name_a, (start_a, end_a)) in enumerate(items):
        for name_b, (start_b, end_b) in items[i + 1:]:
            overlap_start = max(start_a, start_b)
            overlap_end = min(end_a, end_b)
            if overlap_start < overlap_end:
                overlaps.append((name_a, name_b, overlap_end - overlap_start))

    if overlaps:
        print("  Overlap analysis (parallel):")
        for name_a, name_b, overlap_dur in overlaps:
            print(f"    {name_a} + {name_b}: {overlap_dur:.2f}s overlap")
    else:
        print("  Overlap analysis: NO overlaps detected (components ran sequentially)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark serial vs parallel module loading for sglang diffusion pipelines"
    )
    parser.add_argument(
        "--mode",
        choices=["serial", "parallel", "both"],
        default="both",
        help="Which loading mode(s) to benchmark (default: both)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=_DEFAULT_MODEL_PATH,
        help=f"Path to the model (default: {_DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to write JSON report (e.g. report.json). "
        "Contains per-component start/end times for graphing.",
    )
    args = parser.parse_args()
    model_path = args.model_path

    # Initialise distributed environment (required by loaders)
    ensure_distributed()

    serial_result = None
    parallel_result = None

    if args.mode in ("serial", "both"):
        serial_result = run_benchmark(model_path, "serial", parallel=False)

    if args.mode in ("parallel", "both"):
        parallel_result = run_benchmark(model_path, "parallel", parallel=True)

    if serial_result and parallel_result:
        print_comparison(serial_result, parallel_result)
    elif serial_result:
        print(f"\nFinal: serial loading = {serial_result['elapsed']:.2f}s")
    elif parallel_result:
        print(f"\nFinal: parallel loading = {parallel_result['elapsed']:.2f}s")

    # Export JSON report
    output_path = args.output
    if output_path is None:
        output_path = "benchmark_report.json"

    report = _build_report(serial_result, parallel_result)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report written to: {output_path}")


def _build_report(serial_result, parallel_result):
    """Build a JSON-serialisable report dict."""
    def _format_result(r):
        if r is None:
            return None
        # load_timings: {name: (start, end)} -> list of dicts for easy graphing
        phase_data = r.get("phase_timings", {})
        components = []
        for name, (t_start, t_end) in sorted(r["load_timings"].items(), key=lambda kv: kv[1][0]):
            phases = []
            for phase_name, ps, pe in phase_data.get(name, []):
                phases.append({
                    "phase": phase_name,
                    "start": round(ps, 4),
                    "end": round(pe, 4),
                    "duration": round(pe - ps, 4),
                })
            components.append({
                "name": name,
                "start": round(t_start, 4),
                "end": round(t_end, 4),
                "duration": round(t_end - t_start, 4),
                "memory_gib": round(r["mem_usages"].get(name, 0.0), 4),
                "phases": phases,
            })
        return {
            "mode": r["mode"],
            "total_seconds": round(r["elapsed"], 4),
            "gpu_allocated_gib": round(r["mem_after"][0], 4) if r["mem_after"] else None,
            "gpu_reserved_gib": round(r["mem_after"][1], 4) if r["mem_after"] else None,
            "modules": r["modules"],
            "components": components,
        }

    report = {}
    if serial_result:
        report["serial"] = _format_result(serial_result)
    if parallel_result:
        report["parallel"] = _format_result(parallel_result)
    if serial_result and parallel_result:
        report["speedup"] = round(
            serial_result["elapsed"] / parallel_result["elapsed"], 4
        )
    return report


if __name__ == "__main__":
    main()
