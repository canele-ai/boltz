"""Profile Boltz-2 inference time breakdown and test optimizations.

Phase 1: Profile the 20-step/0-recycle configuration to find where time is spent.
Phase 2: Test optimizations targeting identified bottlenecks.

Usage:
    cd /home/liambai/code/boltz/.worktrees/combined-fast
    /home/liambai/code/boltz/.venv/bin/modal run orbits/combined-fast/profile_and_optimize.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Resolve eval directory for image build
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_candidates = [
    _script_dir / ".." / ".." / "research" / "eval",
    Path("/home/liambai/code/boltz/.worktrees/combined-fast/research/eval"),
    Path("/home/liambai/code/boltz/research/eval"),
]
EVAL_DIR = None
for c in _candidates:
    if c.resolve().exists():
        EVAL_DIR = c.resolve()
        break
if EVAL_DIR is None:
    EVAL_DIR = Path("/eval")

boltz_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.5.1",
        "numpy>=1.26,<2.0",
        "pyyaml==6.0.2",
        "boltz==2.2.1",
    )
    .add_local_dir(str(EVAL_DIR), remote_path="/eval")
)

app = modal.App("boltz-combined-fast", image=boltz_image)


# ---------------------------------------------------------------------------
# Phase 1: Profiling wrapper
# ---------------------------------------------------------------------------

PROFILING_WRAPPER_CODE = r'''
"""Profiling wrapper: times each phase of Boltz-2 inference."""
import sys
import time
import json
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--matmul_precision", default="highest")
    parser.add_argument("--compile_pairformer", action="store_true")
    parser.add_argument("--compile_structure", action="store_true")
    parser.add_argument("--compile_confidence", action="store_true")
    parser.add_argument("--compile_msa", action="store_true")
    parser.add_argument("--timing_output", default=None,
                       help="Path to write timing JSON")
    our_args, boltz_args = parser.parse_known_args()

    import torch
    torch.set_float32_matmul_precision(our_args.matmul_precision)

    timings = {}

    # Time the import
    t0 = time.perf_counter()
    import boltz.main as boltz_main
    timings["import_time"] = time.perf_counter() - t0

    # Monkey-patch process_inputs to time MSA generation
    original_process_inputs = boltz_main.process_inputs
    def timed_process_inputs(*args, **kwargs):
        t = time.perf_counter()
        result = original_process_inputs(*args, **kwargs)
        timings["process_inputs_time"] = time.perf_counter() - t
        return result
    boltz_main.process_inputs = timed_process_inputs

    # Monkey-patch load_from_checkpoint to time model loading
    from boltz.model.models.boltz2 import Boltz2
    original_load = Boltz2.load_from_checkpoint
    @classmethod
    def timed_load(cls, *args, **kwargs):
        t = time.perf_counter()
        result = original_load.__func__(cls, *args, **kwargs)
        timings["model_load_time"] = time.perf_counter() - t
        return result
    Boltz2.load_from_checkpoint = timed_load

    # Monkey-patch predict_step to time the forward pass
    original_predict_step = Boltz2.predict_step
    def timed_predict_step(self, batch, batch_idx, dataloader_idx=0):
        torch.cuda.synchronize()
        t = time.perf_counter()
        result = original_predict_step(self, batch, batch_idx, dataloader_idx)
        torch.cuda.synchronize()
        timings["predict_step_time"] = time.perf_counter() - t
        return result
    Boltz2.predict_step = timed_predict_step

    # Override sys.argv and run
    sys.argv = [sys.argv[0]] + boltz_args

    t_total = time.perf_counter()
    boltz_main.predict()
    timings["total_predict_time"] = time.perf_counter() - t_total

    # Write timing output
    if our_args.timing_output:
        with open(our_args.timing_output, "w") as f:
            json.dump(timings, f, indent=2)
    print(f"[TIMINGS] {json.dumps(timings)}")

if __name__ == "__main__":
    main()
'''


def run_boltz_with_timing(
    tc_yaml: Path,
    work_dir: Path,
    config: dict,
    wrapper_path: str = "/eval/boltz_wrapper.py",
    use_profiling: bool = False,
) -> dict:
    """Run Boltz prediction, optionally with profiling wrapper."""
    import subprocess
    import uuid

    timing_file = work_dir / "timings.json"

    if use_profiling:
        # Write profiling wrapper
        wrapper_path = str(work_dir / "profiling_wrapper.py")
        with open(wrapper_path, "w") as f:
            f.write(PROFILING_WRAPPER_CODE)

    cmd = [
        sys.executable, wrapper_path,
        str(tc_yaml),
        "--out_dir", str(work_dir),
        "--sampling_steps", str(config.get("sampling_steps", 200)),
        "--recycling_steps", str(config.get("recycling_steps", 3)),
        "--diffusion_samples", str(config.get("diffusion_samples", 1)),
        "--override",
        "--no_kernels",
    ]

    # MSA handling
    msa_dir = config.get("msa_directory")
    if msa_dir:
        cmd.extend(["--msa_directory", str(msa_dir)])
    else:
        cmd.append("--use_msa_server")

    seed = config.get("seed")
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    precision = config.get("matmul_precision", "highest")
    cmd.extend(["--matmul_precision", precision])

    # Extra CLI args for boltz (these pass through to boltz predict)
    extra_args = config.get("extra_boltz_args", [])
    cmd.extend(extra_args)

    if use_profiling:
        cmd.extend(["--timing_output", str(timing_file)])

    result = {
        "wall_time_s": None,
        "quality": {},
        "error": None,
        "timings": {},
    }

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        t_end = time.perf_counter()
        result["wall_time_s"] = t_end - t_start

        if proc.returncode != 0:
            result["error"] = f"Exit {proc.returncode}: {proc.stderr[-2000:]}"
            return result

        # Parse timings from profiling wrapper output
        if use_profiling and timing_file.exists():
            with open(timing_file) as f:
                result["timings"] = json.load(f)
        elif use_profiling:
            # Try parsing from stdout
            for line in proc.stdout.split("\n"):
                if "[TIMINGS]" in line:
                    timing_json = line.split("[TIMINGS]")[1].strip()
                    result["timings"] = json.loads(timing_json)
                    break

        # Parse confidence metrics
        target_name = tc_yaml.stem
        results_dir = (
            work_dir / f"boltz_results_{target_name}" / "predictions" / target_name
        )
        if not results_dir.exists():
            pred_base = work_dir / f"boltz_results_{target_name}" / "predictions"
            if pred_base.exists():
                subdirs = [d for d in pred_base.iterdir() if d.is_dir()]
                if subdirs:
                    results_dir = subdirs[0]

        if results_dir.exists():
            conf_files = sorted(results_dir.glob("confidence_*.json"))
            if conf_files:
                with conf_files[0].open() as f:
                    conf = json.load(f)
                for key in [
                    "confidence_score", "ptm", "iptm", "ligand_iptm",
                    "protein_iptm", "complex_plddt", "complex_iplddt",
                    "complex_pde", "complex_ipde",
                ]:
                    if key in conf:
                        result["quality"][key] = conf[key]

    except subprocess.TimeoutExpired:
        result["error"] = "Timed out after 1800s"
    except Exception as exc:
        result["error"] = f"Error: {exc}"

    return result


@app.function(gpu="L40S", timeout=7200)
def profile_config(config_json: str) -> str:
    """Profile a single config on L40S, timing each component."""
    import uuid
    import yaml

    config = json.loads(config_json)

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])
    results = {"config": config, "per_complex": []}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"YAML not found: {tc_yaml}",
            })
            continue

        work_dir = Path(f"/tmp/boltz_profile/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir.mkdir(parents=True, exist_ok=True)

        print(f"[profile] Running {tc_name} with profiling...")
        pred_result = run_boltz_with_timing(
            tc_yaml, work_dir, config, use_profiling=True
        )

        results["per_complex"].append({
            "name": tc_name,
            "wall_time_s": pred_result["wall_time_s"],
            "timings": pred_result["timings"],
            "quality": pred_result["quality"],
            "error": pred_result["error"],
        })

        if pred_result["error"]:
            print(f"  ERROR: {pred_result['error'][:200]}")
        else:
            print(f"  Wall time: {pred_result['wall_time_s']:.1f}s")
            for k, v in pred_result["timings"].items():
                print(f"  {k}: {v:.2f}s")
            plddt = pred_result["quality"].get("complex_plddt", "N/A")
            print(f"  pLDDT: {plddt}")

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=7200)
def evaluate_config(config_json: str, num_runs: int = 1) -> str:
    """Standard evaluation for a config (no profiling overhead)."""
    import statistics
    import uuid
    import yaml

    config = json.loads(config_json)
    DEFAULT_CONFIG = {
        "sampling_steps": 200,
        "recycling_steps": 3,
        "matmul_precision": "highest",
        "diffusion_samples": 1,
        "seed": 42,
    }
    merged = {**DEFAULT_CONFIG, **config}

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])
    results = {"config": merged, "num_runs": num_runs, "per_complex": [], "aggregate": {}}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({
                "name": tc_name, "error": f"YAML not found", "wall_time_s": None, "quality": {},
            })
            continue

        run_times = []
        run_qualities = []
        last_error = None

        for run_idx in range(num_runs):
            work_dir = Path(f"/tmp/boltz_eval/{tc_name}_{uuid.uuid4().hex[:8]}")
            work_dir.mkdir(parents=True, exist_ok=True)

            print(f"[eval] {tc_name} run {run_idx+1}/{num_runs} "
                  f"steps={merged['sampling_steps']} recycle={merged['recycling_steps']}")

            pred_result = run_boltz_with_timing(tc_yaml, work_dir, merged)

            if pred_result["error"]:
                last_error = pred_result["error"]
                break

            if pred_result["wall_time_s"] is not None:
                run_times.append(pred_result["wall_time_s"])
            run_qualities.append(pred_result["quality"])

        if last_error:
            entry = {"name": tc_name, "wall_time_s": None, "quality": {}, "error": last_error}
        else:
            median_time = statistics.median(run_times) if run_times else None
            all_plddts = [q.get("complex_plddt") for q in run_qualities if q.get("complex_plddt") is not None]
            mean_plddt = sum(all_plddts) / len(all_plddts) if all_plddts else None
            merged_quality = dict(run_qualities[-1]) if run_qualities else {}
            if mean_plddt is not None:
                merged_quality["complex_plddt"] = mean_plddt
            entry = {
                "name": tc_name, "wall_time_s": median_time,
                "quality": merged_quality, "error": None, "run_times": run_times,
            }

        results["per_complex"].append(entry)

    # Compute aggregates
    successful = [r for r in results["per_complex"] if r["error"] is None and r["wall_time_s"] is not None]

    if len(successful) < len(test_cases):
        failed = [r["name"] for r in results["per_complex"] if r.get("error")]
        results["aggregate"] = {"error": f"Failed: {failed}", "speedup": 0, "passes_quality_gate": False}
    elif successful:
        total_time = sum(r["wall_time_s"] for r in successful)
        mean_time = total_time / len(successful)
        plddts = [r["quality"]["complex_plddt"] for r in successful if "complex_plddt" in r["quality"]]
        iptms = [r["quality"]["iptm"] for r in successful if "iptm" in r["quality"]]

        agg = {
            "num_successful": len(successful),
            "total_wall_time_s": total_time,
            "mean_wall_time_s": mean_time,
            "mean_plddt": sum(plddts) / len(plddts) if plddts else None,
            "mean_iptm": sum(iptms) / len(iptms) if iptms else None,
        }

        baseline = eval_config.get("baseline", {})
        if baseline:
            bl_time = baseline.get("mean_wall_time_s")
            bl_plddt = baseline.get("mean_plddt")
            if bl_time and mean_time > 0:
                agg["speedup"] = bl_time / mean_time
            if bl_plddt and plddts:
                mean_p = sum(plddts) / len(plddts)
                agg["plddt_delta_pp"] = (mean_p - bl_plddt) * 100.0
                regression = (bl_plddt - mean_p) * 100.0
                agg["passes_quality_gate"] = regression <= 2.0

                if baseline.get("per_complex"):
                    bl_by_name = {pc["name"]: pc for pc in baseline["per_complex"]}
                    violations = {}
                    for r in successful:
                        bl_case = bl_by_name.get(r["name"])
                        if bl_case and bl_case.get("complex_plddt") is not None:
                            case_plddt = r["quality"].get("complex_plddt")
                            if case_plddt is not None:
                                case_reg = (bl_case["complex_plddt"] - case_plddt) * 100.0
                                if case_reg > 5.0:
                                    agg["passes_quality_gate"] = False
                                    violations[r["name"]] = f"-{case_reg:.1f}pp"
                    if violations:
                        agg["per_complex_regression"] = violations

        results["aggregate"] = agg

    return json.dumps(results, indent=2)


@app.function(gpu="L40S", timeout=7200)
def profile_with_msa_caching(config_json: str) -> str:
    """Two-pass profiling: first pass generates MSA, second pass uses cached MSA.

    This measures the MSA-free inference time to understand the true GPU bottleneck.
    """
    import subprocess
    import uuid
    import yaml

    config = json.loads(config_json)

    config_path = Path("/eval/config.yaml")
    with config_path.open() as f:
        eval_config = yaml.safe_load(f)

    test_cases = eval_config.get("test_cases", [])
    results = {"config": config, "per_complex": []}

    for tc in test_cases:
        tc_name = tc["name"]
        tc_yaml = Path("/eval") / tc["yaml"]

        if not tc_yaml.exists():
            results["per_complex"].append({"name": tc_name, "error": f"YAML not found"})
            continue

        # Pass 1: Run with MSA server to generate and cache MSAs
        work_dir_1 = Path(f"/tmp/boltz_msa_gen/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir_1.mkdir(parents=True, exist_ok=True)

        print(f"[msa-cache] Pass 1: Generating MSA for {tc_name}...")
        config_pass1 = {**config, "msa_directory": None}  # Force use_msa_server
        if "msa_directory" in config_pass1:
            del config_pass1["msa_directory"]
        result_pass1 = run_boltz_with_timing(tc_yaml, work_dir_1, config_pass1, use_profiling=True)

        if result_pass1["error"]:
            results["per_complex"].append({
                "name": tc_name, "error": f"Pass 1 failed: {result_pass1['error'][:200]}",
                "pass1_time": result_pass1["wall_time_s"],
            })
            continue

        # Find the cached MSA directory
        target_name = tc_yaml.stem
        msa_dir = work_dir_1 / f"boltz_results_{target_name}" / "processed" / "msa"

        if not msa_dir.exists():
            results["per_complex"].append({
                "name": tc_name,
                "error": f"MSA dir not found at {msa_dir}",
                "pass1_time": result_pass1["wall_time_s"],
                "pass1_timings": result_pass1["timings"],
            })
            continue

        print(f"  Pass 1 done: {result_pass1['wall_time_s']:.1f}s")
        for k, v in result_pass1["timings"].items():
            print(f"    {k}: {v:.2f}s")

        # Pass 2: Run with cached MSA (no server call)
        work_dir_2 = Path(f"/tmp/boltz_cached/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir_2.mkdir(parents=True, exist_ok=True)

        print(f"[msa-cache] Pass 2: Running with cached MSA for {tc_name}...")
        config_pass2 = {**config, "msa_directory": str(msa_dir)}
        result_pass2 = run_boltz_with_timing(tc_yaml, work_dir_2, config_pass2, use_profiling=True)

        if result_pass2["error"]:
            results["per_complex"].append({
                "name": tc_name,
                "error": f"Pass 2 failed: {result_pass2['error'][:200]}",
                "pass1_time": result_pass1["wall_time_s"],
                "pass1_timings": result_pass1["timings"],
            })
            continue

        print(f"  Pass 2 done: {result_pass2['wall_time_s']:.1f}s")
        for k, v in result_pass2["timings"].items():
            print(f"    {k}: {v:.2f}s")

        # Pass 3: Run again with cached MSA to measure steady-state
        work_dir_3 = Path(f"/tmp/boltz_cached2/{tc_name}_{uuid.uuid4().hex[:8]}")
        work_dir_3.mkdir(parents=True, exist_ok=True)

        print(f"[msa-cache] Pass 3: Second cached run for {tc_name}...")
        result_pass3 = run_boltz_with_timing(tc_yaml, work_dir_3, config_pass2, use_profiling=True)

        entry = {
            "name": tc_name,
            "pass1_with_msa_server": {
                "wall_time_s": result_pass1["wall_time_s"],
                "timings": result_pass1["timings"],
                "quality": result_pass1["quality"],
            },
            "pass2_cached_msa": {
                "wall_time_s": result_pass2["wall_time_s"],
                "timings": result_pass2["timings"],
                "quality": result_pass2["quality"],
            },
            "pass3_cached_msa_warmcache": {
                "wall_time_s": result_pass3["wall_time_s"] if not result_pass3["error"] else None,
                "timings": result_pass3.get("timings", {}),
                "quality": result_pass3.get("quality", {}),
                "error": result_pass3.get("error"),
            },
            "msa_savings_s": (
                (result_pass1["wall_time_s"] or 0) - (result_pass2["wall_time_s"] or 0)
            ),
            "error": None,
        }
        results["per_complex"].append(entry)

        plddt1 = result_pass1["quality"].get("complex_plddt", "N/A")
        plddt2 = result_pass2["quality"].get("complex_plddt", "N/A")
        print(f"  MSA savings: {entry['msa_savings_s']:.1f}s")
        print(f"  pLDDT server: {plddt1}, pLDDT cached: {plddt2}")

    return json.dumps(results, indent=2)


@app.local_entrypoint()
def main():
    """Run profiling and optimization experiments."""
    import os

    outdir = Path("orbits/combined-fast")
    outdir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Phase 1: Profile the 20-step/0-recycle config
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: Profile 20-step/0-recycle with MSA caching analysis")
    print("=" * 70)

    profile_config_dict = {
        "sampling_steps": 20,
        "recycling_steps": 0,
        "seed": 42,
    }

    profile_result_json = profile_with_msa_caching.remote(
        json.dumps(profile_config_dict)
    )
    profile_result = json.loads(profile_result_json)

    with open(outdir / "profile_results.json", "w") as f:
        json.dump(profile_result, f, indent=2)

    print("\n[Phase 1] Profile results saved.")

    # Print summary
    for pc in profile_result.get("per_complex", []):
        name = pc.get("name", "?")
        if pc.get("error"):
            print(f"  {name}: ERROR - {pc['error'][:100]}")
            continue
        p1 = pc.get("pass1_with_msa_server", {})
        p2 = pc.get("pass2_cached_msa", {})
        p3 = pc.get("pass3_cached_msa_warmcache", {})
        print(f"\n  {name}:")
        print(f"    With MSA server:     {p1.get('wall_time_s', '?'):.1f}s")
        print(f"    Cached MSA (cold):   {p2.get('wall_time_s', '?'):.1f}s")
        if p3.get("wall_time_s"):
            print(f"    Cached MSA (warm):   {p3['wall_time_s']:.1f}s")
        print(f"    MSA savings:         {pc.get('msa_savings_s', '?'):.1f}s")

    # -----------------------------------------------------------------------
    # Phase 2: Run optimized configs through the standard evaluator
    # Each config gets its own L40S GPU via .map()
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: Test optimization configurations")
    print("=" * 70)

    # Config A: Baseline fast (20 steps, 0 recycles) - reproduce parent orbit
    # Config B: 20 steps, 0 recycles, TF32 precision (stack with step reduction)
    # Config C: 15 steps, 0 recycles (push step count lower)
    # Config D: 20 steps, 0 recycles, fewer MSA seqs
    # Config E: 10 steps, 0 recycles (verify cliff - parent said catastrophic)

    opt_configs = [
        {
            "label": "A: 20s/0r (reproduce parent)",
            "config": {"sampling_steps": 20, "recycling_steps": 0, "seed": 42},
        },
        {
            "label": "B: 20s/0r + TF32",
            "config": {"sampling_steps": 20, "recycling_steps": 0, "matmul_precision": "high", "seed": 42},
        },
        {
            "label": "C: 15s/0r (push lower)",
            "config": {"sampling_steps": 15, "recycling_steps": 0, "seed": 42},
        },
        {
            "label": "D: 12s/0r (aggressive)",
            "config": {"sampling_steps": 12, "recycling_steps": 0, "seed": 42},
        },
    ]

    config_jsons = [json.dumps(c["config"]) for c in opt_configs]
    num_runs_list = [1] * len(opt_configs)

    print(f"Running {len(opt_configs)} configs in parallel on L40S GPUs...")

    phase2_results = []
    for i, result_json in enumerate(evaluate_config.map(config_jsons, num_runs_list)):
        result = json.loads(result_json)
        agg = result.get("aggregate", {})
        label = opt_configs[i]["label"]
        print(f"\n  {label}:")
        print(f"    Speedup: {agg.get('speedup', 'N/A')}")
        print(f"    Mean time: {agg.get('mean_wall_time_s', 'N/A')}")
        print(f"    Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
        print(f"    Delta: {agg.get('plddt_delta_pp', 'N/A')} pp")
        print(f"    Gate: {agg.get('passes_quality_gate', 'N/A')}")
        phase2_results.append({"label": label, "result": result})

    with open(outdir / "optimization_results.json", "w") as f:
        json.dump(phase2_results, f, indent=2)

    print("\n[Phase 2] Results saved.")

    # -----------------------------------------------------------------------
    # Phase 3: Validate best config with 3 seeds
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: Validate best config with 3 seeds")
    print("=" * 70)

    # Find the best passing config
    best_config = None
    best_speedup = 0
    best_label = None

    for pr in phase2_results:
        agg = pr["result"].get("aggregate", {})
        if agg.get("passes_quality_gate") and (agg.get("speedup", 0) > best_speedup):
            best_speedup = agg["speedup"]
            best_config = pr["result"]["config"]
            best_label = pr["label"]

    if best_config is None:
        print("No config passed quality gate! Using 20s/0r as fallback.")
        best_config = {"sampling_steps": 20, "recycling_steps": 0}
        best_label = "fallback: 20s/0r"

    print(f"Best config: {best_label} (speedup={best_speedup:.2f}x)")
    print("Validating with seeds 42, 123, 7...")

    validation_configs = []
    for seed in [42, 123, 7]:
        cfg = {**best_config, "seed": seed}
        validation_configs.append(cfg)

    val_jsons = [json.dumps(c) for c in validation_configs]
    val_runs = [1] * len(validation_configs)

    validation_results = []
    for i, result_json in enumerate(evaluate_config.map(val_jsons, val_runs)):
        result = json.loads(result_json)
        agg = result.get("aggregate", {})
        seed = validation_configs[i]["seed"]
        print(f"\n  Seed {seed}:")
        print(f"    Speedup: {agg.get('speedup', 'N/A')}")
        print(f"    Mean time: {agg.get('mean_wall_time_s', 'N/A')}")
        print(f"    Mean pLDDT: {agg.get('mean_plddt', 'N/A')}")
        validation_results.append({"seed": seed, "result": result})

    with open(outdir / "validation_results.json", "w") as f:
        json.dump({
            "best_config": best_config,
            "best_label": best_label,
            "validation": validation_results,
        }, f, indent=2)

    # Compute mean across seeds
    speedups = []
    plddts = []
    times = []
    for vr in validation_results:
        agg = vr["result"].get("aggregate", {})
        if agg.get("speedup"):
            speedups.append(agg["speedup"])
        if agg.get("mean_plddt"):
            plddts.append(agg["mean_plddt"])
        if agg.get("mean_wall_time_s"):
            times.append(agg["mean_wall_time_s"])

    if speedups:
        import statistics
        mean_speedup = sum(speedups) / len(speedups)
        std_speedup = statistics.stdev(speedups) if len(speedups) > 1 else 0
        mean_plddt = sum(plddts) / len(plddts) if plddts else 0
        mean_time = sum(times) / len(times) if times else 0

        print(f"\n{'='*70}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Config: {best_label}")
        print(f"Speedup: {mean_speedup:.2f}x +/- {std_speedup:.2f}")
        print(f"Mean time: {mean_time:.1f}s")
        print(f"Mean pLDDT: {mean_plddt:.4f}")

    print("\nDone. All results in orbits/combined-fast/")
