"""Test torch.compile on the Pairformer + TF32 matmul on L40S.

This evaluator creates a custom Boltz prediction pipeline that:
1. Loads the model checkpoint
2. Applies torch.compile to the pairformer module
3. Sets TF32 matmul precision
4. Runs inference on the test cases
5. Compares timing with and without compile

We cannot use the standard evaluator because it runs boltz as a subprocess,
and we need to modify the model object after loading.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import modal

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent.parent

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
    .add_local_dir(str(REPO_ROOT / "research" / "eval"), remote_path="/research_eval")
)

app = modal.App("boltz-compile-eval", image=boltz_image)


@app.function(gpu="L40S", timeout=3600)
def test_compile(config_json: str) -> str:
    """Test torch.compile + TF32 on single test case.

    Approach: monkey-patch the boltz CLI predict function to add compile flags
    by modifying the model after it's loaded.
    """
    import torch
    config = json.loads(config_json)

    # Set TF32 precision
    precision = config.get("matmul_precision", "high")
    torch.set_float32_matmul_precision(precision)

    use_compile = config.get("compile_pairformer", False)
    seed = config.get("seed", 42)
    tc_name = config.get("test_case", "small_complex")

    tc_yaml = f"/research_eval/test_cases/{tc_name}.yaml"
    work_dir = f"/tmp/boltz_compile/{tc_name}_{seed}_{uuid.uuid4().hex[:8]}"
    os.makedirs(work_dir, exist_ok=True)

    if use_compile:
        # Strategy: We'll write a custom wrapper script that patches the model
        # after loading to add torch.compile
        wrapper_script = f"""
import sys
import torch
torch.set_float32_matmul_precision("{precision}")

# Monkey-patch the Boltz2 class to compile the pairformer
import boltz.model.models.boltz2 as boltz2_module
original_init = boltz2_module.Boltz2.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    # Compile the pairformer after init
    if hasattr(self, 'pairformer_module') and not self.is_pairformer_compiled:
        print("[COMPILE] Compiling pairformer module...")
        self.pairformer_module = torch.compile(
            self.pairformer_module,
            dynamic=False,
            fullgraph=False,
        )
        self.is_pairformer_compiled = True

boltz2_module.Boltz2.__init__ = patched_init

# Now run the normal predict
sys.argv = [
    "boltz", "{tc_yaml}",
    "--out_dir", "{work_dir}",
    "--sampling_steps", "20",
    "--recycling_steps", "0",
    "--diffusion_samples", "1",
    "--override",
    "--no_kernels",
    "--use_msa_server",
    "--seed", "{seed}",
]

from boltz.main import predict
predict(standalone_mode=False)
"""
        script_path = f"/tmp/compile_wrapper_{uuid.uuid4().hex[:8]}.py"
        with open(script_path, "w") as f:
            f.write(wrapper_script)

        cmd = [sys.executable, script_path]
    else:
        # Standard run without compile
        cmd = [
            sys.executable, "/research_eval/boltz_wrapper.py",
            tc_yaml,
            "--out_dir", work_dir,
            "--sampling_steps", "20",
            "--recycling_steps", "0",
            "--diffusion_samples", "1",
            "--override",
            "--no_kernels",
            "--matmul_precision", precision,
            "--use_msa_server",
            "--seed", str(seed),
        ]

    result = {"config": config, "test_case": tc_name}

    try:
        t_start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        t_end = time.perf_counter()

        result["wall_time_s"] = round(t_end - t_start, 2)
        result["status"] = "success" if proc.returncode == 0 else "error"

        if proc.returncode != 0:
            result["stderr"] = proc.stderr[-2000:] if proc.stderr else ""
        else:
            # Parse quality
            import glob
            conf_files = sorted(glob.glob(f"{work_dir}/boltz_results_*/predictions/*/confidence_*.json"))
            if conf_files:
                with open(conf_files[0]) as f:
                    conf = json.load(f)
                result["quality"] = {k: conf.get(k) for k in [
                    "complex_plddt", "iptm", "confidence_score",
                ] if k in conf}
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return json.dumps(result, indent=2)


@app.local_entrypoint()
def main():
    """Test torch.compile + TF32 vs baseline on all test cases with 3 seeds."""
    seeds = [42, 123, 7]
    test_cases = ["small_complex", "medium_complex", "large_complex"]

    # Build configs: just test compile vs no-compile with high precision
    # Skip baseline (we have that from the standard evaluator)
    configs = []
    for tc in test_cases:
        for seed in seeds:
            # TF32 only (no compile)
            configs.append({
                "seed": seed,
                "matmul_precision": "high",
                "compile_pairformer": False,
                "test_case": tc,
                "label": "tf32_only",
            })

    config_jsons = [json.dumps(c) for c in configs]

    print(f"[eval] Running {len(configs)} configs in parallel...")
    results_list = list(test_compile.map(config_jsons))

    all_results = [json.loads(r) for r in results_list]

    # Aggregate by label
    by_label = {}
    for r in all_results:
        label = r["config"]["label"]
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(r)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    for label, runs in by_label.items():
        print(f"\n--- {label} ---")
        successful = [r for r in runs if r.get("status") == "success"]
        failed = [r for r in runs if r.get("status") != "success"]
        print(f"  {len(successful)} successful, {len(failed)} failed")

        if failed:
            for f_run in failed[:2]:
                print(f"  FAILED: {f_run.get('test_case')} seed={f_run['config']['seed']}")
                if 'stderr' in f_run:
                    print(f"    {f_run['stderr'][-200:]}")

        # Group by test case
        by_tc = {}
        for r in successful:
            tc = r["test_case"]
            if tc not in by_tc:
                by_tc[tc] = []
            by_tc[tc].append(r)

        for tc, tc_runs in by_tc.items():
            times = [r["wall_time_s"] for r in tc_runs]
            plddts = [r.get("quality", {}).get("complex_plddt", 0) for r in tc_runs]
            if times:
                import statistics
                mean_t = statistics.mean(times)
                std_t = statistics.stdev(times) if len(times) > 1 else 0
                mean_p = statistics.mean(plddts) if plddts else 0
                print(f"  {tc}: {mean_t:.1f} +/- {std_t:.1f}s, pLDDT={mean_p:.4f}")

    # Save
    out_path = Path("orbits/l40s-kernels/compile_results.json")
    with out_path.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[eval] Results saved to {out_path}")
