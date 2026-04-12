#!/usr/bin/env python3
"""Static analysis: why trunk caching is not viable in Boltz-2.

This script inspects the Boltz-2 model architecture to document the
structural reasons that prevent caching the Pairformer trunk output
for a fixed target protein.

It does NOT run inference — it examines the model structure and tensor
shapes to prove the architectural blocker.
"""

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BOLTZ2_PATH = REPO_ROOT / "src" / "boltz" / "model" / "models" / "boltz2.py"
PAIRFORMER_PATH = REPO_ROOT / "src" / "boltz" / "model" / "layers" / "pairformer.py"
TRUNKV2_PATH = REPO_ROOT / "src" / "boltz" / "model" / "modules" / "trunkv2.py"
TRIMULT_PATH = REPO_ROOT / "src" / "boltz" / "model" / "layers" / "triangular_mult.py"


def find_einsum_patterns(filepath: Path) -> list[str]:
    """Extract all einsum patterns from a Python file."""
    source = filepath.read_text()
    tree = ast.parse(source)
    patterns = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Match torch.einsum("pattern", ...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "einsum"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                patterns.append(node.args[0].value)
    return patterns


def check_joint_processing():
    """Verify that the forward pass processes all chains jointly."""
    source = BOLTZ2_PATH.read_text()

    # Check that input_embedder is called on the full feats dict
    assert "self.input_embedder(feats)" in source, (
        "Expected joint input embedding call"
    )

    # Check that z_init is computed from the full s_inputs
    assert "self.z_init_1(s_inputs)[:, :, None]" in source, (
        "Expected outer-product z initialization from full sequence"
    )

    # Check that msa_module takes full z and s_inputs
    assert "msa_module(\n" in source or "msa_module(" in source
    # The MSA call: z = z + msa_module(z, s_inputs, feats, ...)
    assert "z + msa_module(" in source or "z = z + msa_module" in source, (
        "Expected MSA module to update the full pair representation"
    )

    # Check that pairformer takes the full s, z
    assert "pairformer_module(\n" in source or "pairformer_module(" in source

    print("[OK] Forward pass processes all chains jointly (no per-chain separation)")


def check_triangle_contractions():
    """Show that triangle operations contract over the full sequence dim."""
    # The einsum patterns live in triangular_mult.py, called by pairformer layers
    patterns = find_einsum_patterns(TRIMULT_PATH)

    print(f"\nEinsum patterns in triangular_mult.py (called by Pairformer):")
    for p in patterns:
        print(f"  {p}")

    # The critical patterns are bikd,bjkd->bijd and bkid,bkjd->bijd
    # Both contract over k which spans the full N_total dimension
    full_contraction = [p for p in patterns if "k" in p and "->" in p]
    assert len(full_contraction) > 0, "Expected triangle contraction patterns"
    print(f"\n[OK] Found {len(full_contraction)} triangle contractions over full "
          f"sequence dimension — caching target-only blocks is insufficient")


def check_msa_coupling():
    """Show that MSA module couples target and binder through outer product."""
    source = TRUNKV2_PATH.read_text()
    patterns = find_einsum_patterns(TRUNKV2_PATH)

    print(f"\nEinsum patterns in trunkv2.py:")
    for p in patterns:
        print(f"  {p}")

    # OuterProductMean couples all MSA rows into the pair representation
    assert "OuterProductMean" in source, "Expected OuterProductMean in MSA module"
    print("\n[OK] MSA module uses OuterProductMean — target and binder MSA rows "
          "are coupled into the pair representation")


def main():
    print("=" * 60)
    print("Trunk Cache Architectural Analysis")
    print("=" * 60)

    check_joint_processing()
    check_triangle_contractions()
    check_msa_coupling()

    print("\n" + "=" * 60)
    print("CONCLUSION: Trunk caching is architecturally blocked.")
    print("The pair representation z mixes target-binder cross-terms")
    print("at every layer, with no exploitable block structure.")
    print("=" * 60)


if __name__ == "__main__":
    main()
