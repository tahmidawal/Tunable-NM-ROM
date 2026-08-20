"""Reproduce the read-only Phase-5 coefficient-rank diagnostic.

This script reads only the immutable, already-exposed P4-D H1 artifact.  It is a
post-hoc geometry diagnostic, not a scientific cell or a field-error lower bound.
Run with single-threaded BLAS for byte-reproducible generated artifacts:

  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    /home/tahmid/Dev/.venv/bin/python \
    experiments/burgers-1e3-10x/b10_phase5_rank_diagnostic.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


HERE = os.path.abspath(os.path.dirname(__file__))
P4_DIR = os.path.join(HERE, "runs", "p4_d_r3")
P4_JSON = os.path.join(P4_DIR, "out", "phase4_d.json")
P4_NPZ = os.path.join(P4_DIR, "out", "phase4_d.npz")
P4_AUDIT = os.path.join(P4_DIR, "out", "AUDIT.json")
P4_MANIFEST = os.path.join(P4_DIR, "MANIFEST.sha256")
OUTPUT_JSON = os.path.join(HERE, "phase5_rank_diagnostic.json")
OUTPUT_MD = os.path.join(HERE, "PHASE-5-RANK-CHECKPOINT.md")

EXPECTED = {
    "phase4_d.json": "ff425dfa1f73ac2d8559df2d780ef09dc1ade179ed5636458e53e0f389f5d617",
    "phase4_d.npz": "720c5890b22709c83858f46e43f18fa3d28bb1305544f02ad9aea71625a2228f",
    "AUDIT.json": "f41010b72ad9ddae43409a1c1d2073dc2839edea22e57a7d5bafc8e2359e08a3",
    "MANIFEST.sha256": "67a3bf8d0d8c755ec39cd4056a0bca0e852b4ba17493e8f052a0b288e63a201a",
}
SEED = 20_260_820
SKETCH_COLUMNS = 96
POWER_ITERATIONS = 2
RANKS = (19, 24, 32, 48, 64, 96)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def diagnostic():
    paths = {
        "phase4_d.json": P4_JSON,
        "phase4_d.npz": P4_NPZ,
        "AUDIT.json": P4_AUDIT,
        "MANIFEST.sha256": P4_MANIFEST,
    }
    actual = {name: sha256(path) for name, path in paths.items()}
    require(actual == EXPECTED, f"immutable P4-D hash mismatch: {actual}")
    report, audit = load_json(P4_JSON), load_json(P4_AUDIT)
    require(report["status"] == "complete", "P4-D is not complete")
    require(report["provenance"]["jax_backend"] == "gpu", "P4-D backend")
    require(report["provenance"]["x64"], "P4-D x64")
    require(report["provenance"]["matmul_precision"] == "highest", "P4-D precision")
    require(not report["config"]["model_validation_touched"], "model validation touched")
    require(not report["config"]["confirmation_touched"], "confirmation touched")
    require(audit["status"] == "pass", "P4-D independent audit")
    require(audit["source_json_sha256"] == actual["phase4_d.json"], "JSON audit binding")
    require(audit["source_npz_sha256"] == actual["phase4_d.npz"], "NPZ audit binding")
    require(audit["manifest_sha256"] == actual["MANIFEST.sha256"], "manifest audit binding")

    blocks, health = [], {}
    with np.load(P4_NPZ, allow_pickle=False) as arrays:
        for n, cases in ((64, 64), (128, 32), (256, 16)):
            prefix = f"H1_N{n}"
            coefficients = np.asarray(arrays[f"{prefix}_coefficients"], np.float64)
            normal = np.asarray(arrays[f"{prefix}_normal"], np.float64)
            healthy = np.asarray(arrays[f"{prefix}_healthy"], bool)
            boundary = np.asarray(arrays[f"{prefix}_boundary"], bool)
            pou = np.asarray(arrays[f"{prefix}_pou"], np.float64)
            support = np.asarray(arrays[f"{prefix}_support"], np.int64)
            require(coefficients.shape == (cases, 51, 3328), f"N{n} coefficient shape")
            require(normal.shape == healthy.shape == boundary.shape == pou.shape
                    == support.shape == (cases, 51), f"N{n} diagnostic shapes")
            require(np.all(np.isfinite(coefficients)) and np.all(np.isfinite(normal)),
                    f"N{n} finite")
            require(np.all(healthy) and np.all(boundary), f"N{n} health/boundary")
            require(float(np.max(normal)) <= 1e-8, f"N{n} normal residual")
            require(float(np.max(np.abs(pou))) == 0.0, f"N{n} exact POU")
            require(int(np.max(support)) <= 32, f"N{n} support")
            blocks.append(coefficients.reshape(-1, 3328))
            health[str(n)] = {
                "snapshots": cases * 51,
                "healthy": int(np.sum(healthy)),
                "normal_worst": float(np.max(normal)),
                "exact_boundary_count": int(np.sum(boundary)),
                "pou_max_abs": float(np.max(np.abs(pou))),
                "support_max": int(np.max(support)),
            }

    matrix = np.concatenate(blocks, axis=0)
    require(matrix.shape == (5712, 3328), "pooled coefficient shape")
    coordinate_mean = np.mean(matrix, axis=0, dtype=np.float64)
    matrix -= coordinate_mean
    centered_frobenius_sq = float(np.sum(matrix * matrix, dtype=np.float64))
    require(centered_frobenius_sq > 0.0, "zero centered coefficient variation")
    rng = np.random.default_rng(SEED)
    omega = rng.normal(size=(matrix.shape[1], SKETCH_COLUMNS))
    sketch = matrix @ omega
    for _ in range(POWER_ITERATIONS):
        basis = np.linalg.qr(sketch, mode="reduced")[0]
        sketch = matrix @ (matrix.T @ basis)
    basis = np.linalg.qr(sketch, mode="reduced")[0]
    compressed = basis.T @ matrix
    singular_values = np.linalg.svd(compressed, compute_uv=False)
    require(singular_values.shape == (SKETCH_COLUMNS,), "compressed SVD shape")
    require(np.all(np.isfinite(singular_values)) and singular_values[0] > 0.0,
            "compressed SVD health")
    summaries = {}
    for rank in RANKS:
        captured = float(np.sum(singular_values[:rank] ** 2) / centered_frobenius_sq)
        residual = float(np.sqrt(max(0.0, 1.0 - captured)))
        summaries[str(rank)] = {
            "estimated_captured_centered_frobenius_fraction": captured,
            "estimated_relative_centered_frobenius_residual": residual,
            "singular_value_over_first": float(singular_values[rank - 1] / singular_values[0]),
        }
    result = {
        "status": "pass",
        "classification": "read-only exposed-development post-hoc geometry diagnostic",
        "scientific_cell": False,
        "promotion_claim": False,
        "field_error_lower_bound": False,
        "interpretation": (
            "The dense final affine layer proves an output affine-hull cap of 32. "
            "The randomized SVD estimates coefficient-space variation outside its best "
            "rank-32 sketch, but is not a rigorous singular-value bound and is not a "
            "field-error or promotion gate."
        ),
        "diagnostic_source_sha256": sha256(os.path.abspath(__file__)),
        "immutable_inputs": actual,
        "phase4_identity": {
            "commit": report["provenance"]["commit"],
            "job_id": report["provenance"]["slurm_job_id"],
            "gpu": report["provenance"]["gpu_kind"],
            "backend": report["provenance"]["jax_backend"],
            "x64": report["provenance"]["x64"],
            "matmul_precision": report["provenance"]["matmul_precision"],
            "model_validation_touched": report["config"]["model_validation_touched"],
            "confirmation_touched": report["config"]["confirmation_touched"],
        },
        "dense_hyperdecoder_geometry": {
            "map": "q19 -> 32 -> 32 -> 3328",
            "last_hidden_dimension": 32,
            "coefficient_dimension": 3328,
            "exact_output_affine_hull_upper_bound": 32,
            "proof": "c(q)=b+W*h(q), hence c(q)-b is in range(W), whose rank is at most 32",
        },
        "selection_health": health,
        "matrix": {
            "row_order": "N64,N128,N256; case-major then time-major within each mesh",
            "shape": [5712, 3328],
            "centering": "per coefficient coordinate across all 5712 exposed snapshots",
            "centered_frobenius_sq": centered_frobenius_sq,
        },
        "randomized_svd": {
            "algorithm": "Gaussian range finder; QR; two subspace power iterations; SVD(Q^T X)",
            "rng": "numpy.random.Generator(PCG64)",
            "seed": SEED,
            "sketch_columns": SKETCH_COLUMNS,
            "power_iterations": POWER_ITERATIONS,
            "required_blas_threads": 1,
            "numpy_version": np.__version__,
            "summaries": summaries,
            "s33_over_s1": float(singular_values[32] / singular_values[0]),
            "reported_numerical_rank_lower_bound_at_s_over_s1_gt_1e-10": int(
                np.sum(singular_values / singular_values[0] > 1e-10)
            ),
            "reported_rank_is_capped_by_sketch_columns": True,
        },
        "phase5_bracket_adaptation_allowed": False,
    }
    return result


def render_json(result):
    return json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_markdown(result):
    rank32 = result["randomized_svd"]["summaries"]["32"]
    return f"""# Phase 5 rank-geometry checkpoint

Status: **read-only exposed-development diagnostic; not a scientific cell or promotion claim**.

The immutable P4-D H1 coefficient artifact and its independent audit pass every hash,
finite, exact-boundary, support, partition-of-unity, and normal-residual check.  The
current dense hyperdecoder has the exact form `q19 -> 32 -> 32 -> 3328`; therefore
`c(q)=b+W*h(q)` lies in one affine coefficient subspace of dimension at most 32.

Using the fixed single-threaded randomized-SVD protocol recorded in
`phase5_rank_diagnostic.json`, the first 32 sketch singular directions capture
`{rank32['estimated_captured_centered_frobenius_fraction']:.16f}` of centered
coefficient Frobenius variation, leaving estimated relative residual
`{rank32['estimated_relative_centered_frobenius_residual']:.16f}`.  The reported
`s33/s1` is `{result['randomized_svd']['s33_over_s1']:.16f}`.  This supports testing
a generator with nonlinear spatial processing after spatialization, but the randomized
quantity is explicitly **not** a rigorous singular-value lower bound, a field-error
lower bound, or an accuracy gate.

The Phase-5 G1/G2 bracket was fixed prospectively before P5-D and may not adapt to
this diagnostic or to P5-D coefficient-SVD results.  Model-validation and confirmation
remain untouched.  No generator was implemented and no scientific job was submitted.

Reproduce/check:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  /home/tahmid/Dev/.venv/bin/python \\
  experiments/burgers-1e3-10x/b10_phase5_rank_diagnostic.py --check
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = diagnostic()
    json_text, markdown_text = render_json(result), render_markdown(result)
    if args.check:
        with open(OUTPUT_JSON) as handle:
            require(handle.read() == json_text, "generated JSON is stale")
        with open(OUTPUT_MD) as handle:
            require(handle.read() == markdown_text, "generated checkpoint is stale")
        print(json.dumps({"status": "pass", "mode": "check"}, sort_keys=True))
        return
    with open(OUTPUT_JSON, "w") as handle:
        handle.write(json_text)
    with open(OUTPUT_MD, "w") as handle:
        handle.write(markdown_text)
    print(json.dumps({"status": "pass", "mode": "write"}, sort_keys=True))


if __name__ == "__main__":
    main()
