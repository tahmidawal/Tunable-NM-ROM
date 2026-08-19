"""Audit pulled nonlinear-decoder cells and their timing provenance.

This is deliberately independent of the result summarizer: a numerically
attractive row cannot enter the report merely because it parses successfully.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import statistics


HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")

# These artifacts remain useful for diagnosis, but their timing values cannot
# support a result.  The reasons are rendered into the generated audit report.
EXCLUDED = {
    "nda_pbench_g98_r8": "exact kernels were warmed before the post-fit GPU burn",
    "nda_be2e_g160_r12_failed": "driver lost compact decoder metadata and aborted",
    "nda_pe2e_g98_r11": "raw timing repetitions were not persisted",
    "nda_be2e_g160_r14": "raw timing repetitions were not persisted",
    "nda_be2e_g160m640_r21": "raw timing repetitions were not persisted",
}

BANNED = re.compile(
    r"captured.*constant|out of memory|\boom\b|no space left|disk quota|"
    r"traceback|FAILED rc=", re.IGNORECASE)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checksum_audit(cell_dir):
    manifest = os.path.join(cell_dir, "REMOTE.sha256")
    if not os.path.isfile(manifest):
        return False, ["missing REMOTE.sha256"]
    errors = []
    with open(manifest) as fp:
        for line in fp:
            expected, rel = line.rstrip("\n").split(None, 1)
            rel = rel.lstrip("* ")
            path = os.path.join(cell_dir, rel)
            if not os.path.isfile(path):
                errors.append(f"missing {rel}")
            elif digest(path) != expected:
                errors.append(f"checksum mismatch {rel}")
    return not errors, errors


def launch_audit(cell_dir, log):
    """Close the chain from the staged batch script to its cluster check."""
    script = os.path.join(cell_dir, "run.sbatch")
    manifest = os.path.join(cell_dir, "MANIFEST.sha256")
    if not os.path.isfile(script) or not os.path.isfile(manifest):
        return False, False
    expected = None
    with open(manifest) as fp:
        for line in fp:
            if line.rstrip().endswith("  ./run.sbatch"):
                expected = line.split()[0]
                break
    script_ok = expected == digest(script) and "./run.sbatch: OK" in log
    with open(script) as fp:
        highest = "JAX_DEFAULT_MATMUL_PRECISION=highest" in fp.read()
    return script_ok, highest


def json_text(cell_dir):
    chunks = []
    for path in glob.glob(os.path.join(cell_dir, "out", "**", "*.json"), recursive=True):
        try:
            with open(path) as fp:
                chunks.append(json.dumps(json.load(fp), sort_keys=True))
        except (OSError, ValueError):
            pass
    return "\n".join(chunks)


def audit_cells():
    rows = []
    for cell_dir in sorted(glob.glob(os.path.join(RUNS, "nda_*"))):
        if not os.path.isdir(cell_dir):
            continue
        cell = os.path.basename(cell_dir)
        logs = []
        log_paths = glob.glob(os.path.join(cell_dir, "logs", "*"))
        for path in log_paths:
            if os.path.isfile(path):
                with open(path, errors="replace") as fp:
                    logs.append(fp.read())
        log = "\n".join(logs)
        job_ids = sorted({os.path.basename(path).split(".")[0] for path in log_paths
                          if os.path.basename(path).split(".")[0].isdigit()})
        meta = re.search(r"cell=\S+ commit=(\S+)(?: dirty_hash=\S+)? host=(\S+) gpu=([^\n]+)", log)
        js = json_text(cell_dir)
        checksums_ok, checksum_errors = checksum_audit(cell_dir)
        launch_ok, highest = launch_audit(cell_dir, log)
        excluded = cell in EXCLUDED
        gpu_ok = "jax_backend=gpu" in log
        done_ok = "ALL-DONE" in log
        failure_marker = bool(BANNED.search(log))
        precision_ok = (("x64=True" in log or "f64/highest" in log or
                         '"x64": true' in js) and highest)
        passed = (checksums_ok and launch_ok and gpu_ok and done_ok and precision_ok and
                  not failure_marker)
        rows.append(dict(
            cell=cell, excluded=excluded, exclusion_reason=EXCLUDED.get(cell),
            checksums_ok=checksums_ok, checksum_errors=checksum_errors,
            launch_provenance_ok=launch_ok,
            gpu_ok=gpu_ok, done_ok=done_ok, precision_ok=precision_ok,
            failure_marker=failure_marker, passed=passed,
            job_ids=job_ids, commit=None if meta is None else meta.group(1),
            host=None if meta is None else meta.group(2),
            gpu=None if meta is None else meta.group(3).strip(),
        ))
    return rows


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def audit_benchmark(cell):
    path = os.path.join(RUNS, cell, "out", "benchmark.json")
    if not os.path.isfile(path):
        return dict(cell=cell, present=False, passed=False, errors=["missing artifact"])
    with open(path) as fp:
        d = json.load(fp)
    errors = []
    for arm, kernels in d["timings"].items():
        for kernel, row in kernels.items():
            values = row.get("all_s")
            if not isinstance(values, list) or len(values) != d["reps"]:
                errors.append(f"{arm}/{kernel}: repetition count")
            elif not close(statistics.median(values), row["median_s"]):
                errors.append(f"{arm}/{kernel}: median mismatch")
    return dict(cell=cell, present=True, passed=not errors, errors=errors,
                reps=d["reps"], post_burn_warmups=d["warm"])


def audit_e2e(cell):
    errors = []
    files = [os.path.join(RUNS, cell, "out", f"{arm}.json")
             for arm in ("control", "variant")]
    if not all(os.path.isfile(path) for path in files):
        return dict(cell=cell, present=False, passed=False, errors=["missing artifact"])
    for path in files:
        with open(path) as fp:
            d = json.load(fp)
        reps = d["config"]["time_reps"]
        for index, row in enumerate(d["rows"]):
            raw = row.get("time_ms_e2e_repetitions_per_source")
            medians = row.get("time_ms_e2e_per_source")
            tag = f"{os.path.basename(path)} row {index}"
            if not isinstance(raw, list) or len(raw) != row["n_sources"]:
                errors.append(f"{tag}: missing per-source repetition arrays")
                continue
            if any(not isinstance(values, list) or len(values) != reps for values in raw):
                errors.append(f"{tag}: repetition count")
                continue
            recomputed = [statistics.median(values) for values in raw]
            if len(medians) != len(recomputed) or any(
                    not close(a, b) for a, b in zip(medians, recomputed)):
                errors.append(f"{tag}: per-source median mismatch")
            if not close(statistics.median(recomputed), row["time_ms"]):
                errors.append(f"{tag}: aggregate median mismatch")
    return dict(cell=cell, present=True, passed=not errors, errors=errors)


def main():
    cells = audit_cells()
    timing = [
        audit_benchmark("nda_pbench_g98b_r8"),
        audit_benchmark("nda_bbench_g160_r12"),
    ]
    e2e_cells = sorted({os.path.basename(path)
                        for pattern in ("nda_pe2e_*", "nda_be2e_*")
                        for path in glob.glob(os.path.join(RUNS, pattern))
                        if os.path.isdir(path) and
                        os.path.basename(path) not in EXCLUDED and
                        all(os.path.isfile(os.path.join(path, "out", f"{arm}.json"))
                            for arm in ("control", "variant"))})
    timing.extend(audit_e2e(cell) for cell in e2e_cells)
    overall = (all(row["passed"] for row in cells if not row["excluded"])
               and all(row["passed"] for row in timing))
    result = dict(overall_pass=overall, cells=cells, accepted_timing=timing,
                  excluded=EXCLUDED)
    with open(os.path.join(HERE, "audit.json"), "w") as fp:
        json.dump(result, fp, indent=2)

    md = [
        "# Nonlinear-decoder result audit", "",
        "Generated from pulled artifacts and checksum manifests. Excluded cells are retained for diagnosis but do not support reported timing claims.", "",
        f"Overall accepted-result audit: **{'PASS' if overall else 'PENDING/FAIL'}**", "",
        "## Cell integrity", "",
        "| cell | job | device | disposition | checksums | launch provenance | GPU backend | f64/highest | complete | failure marker |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in cells:
        disposition = (f"excluded: {row['exclusion_reason']}" if row["excluded"]
                       else ("accepted" if row["passed"] else "failed audit"))
        md.append(
            f"| {row['cell']} | {','.join(row['job_ids'])} | {row['gpu'] or '—'} | {disposition} | "
            f"{row['checksums_ok']} | {row['launch_provenance_ok']} | "
            f"{row['gpu_ok']} | {row['precision_ok']} | {row['done_ok']} | "
            f"{row['failure_marker']} |")
    md += ["", "## Accepted timing arrays", "",
           "| cell | present | repetition audit | details |",
           "|---|---:|---:|---|"]
    for row in timing:
        details = "; ".join(row["errors"]) or "raw arrays and medians agree"
        md.append(f"| {row['cell']} | {row['present']} | {row['passed']} | {details} |")
    md.append("")
    with open(os.path.join(HERE, "AUDIT.md"), "w") as fp:
        fp.write("\n".join(md))
    print(f"wrote audit.json and AUDIT.md: {'PASS' if overall else 'PENDING/FAIL'}")


if __name__ == "__main__":
    main()
