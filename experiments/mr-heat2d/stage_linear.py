"""Stage committed source/checkpoints by git-content hashes; optionally smoke config."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


FILES = ["experiments/separable-decoder/sep_common.py"]+["experiments/mr-heat2d/"+name for name in (
    "heat_core.py", "run_pilot.py", "runtime_paths.py", "verify_heat.py", "config-pilot.json", "config-transfer.json",
    "transfer_core.py", "linear_paths.py", "run_linear.py", "config-linear.json", "cluster/linear.sbatch", "LINEAR-DESIGN.md")]


def stage(destination, commit, smoke=False):
    commit = subprocess.check_output(["git", "rev-parse", commit]).decode().strip()
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {}; hashes = {}
    def content(name, origin_commit, git_path):
        blob = subprocess.check_output(["git", "show", origin_commit+":"+git_path])
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(blob)
        hashes[name] = hashlib.sha256(blob).hexdigest()
        provenance[name] = dict(commit=origin_commit, git_path=git_path)
    for name in FILES: content(name, commit, name)
    cfgpath = destination/"experiments/mr-heat2d/config-transfer.json"
    settings = json.loads(cfgpath.read_text())
    for spec in settings["checkpoints"]:
        content(spec["name"]+".pkl", settings["checkpoint_commit"], spec["path"])
        assert hashes[spec["name"]+".pkl"] == spec["sha256"]
    linear_path = destination/"experiments/mr-heat2d/config-linear.json"
    linear_settings = json.loads(linear_path.read_text())
    if smoke:
        linear_settings.update(requested_intervals=[64], reference_pair=[64, 128],
                               cohorts=[dict(name="smoke", seed=790711, count=1)], timing_repetitions=1)
        path = linear_path.with_name("config-linear-smoke.json")
        path.write_text(json.dumps(linear_settings, indent=2)+"\n")
        hashes[str(path.relative_to(destination))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = dict(source_commit=commit, checkpoint_commit=settings["checkpoint_commit"], sha256=hashes, provenance=provenance, smoke_fixture=smoke)
    (destination/"SOURCE-MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (destination/"SOURCE.sha256").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("destination", type=Path); p.add_argument("--commit", required=True); p.add_argument("--smoke", action="store_true")
    args = p.parse_args(); print(json.dumps(stage(args.destination, args.commit, args.smoke), indent=2))
