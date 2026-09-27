"""Fail-closed reader for the matched field-input pilot; never opens final cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_index(path, splits=("train", "validation"), require_nonempty=True):
    path = Path(path).resolve()
    index = json.loads(path.read_text())
    if index.get("pde") not in ("poisson", "burgers"):
        raise ValueError("index.pde must be poisson or burgers")
    if index.get("complete") is not True:
        raise ValueError("Dataset is not marked complete")
    # t2-burgers-test: "test" allowed -- this lane evaluates frozen checkpoints on the held-out test cases (DESIGN.md)
    if not set(splits) <= {"train", "validation", "calibration", "development", "test"}:
        raise ValueError("Final/test cohort access is disabled in this pilot reader")
    records = index.get("records", index.get("cases"))
    if not isinstance(records, list):
        raise ValueError("index must provide a records list")
    ids, identities, by_split = set(), set(), {s: [] for s in splits}
    selected_hashes = set()
    for record in records:
        # Read metadata only for unselected splits, never their field files.
        cid = record["case_id"]
        if cid in ids:
            raise ValueError(f"Duplicate case_id: {cid}")
        ids.add(cid)
        identity = (record.get("seed"), record.get("case_index", cid))
        if identity in identities:
            raise ValueError(f"Duplicate generation identity: {identity}")
        identities.add(identity)
        if record["split"] not in by_split:
            continue
        relative = Path(record["path"])
        full = (path.parent / relative).resolve()
        if relative.is_absolute() or not full.is_relative_to(path.parent):
            raise ValueError(f"Case path escapes dataset: {relative}")
        actual = sha256(full)
        if actual != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {cid}")
        if actual in selected_hashes:
            raise ValueError(f"Duplicate case archive: {cid}")
        selected_hashes.add(actual)
        item = dict(record, absolute_path=str(full))
        validate_case(item, index["pde"])
        by_split[record["split"]].append(item)
    if require_nonempty and any(not rows for rows in by_split.values()):
        raise ValueError(f"Requested splits must be nonempty: {list(by_split)}")
    # Content equality catches reused fields even if compressed NPZ bytes differ.
    field_ids = {}
    signatures = set()
    for split, rows in by_split.items():
        for row in rows:
            arrays = read_case(row)
            key = hashlib.sha256(arrays["input"].tobytes() + arrays["parameters"].tobytes()).hexdigest()
            if key in field_ids:
                raise ValueError(f"Duplicated physical input: {field_ids[key]} / {row['case_id']}")
            field_ids[key] = row["case_id"]
            signatures.add((tuple(arrays["input"].shape), tuple(arrays["target"].shape),
                            tuple(arrays["times"]), arrays["parameters"].size))
    if len(signatures) > 1:
        raise ValueError("Selected data must share mesh, channels and output times")
    return index, by_split


def read_case(record):
    with np.load(record["absolute_path"], allow_pickle=False) as data:
        if set(data.files) != {"input", "target", "parameters", "times"}:
            raise ValueError("Model-facing NPZ must contain exactly the four declared arrays")
        return {name: np.array(data[name], copy=True) for name in ("input", "target", "parameters", "times")}


def validate_case(record, pde):
    data = read_case(record)
    for name, value in data.items():
        if value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError(f"{record['case_id']} {name} must be finite float64")
    x, y, params, times = (data[k] for k in ("input", "target", "parameters", "times"))
    mesh = record["mesh"]
    if isinstance(mesh, dict):
        mesh = mesh.get("intervals")
    if not isinstance(mesh, int) or mesh < 4:
        raise ValueError("mesh must specify integer intervals")
    if x.shape != (1, mesh + 1, mesh + 1):
        raise ValueError(f"Unexpected input shape: {x.shape}, mesh={mesh}")
    expected_times = np.array([0.] if pde == "poisson" else [0., .05, .1, .15, .2, .25])
    if times.shape != expected_times.shape or not np.allclose(times, expected_times, rtol=0, atol=1e-14):
        raise ValueError(f"Unexpected output times: {times}")
    if y.shape != (times.size, 1, mesh + 1, mesh + 1):
        raise ValueError(f"Unexpected target shape: {y.shape}")
    if params.shape != ((0,) if pde == "poisson" else (1,)):
        raise ValueError("Only known PDE parameters belong in model inputs")
    if pde == "burgers" and not (params[0] > 0 and np.array_equal(x, y[0])):
        raise ValueError("Burgers requires positive viscosity and exact t=0 supplied field")
    for field in (x, y):
        for edge in (field[..., 0, :], field[..., -1, :], field[..., :, 0], field[..., :, -1]):
            if np.any(edge != 0):
                raise ValueError("Fields must satisfy exact homogeneous Dirichlet boundary")
    if np.linalg.norm(y[-1]) == 0:
        raise ValueError("Zero target is incompatible with declared relative errors")


def load_pair(train_path, validation_path):
    train_index, train = load_index(train_path, ("train",))
    validation_index, validation = load_index(validation_path, ("validation",))
    if train_index["pde"] != validation_index["pde"]:
        raise ValueError("Train and validation PDEs differ")
    records = train["train"] + validation["validation"]
    if len({r["case_id"] for r in records}) != len(records):
        raise ValueError("Train/validation case overlap")
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("Train/validation generation seed overlap")
    inputs, signatures = set(), set()
    for row in records:
        data = read_case(row)
        identity = hashlib.sha256(data["input"].tobytes() + data["parameters"].tobytes()).hexdigest()
        if identity in inputs:
            raise ValueError("Train/validation physical input overlap")
        inputs.add(identity)
        signatures.add((data["input"].shape, data["target"].shape, tuple(data["times"])))
    if len(signatures) != 1:
        raise ValueError("Train and validation field contracts differ")
    return train_index["pde"], train["train"], validation["validation"]


def audit(path, splits):
    index, rows = load_index(path, splits)
    return {"passed": True, "pde": index["pde"], "index_sha256": sha256(path),
            "counts": {key: len(value) for key, value in rows.items()},
            "case_ids": {key: [r["case_id"] for r in value] for key, value in rows.items()},
            "final_files_opened": False,
            "limitation": "Hashes/schema/disjointness checks do not certify reference solver accuracy."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.index, args.splits)
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text)
