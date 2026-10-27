"""Bind retained local Git blobs to the complete, checksum-verified run evidence.

This reads files and the staging index; it does not run any scientific calculation.
Stage the five runs, cleanup receipt, and compact EXPORT copies before invoking it.
"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LANE = Path("experiments/burgers-recon2d")
RUNS = ("prep01", "inputplus01", "dev01", "dev02", "val01")


def digest_file(path, object_digest=None):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
            if object_digest is not None:
                object_digest.update(chunk)
    return sha.hexdigest()


def main():
    results = ROOT / LANE / "results"
    results.mkdir(exist_ok=True)
    targets = [LANE / "runs" / name for name in RUNS]
    targets += [LANE / "runs/REMOTE-CLEANUP.txt"]
    targets += [LANE / "results/dev02-EXPORT.json.zst", LANE / "results/val01-EXPORT.json.zst"]
    expected_files = set()
    for target in targets:
        absolute = ROOT / target
        if absolute.is_dir():
            expected_files.update(str(p.relative_to(ROOT)) for p in absolute.rglob("*") if p.is_file())
        else:
            assert absolute.is_file(), absolute
            expected_files.add(str(target))
    index = subprocess.check_output(
        ["git", "ls-files", "--stage", "-z", "--", *map(str, targets)], cwd=ROOT
    ).split(b"\0")
    records = []
    for entry in index:
        if not entry:
            continue
        header, name = entry.decode().split("\t", 1)
        mode, oid, stage = header.split()
        assert stage == "0" and mode in ("100644", "100755"), entry
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), path
        size = path.stat().st_size
        blob = hashlib.sha1(f"blob {size}\0".encode())
        sha = digest_file(path, blob)
        assert blob.hexdigest() == oid, (name, "index differs from expanded evidence")
        records.append({"path": name, "bytes": size, "sha256": sha, "git_blob": oid, "mode": mode})
    by_path = {r["path"]: r for r in records}
    assert len(by_path) == len(records)
    assert set(by_path) == expected_files, (expected_files - set(by_path), set(by_path) - expected_files)
    collections = []
    for name in RUNS:
        base = LANE / "runs" / name
        manifest = ROOT / base / "COLLECTION.sha256"
        members = 0
        for line in manifest.read_text().splitlines():
            sha, member = line.split(maxsplit=1)
            member = member.lstrip("*")
            path = str(base / member)
            assert by_path[path]["sha256"] == sha, path
            members += 1
        collections.append({"run": name, "collection_sha256": digest_file(manifest),
                            "collection_members_verified": members,
                            "tracked_files": sum(r["path"].startswith(str(base) + "/") for r in records),
                            "bytes": sum(r["bytes"] for r in records if r["path"].startswith(str(base) + "/"))})
    output = results / "RETENTION.members.jsonl"
    output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(records, key=lambda x: x["path"])))
    summary = {
        "status": "all_expanded_evidence_matches_staged_local_git_blobs",
        "storage": "Direct Git blobs; expanded evidence is retained. Numerical runs are excluded from the code-only mirror.",
        "member_manifest": str(output.relative_to(ROOT)),
        "member_manifest_sha256": digest_file(output),
        "tracked_files": len(records),
        "total_logical_bytes": sum(r["bytes"] for r in records),
        "unique_git_blobs": len({r["git_blob"] for r in records}),
        "collections": collections,
        "compact_exports": json.loads((results / "EXPORTS.json").read_text()),
        "training_retention": "Regenerable training matrices were omitted under the approved contract; hashes, seed, recipe, fingerprints and native POD bases are retained.",
        "invalid_attempt": "dev01 is retained for audit, fully ineligible for scientific or timing conclusions.",
        "durability_repair": "Earlier closing entry described ignored expanded copies; this manifest records the subsequent direct-Git retention repair without replacing those copies.",
    }
    (results / "RETENTION.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("tracked_files", "total_logical_bytes", "unique_git_blobs", "member_manifest_sha256")}))


if __name__ == "__main__":
    main()
