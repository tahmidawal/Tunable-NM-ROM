"""Verify a content-preserving consolidation without executing scientific models."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "gc.auto=0", *args], cwd=ROOT, text=True
    ).strip()


def records(ref: str, path: str) -> list[dict]:
    raw = git("ls-tree", "-rl", ref, "--", path)
    out = []
    for line in raw.splitlines():
        meta, filename = line.split("\t", 1)
        mode, kind, blob, size = meta.split()
        out.append({"path": filename, "mode": mode, "type": kind,
                    "git_blob": blob, "bytes": int(size)})
    return out


def main() -> None:
    registry = json.loads((HERE / "SOURCES.json").read_text())
    head = registry["consolidation_merge_head"]
    lanes = []
    expected_paths = set()
    for source in registry["source_branches"]:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", source["commit"], head],
            cwd=ROOT, check=True,
        )
        original = records(source["commit"], source["path"])
        merged = records(head, source["path"])
        assert original == merged, f"Changed source content: {source['path']}"
        expected_paths.update(row["path"] for row in original)
        lanes.append({**source, "ancestor": True, "byte_identical_git_trees": True,
                      "file_count": len(original),
                      "total_bytes": sum(row["bytes"] for row in original),
                      "file_records_sha256": hashlib.sha256(
                          json.dumps(original, sort_keys=True).encode()).hexdigest(),
                      "files": original})
    changed = set(git("diff", "--name-only", registry["canonical_base"], head).splitlines())
    allowed = expected_paths | {p for p in changed if p.startswith("experiments/jcp-consolidated-20261123/")}
    assert changed <= allowed, f"Unrelated paths merged: {sorted(changed-allowed)}"
    merges = git("log", "--first-parent", "--merges", "--format=%H %P %s",
                 f"{registry['canonical_base']}..{head}").splitlines()
    mirror_ref = f"refs/heads/codeonly/{registry['branch']}"
    remote_tip = git("ls-remote", "origin", mirror_ref).split()[0]
    mirror_head = registry["consolidation_mirror_head"]
    subprocess.run(["git", "merge-base", "--is-ancestor", mirror_head, remote_tip],
                   cwd=ROOT, check=True)
    remote_files = {row["path"]: row for lane in lanes
                    for row in records(mirror_head, lane["path"])}
    matching_remote_files = 0
    large_excluded = []
    cost_retention = json.loads(
        (ROOT / "experiments/jcp-cost-confirm/RETENTION.json").read_text()
    )
    for lane in lanes:
        for row in lane["files"]:
            remote_row = remote_files.get(row["path"])
            if remote_row is None:
                assert row["bytes"] > 50 * 1024 * 1024, row["path"]
                rel = row["path"].removeprefix("experiments/jcp-cost-confirm/")
                retained = cost_retention["files"][rel]
                assert retained["bytes"] == row["bytes"]
                large_excluded.append({**row, "sha256": retained["sha256"],
                                       "retention_source": "experiments/jcp-cost-confirm/RETENTION.json"})
            else:
                assert row == remote_row, f"Mirror mismatch: {row['path']}"
                matching_remote_files += 1
    output = {"passed": True, "verified_consolidated_head": head,
              "canonical_base": registry["canonical_base"],
              "lanes": lanes, "first_parent_merge_records": merges,
              "backup": {"remote_ref": mirror_ref, "verified_remote_head": mirror_head,
                         "matching_scientific_file_count": matching_remote_files,
                         "large_local_git_only_files": large_excluded},
              "changed_paths_only_from_named_lanes_and_consolidation_metadata": True,
              "scope": "Git ancestry and exact blob/mode identity; no numerical recomputation or change to prior qualification."}
    (HERE / "CONSOLIDATION.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"passed": True, "head": head,
                      "lanes": [{"path": x["path"], "files": x["file_count"]} for x in lanes]}))


if __name__ == "__main__":
    main()
