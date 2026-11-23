"""Verify the consolidated branch, code-only backup and owned cluster cleanup."""
from __future__ import annotations
import argparse
import json
import subprocess
from pathlib import Path

PREFIX = "experiments/jcp-ns-initializer"
REMOTE_REF = "refs/heads/codeonly/exp/2026-11-23-jcp-consolidated"


def audit(worktree, expected_mirror):
    worktree = Path(worktree).resolve()

    def git(*args, binary=False, input=None):
        return subprocess.check_output(["git", "-c", "gc.auto=0", "-C", str(worktree), *args],
                                       text=not binary, input=input)

    head = git("rev-parse", "HEAD").strip()
    assert git("branch", "--show-current").strip() == "exp/2026-11-23-jcp-consolidated"
    mirror = git("ls-remote", "origin", REMOTE_REF).split()[0]
    assert mirror == expected_mirror
    assert not git("status", "--porcelain", "--", PREFIX).strip()

    def tree(commit):
        result = {}
        for row in git("ls-tree", "-rlz", commit, "--", PREFIX, binary=True).split(b"\0"):
            if not row:
                continue
            metadata, path = row.split(b"\t", 1)
            mode, kind, blob, size = metadata.decode().split()
            assert kind == "blob"
            result[path.decode()] = {"mode": mode, "blob": blob, "bytes": int(size)}
        return result

    original, backup = tree(head), tree(mirror)
    eligible, excluded = [], []
    for path, record in original.items():
        if record["bytes"] <= 50 * 1024 * 1024:
            assert backup.get(path) == record, path
            eligible.append(path)
        else:
            excluded.append({"path": path, **record})
    assert [x["path"] for x in excluded] == [PREFIX + "/runs/f01/result.tar.gz"]
    raw_paths = []
    for tag in ("d01", "d02", "f01"):
        run = worktree / PREFIX / "runs" / tag
        for line in (run / "RESULTS.sha256").read_text().splitlines():
            _, relative = line.split(maxsplit=1)
            path = str((run / relative.lstrip("*")).relative_to(worktree))
            assert path in original and path in backup, path
            raw_paths.append(path)
    hashes = git("hash-object", "--stdin-paths", input="\n".join(raw_paths) + "\n").splitlines()
    assert len(hashes) == len(raw_paths)
    assert all(original[path]["blob"] == digest for path, digest in zip(raw_paths, hashes))
    lanes = []
    for slug, commit in [("phase-confirm", "8394b33cb3c2ce3912598ff80a64fd3b7d9b5e97"),
                         ("width-deploy", "d049dda2ab0e7b6e90510e3bbfe0971069617a08"),
                         ("cost-confirm", "040fb32c1e0ba76505c692974ce7bca22606ec31")]:
        git("merge-base", "--is-ancestor", commit, head)
        assert not git("diff", "--name-only", commit, head, "--", "experiments/jcp-" + slug).strip()
        lanes.append({"lane": slug, "original_commit": commit, "ancestor": True, "original_scientific_files_unchanged": True})
    remote_script = """import json,subprocess\nfrom pathlib import Path\nowned={'7477604','7478133','7478826'}\nrows=subprocess.check_output(['squeue','-h','-u','tawal01','-o','%i|%j|%T'],text=True).splitlines()\nqueue=[r for r in rows if r.split('|')[0] in owned]\np=Path('/cluster/tufts/paralab/tawal01/jcp-ns-initializer-20261123')\nprint(json.dumps({'owned_queued_jobs':queue,'owned_namespace_exists':p.exists(),'scope':'read-only independent cluster check'}))\n"""
    remote = json.loads(subprocess.check_output([
        "ssh", "tufts-login", "/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python", "-"
    ], input=remote_script, text=True))
    assert not remote["owned_queued_jobs"] and not remote["owned_namespace_exists"]
    return {"passed": True, "source_commit": head, "remote_ref": REMOTE_REF, "verified_remote_commit": mirror,
            "eligible_experiment_files_byte_identical": len(eligible),
            "all_raw_manifest_files_tracked_and_mirrored": len(raw_paths),
            "excluded_large_aggregate_archives": excluded,
            "large_archive_scope": "local aggregate retained and checksum-verified; every individual raw member is mirrored",
            "merge_preservation": lanes, "independent_cluster_cleanup": remote}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--expected-mirror", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.worktree, args.expected_mirror)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
