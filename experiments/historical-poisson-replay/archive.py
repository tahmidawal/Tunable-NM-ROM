"""Archive captured fields as bounded tracked parts and verify reconstruction.

Usage: absolute-python archive.py runs/replay01
Restore a field by concatenating its ordered parts listed in ARCHIVE.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def main():
    run = Path(sys.argv[1])
    destination = run / "archive"
    destination.mkdir(exist_ok=True)
    manifest = dict(format="ordered raw file parts; concatenate without decompression",
                    part_bytes=48 << 20, files=[])
    for field in sorted((run / "out").glob("fields_n*.npz")):
        expected = json.loads((run / "out" / field.name.replace("fields_", "poisson_").replace(".npz", ".json")).read_text())["fields_sha256"]
        parts = []
        original = hashlib.sha256()
        with field.open("rb") as stream:
            number = 0
            while block := stream.read(manifest["part_bytes"]):
                original.update(block)
                part = destination / f"{field.name}.part{number:03d}"
                digest = hashlib.sha256(block).hexdigest()
                if part.exists():
                    assert hashlib.sha256(part.read_bytes()).hexdigest() == digest
                else:
                    part.write_bytes(block)
                parts.append(dict(path=str(part.relative_to(run)), size=len(block), sha256=digest))
                number += 1
        assert original.hexdigest() == expected
        restored = hashlib.sha256()
        for part in parts:
            data = (run / part["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == part["sha256"]
            restored.update(data)
        assert restored.hexdigest() == expected
        manifest["files"].append(dict(path=str(field.relative_to(run)), size=field.stat().st_size,
                                      sha256=expected, parts=parts, reconstruction_verified=True))
        print(field.name, len(parts), "parts; reconstructed hash matches", flush=True)
    assert len(manifest["files"]) == 4
    (run / "ARCHIVE.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (run / "ARCHIVE.sha256").write_text("".join(
        f"{part['sha256']}  {part['path']}\n" for item in manifest["files"] for part in item["parts"]))


if __name__ == "__main__":
    main()
