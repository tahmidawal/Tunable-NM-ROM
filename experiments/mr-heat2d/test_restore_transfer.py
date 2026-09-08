"""CPU-only fixtures for chunk order, missing-file restoration and corruption."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from restore_transfer import restore


class ArchiveTests(unittest.TestCase):
    def test_roundtrip_and_refuse_modified_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"scientific bytes\x00"*4096
            files = {"outputs/field.bin": payload, "ARCHIVE.sha256": (hashlib.sha256(payload).hexdigest()+"  outputs/field.bin\n").encode()}
            stream = io.BytesIO()
            with tarfile.open(fileobj=stream, mode="w:gz") as archive:
                for name, data in files.items():
                    member = tarfile.TarInfo(name); member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
            blob = stream.getvalue(); parts = []
            for i, start in enumerate(range(0, len(blob), 59)):
                chunk = blob[start:start+59]; path = root/f"part{i:03}"
                path.write_bytes(chunk)
                parts.append(dict(path=path.name, bytes=len(chunk), sha256=hashlib.sha256(chunk).hexdigest()))
            (root/"ARCHIVE.json").write_text(json.dumps(dict(parts=parts, joined_sha256=hashlib.sha256(blob).hexdigest())))
            self.assertTrue(restore(root)["archive_verified"])
            self.assertEqual((root/"archive/outputs/field.bin").read_bytes(), payload)
            self.assertTrue(restore(root)["archive_verified"])
            (root/"archive/outputs/field.bin").unlink()
            self.assertTrue(restore(root)["archive_verified"])
            (root/"archive/outputs/field.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "refusing overwrite"): restore(root)
            (root/parts[0]["path"]).write_bytes(b"corrupt chunk")
            with self.assertRaises(AssertionError): restore(root)


if __name__ == "__main__": unittest.main()
