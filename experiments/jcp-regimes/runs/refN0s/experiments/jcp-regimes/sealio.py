"""jcp-references: cooperative (unenforced) sealing helpers (DESIGN A1.11, A2.9).

AES-256-CBC via the system `openssl enc -aes-256-cbc -pbkdf2 -md sha256 -salt -pass file:<key>`, piped from memory
(no plaintext file is ever written). The key file holds one 64-character hex line. CBC is unauthenticated, so callers
record sha256 of every plaintext (inside the encrypted manifest) and of every ciphertext (in git), and verify both.
On a shared cluster account this protects against accidental reading only.
"""
from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

CMD = ['openssl', 'enc', '-aes-256-cbc', '-pbkdf2', '-md', 'sha256', '-salt']


def _check_key(key):
    s = Path(key).read_text().strip()
    assert len(s) == 64 and all(c in '0123456789abcdef' for c in s), 'key must be one 64-char lowercase hex line'


def encrypt(data: bytes, key) -> bytes:
    _check_key(key)
    return subprocess.run(CMD + ['-pass', f'file:{key}'], input=data, capture_output=True, check=True).stdout


def decrypt(blob: bytes, key) -> bytes:
    _check_key(key)
    return subprocess.run(CMD + ['-d', '-pass', f'file:{key}'], input=blob, capture_output=True, check=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def write_sealed(path, data: bytes, key):
    """Encrypt and write atomically, read-only; returns (plaintext sha256, ciphertext sha256)."""
    import os
    blob = encrypt(data, key)
    assert decrypt(blob, key) == data
    tmp = Path(str(path) + '.tmp')
    tmp.write_bytes(blob)
    os.replace(tmp, path)
    os.chmod(path, 0o444)
    return sha(data), sha(blob)


def read_seeds(path, key, expect_plain_sha, expect_cipher_sha):
    """Decrypt the secret seed file, verifying the committed ciphertext hash BEFORE and the plaintext hash AFTER."""
    blob = Path(path).read_bytes()
    if sha(blob) != expect_cipher_sha:
        raise ValueError('seed file ciphertext hash mismatch')
    data = decrypt(blob, key)
    if sha(data) != expect_plain_sha:
        raise ValueError('seed file plaintext hash mismatch (wrong key?)')
    return json.loads(data)
