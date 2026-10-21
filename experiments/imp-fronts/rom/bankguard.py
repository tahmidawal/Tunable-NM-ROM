"""imp-fronts C-B1 bank identity guard (DESIGN B2, Amendment 1 B2: a deliberate checkpoint / rotation / hash mismatch must
be rejected, deterministically). Pure hashlib + json; no JAX.

A bank is named in a job config as {"dir": "inputs/banks/s<label>"} (resolved relative to rom/) or with explicit
{"ckpt", "ckpt_sha256", "rotation", "rotation_sha256"} (r2acc's form). verify() requires ALL of:
  * the directory's BANK.json exists and names ckpt_sha256 / rotation_sha256;
  * the files hash to BANK.json's values;
  * the config's own hashes (when given) equal BANK.json's;
  * an optional expected label / seed equals BANK.json's.
Any disagreement raises BankMismatch (never a warning).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


class BankMismatch(ValueError):
    pass


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def resolve(bank, root=HERE):
    """Return dict(dir, ckpt, rotation, meta_path) for a bank config (paths absolute)."""
    root = Path(root)
    if 'dir' in bank:
        d = root / bank['dir']
        meta = json.loads((d / 'BANK.json').read_text()) if (d / 'BANK.json').exists() else {}
        names = [meta.get('ckpt', 'ckpt.pkl'), meta.get('rotation', 'rotation_R512.npz')]   # register_arm's canonical keys
        if any(Path(n).is_absolute() or '..' in Path(n).parts or len(Path(n).parts) != 1 for n in names):
            raise BankMismatch(f'{d}/BANK.json: ckpt / rotation must be plain file names')
        ck, rot = d / names[0], d / names[1]
    else:
        ck, rot = root / bank['ckpt'], root / bank['rotation']
        d = ck.parent
        if rot.parent != d:
            raise BankMismatch(f'checkpoint and rotation live in different directories: {ck.parent} vs {rot.parent}')
    return dict(dir=d, ckpt=ck, rotation=rot, meta_path=d / 'BANK.json')


def verify(bank, root=HERE, expect_label=None):
    """C-B1 hash guard. Returns the r2acc-style bank dict {ckpt, ckpt_sha256, rotation, rotation_sha256, ...} with
    paths relative to `root`; raises BankMismatch on any disagreement."""
    r = resolve(bank, root)
    if not r['meta_path'].exists():
        raise BankMismatch(f'no BANK.json in {r["dir"]}')
    meta = json.loads(r['meta_path'].read_text())
    for k in ('ckpt_sha256', 'rotation_sha256'):
        if not isinstance(meta.get(k), str) or len(meta[k]) != 64:
            raise BankMismatch(f'BANK.json lacks a valid {k}')
    got = dict(ckpt_sha256=sha_file(r['ckpt']), rotation_sha256=sha_file(r['rotation']))
    for k in ('ckpt_sha256', 'rotation_sha256'):
        if got[k] != meta[k]:
            raise BankMismatch(f'{k}: file {got[k][:12]} != BANK.json {meta[k][:12]} ({r["dir"]})')
        if bank.get(k) not in (None, meta[k]):
            raise BankMismatch(f'{k}: config {str(bank[k])[:12]} != BANK.json {meta[k][:12]}')
    if expect_label is not None and str(meta.get('label', meta.get('seed'))) != str(expect_label):
        raise BankMismatch(f'bank label {meta.get("label", meta.get("seed"))} != expected {expect_label}')
    rel = lambda p: str(Path(p).relative_to(Path(root))) if str(p).startswith(str(Path(root))) else str(p)
    return dict(ckpt=rel(r['ckpt']), ckpt_sha256=got['ckpt_sha256'], rotation=rel(r['rotation']),
                rotation_sha256=got['rotation_sha256'], bank_json=rel(r['meta_path']),
                bank_json_sha256=sha_file(r['meta_path']), label=meta.get('label', meta.get('seed')))


def must_reject(bank, root=HERE, **kw):
    """Must-fail control: returns dict(fired=True, message) iff verify() raised BankMismatch."""
    try:
        verify(bank, root, **kw)
        return dict(fired=False, raised=False)
    except BankMismatch as e:
        return dict(fired=True, raised=True, message=str(e))
