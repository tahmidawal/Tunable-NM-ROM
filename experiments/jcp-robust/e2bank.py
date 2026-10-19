"""jcp-robust: point the independent NumPy decode (npcheck) at a retrained 2D bank (DESIGN 3.2). CPU only.

    e2bank.use(None)                     incumbent (dn256b checkpoint + vendored rotation_R512.npz)
    e2bank.use(cfg['bank'])              {ckpt, ckpt_sha256, rotation, rotation_sha256} relative to the lane
The files' sha256 are verified; npcheck's caches are reset so no table of another bank survives.
"""
import hashlib
from pathlib import Path

import npcheck as NC

HERE = Path(__file__).resolve().parent
_INC = (NC.CKPT, NC.ROT)


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def use(bank):
    if bank:
        ck, rot = HERE / bank['ckpt'], HERE / bank['rotation']
        assert _sha(ck) == bank['ckpt_sha256'] and _sha(rot) == bank['rotation_sha256'], 'bank files differ from their hashes'
    else:
        ck, rot = _INC
    NC.CKPT, NC.ROT = ck, rot
    NC._P, NC._T = None, None
    NC._G257.clear()
    return ck, rot
