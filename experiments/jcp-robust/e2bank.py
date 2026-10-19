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


OUTPUT_ROOT = None          # smoke acceptance: where '../../output/...' (a job's own output dir) was collected locally


def use(bank, recorded=None):
    """recorded = (checkpoint sha256, rotation sha256) as recorded by the job (result.json); required for smoke banks whose
    config hashes are 'SMOKE-RUNTIME' (hashed in the job), whose files are then resolved under OUTPUT_ROOT."""
    if bank:
        res_ = lambda p_: (Path(OUTPUT_ROOT) / p_[len('../../output/'):]) if (OUTPUT_ROOT and p_.startswith('../../output/')) else HERE / p_
        ck, rot = res_(bank['ckpt']), res_(bank['rotation'])
        want = (bank['ckpt_sha256'], bank['rotation_sha256'])
        if want[0] == 'SMOKE-RUNTIME':
            assert OUTPUT_ROOT and recorded, 'runtime-hashed smoke bank needs the collected output and the recorded hashes'
            want = recorded
        assert _sha(ck) == want[0] and _sha(rot) == want[1], 'bank files differ from their hashes'
    else:
        ck, rot = _INC
    NC.CKPT, NC.ROT = ck, rot
    NC._P, NC._T = None, None
    NC._G257.clear()
    return ck, rot
