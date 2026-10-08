"""Unit test of jcp-wide-bank change (ii) in train2w.py (DESIGN A2-smoke): a fake 32-column comparison bank and the
ladder [16, 64] -> rank 64 must come back unavailable and floors computed only for 16. Runs the exact code block of
train2w.py extracted by text, so the test exercises the deployed lines."""
import re
from pathlib import Path

import numpy as np

src = Path(__file__).with_name('train2w.py').read_text()
blk = src[src.index('        # jcp-wide-bank change (ii)'):src.index("        log('COMPARE BANK FLOORS")]
calls = []


class _C:
    @staticmethod
    def clean(x):
        return x


def floors_full(p, T, n, f, ladder):
    calls.append(list(ladder))
    return {str(r): dict(worst=0.1, rms=0.1) for r in ladder}


ns = dict(np=np, C=_C, floors_full=floors_full, cb=dict(rotation=np.zeros((32, 32))), cp_=None, full_val={17: None},
          cfg=dict(ladder=[16, 64], meshes=[17]), rep={})
exec(compile(re.sub(r'^        ', '', blk, flags=re.M), 'blk', 'exec'), ns)
rep = ns['rep']
assert rep['compare_bank_columns'] == 32 and rep['compare_bank_unavailable_ranks'] == [64], rep
assert calls == [[16]] and list(rep['compare_bank_floors_full']['17']) == ['16'], (calls, rep)
print('filter unit test: PASS', rep)
