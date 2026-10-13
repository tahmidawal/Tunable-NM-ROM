"""jcp-references gates G6 / G7: loader integrity fault injections and sealed refusals (DESIGN §7, A1.12, A3.8, A5.9).

    python test_refstore.py      -> results/test_refstore.json (all_pass)

G6: a loader pointed (via $JCPREF_STORE) at a copy of the store in which (a) two cases' files are swapped and (b) one
    field is perturbed by 1 % must raise ReferenceIntegrityError; the pristine copy must load; a STATUS written for
    another store version must be refused.
G7: every sealed / OOD cohort must raise SealedCohortError in load, reference_status and score; restricted metrics
    are never eligible; resolved() refuses mismatched or ineligible scores.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def run():
    import refstore as RS
    out = {}
    rel0, rel1 = '2d/val32/ST/c000', '2d/val32/ST/c001'
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for rel in (rel0, rel1):
            for suf in ('.npz', '.json'):
                (root / (rel + suf)).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(HERE / 'store' / (rel + suf), root / (rel + suf))
        os.environ['JCPREF_STORE'] = str(root)
        try:
            RS.load(2, 'val32', 0)
            out['pristine_loads'] = True
        except Exception as e:                          # noqa: BLE001
            out['pristine_loads'] = f'unexpected {e!r}'
        # (a) swapped case files
        for suf in ('.npz', '.json'):
            a, b = root / (rel0 + suf), root / (rel1 + suf)
            ta = a.read_bytes()
            a.write_bytes(b.read_bytes())
            b.write_bytes(ta)
        try:
            RS.load(2, 'val32', 0)
            out['swap_raises'] = False
        except RS.ReferenceIntegrityError:
            out['swap_raises'] = True
        for suf in ('.npz', '.json'):                   # restore
            shutil.copyfile(HERE / 'store' / (rel0 + suf), root / (rel0 + suf))
        # (b) one field perturbed by 1 %
        meta = json.loads((root / (rel0 + '.json')).read_text())
        arr = dict(np.load(root / (rel0 + '.npz')))
        k = meta['final'][2]
        arr[k] = arr[k] * 1.01
        buf = io.BytesIO()
        np.savez(buf, **arr)
        (root / (rel0 + '.npz')).write_bytes(buf.getvalue())
        try:
            RS.load(2, 'val32', 0)
            out['perturb_raises'] = False
        except RS.ReferenceIntegrityError:
            out['perturb_raises'] = True
        del os.environ['JCPREF_STORE']
    # stale STATUS refused (on an isolated copy of the metadata; the canonical files are never touched)
    with tempfile.TemporaryDirectory() as td:
        meta = Path(td)
        shutil.copyfile(HERE / 'store' / 'MANIFEST.sha256', meta / 'MANIFEST.sha256')
        st = json.loads((HERE / 'store' / 'STATUS.json').read_text())
        st['store_version'] = '0' * 64
        (meta / 'STATUS.json').write_text(json.dumps(st))
        os.environ['JCPREF_META'] = str(meta)
        try:
            RS.reference_status(2, 'val32', 0)
            out['stale_status_refused'] = False
        except RS.ReferenceIntegrityError:
            out['stale_status_refused'] = True
        finally:
            del os.environ['JCPREF_META']
    # G7 sealed refusals
    refusals = {}
    for d, coh in sorted(RS.SEALED):
        r = []
        for fn in (lambda: RS.load(d, coh, 0), lambda: RS.reference_status(d, coh, 0),
                   lambda: RS.score(np.zeros(1), d, coh, 0)):
            try:
                fn()
                r.append(False)
            except RS.SealedCohortError:
                r.append(True)
        refusals[f'{d}d/{coh}'] = all(r)
    out['sealed_refused'] = refusals
    rr = RS.restrict(2, 'val32', 0, 'ST', 2)
    r3 = RS.restrict(3, 'val64', 0, 'ST', 2)
    out['restricted_never_eligible'] = rr['eligible'] is False and r3['eligible'] is False
    f, _ = RS.load(2, 'val32', 0)
    sa = RS.score(f, 2, 'val32', 0)
    sb = dict(sa, case=1)
    try:
        RS.resolved(sa, sb)
        out['resolved_refuses_mismatch'] = False
    except ValueError:
        out['resolved_refuses_mismatch'] = True
    sc = dict(sa, eligible=False)
    try:
        RS.resolved(sa, sc)
        out['resolved_refuses_ineligible'] = False
    except ValueError:
        out['resolved_refuses_ineligible'] = True
    out['all_pass'] = bool(out['pristine_loads'] is True and out['swap_raises'] and out['perturb_raises']
                           and out['stale_status_refused'] and all(refusals.values())
                           and out['restricted_never_eligible'] and out['resolved_refuses_mismatch']
                           and out['resolved_refuses_ineligible'])
    return out


if __name__ == '__main__':
    r = run()
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'test_refstore.json').write_text(json.dumps(r, indent=1) + '\n')
    print(json.dumps(r, indent=1))
    if not r['all_pass']:
        raise SystemExit(1)
