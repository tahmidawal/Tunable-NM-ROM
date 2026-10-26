"""Direct CPU re-decode of saved timing-job coefficients against approved E0 ST fields.

This is a four-case historical timing workload, NOT a full-validation frontier. The timing
driver linked saved coefficients/internal records to every timed payload; retain that gate
and every timing repetition. No cross-job cost/error assembly is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from inventory import ROOT, EXPECTED_STORE, EXPECTED_STATUS, BANK_HASH, identity, read, sha, write

HERE = Path(__file__).resolve().parent
DIMS = ROOT / 'worktrees/2026-10-18-jcp-dims/experiments/jcp-dims'


def forward(params, rotation, x):
    ang = 2 * np.pi * (x @ params['freq'])
    h = np.concatenate([np.sin(ang), np.cos(ang)], axis=1)
    for w, b in params['net'][:-1]:
        a = h @ np.asarray(w) + np.asarray(b)
        h = a * (.5 * (1 + np.tanh(.5 * a)))
    w, b = params['net'][-1]
    y = h @ np.asarray(w) + np.asarray(b)
    mu = 64 * np.prod(x * (1 - x), axis=1)
    return (float(params['scale']) * mu)[:, None] * y @ rotation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=HERE / 'data/timing-rescore.json')
    args = ap.parse_args()
    run = DIMS / 'runs/t3s/slim/output/d3run_t3s'
    rp, wp = run / 'result.json', run / 'W_t3s.npz'
    result = read(rp)
    panel = result['timing']['screen']
    assert panel['valid'] and result['complete'] and result['backend'] == 'gpu'
    analysis = read(DIMS / 'checks/analysis-3d.json')
    keys = list(dict.fromkeys(analysis['front']['median'] + analysis['front']['worst']))
    keys += [panel['A']] if panel['A'] not in keys else []
    refdir = ROOT / 'worktrees/2026-10-06-jcp-references/experiments/jcp-references'
    sys.path.insert(0, str(refdir))
    import refstore as rs
    assert rs.store_version() == EXPECTED_STORE and sha(refdir / 'store/STATUS.json') == EXPECTED_STATUS
    # Exact population comes from the archived panel, not a guessed cohort prefix.
    ids = sorted(set(int(v['case']) for v in panel['invocations']))
    assert ids == sorted(int(v) for v in panel['cases']), ids
    fields, statuses = {}, {}
    for cid in ids:
        fields[cid], _ = rs.load(3, 'val64', cid, 'ST', 'finest')
        statuses[cid] = rs.reference_status(3, 'val64', cid, 'ST')
        assert statuses[cid]['eligible'] and fields[cid].shape == (6, 63 ** 3)
    records, checks, sources = {}, {}, {str(rp): sha(rp), str(wp): sha(wp)}
    with np.load(wp, allow_pickle=False) as z:
        for bank in ['M2', 'W1024']:
            selected = [k for k in keys if (k.startswith('M2') if bank == 'M2' else k.startswith('W'))]
            if not selected:
                continue
            bp = DIMS / ('vendor/quad3d/inputs/model_M2/bank.pkl' if bank == 'M2' else 'inputs/model_W1024/bank.pkl')
            assert sha(bp) == BANK_HASH[bank]
            sources[str(bp)] = sha(bp)
            b = pickle.loads(bp.read_bytes())
            params = b['params']
            rotation = np.asarray(b['rotation'])
            maximum = max(identity(k)['r'] for k in selected)
            rotation = rotation[:, :maximum]
            weights, subjects = [], []
            for key in selected:
                assert key in z.files
                case_ids = [int(v) for v in z['cases|' + key]]
                assert case_ids == ids, (key, case_ids, ids)
                W = z[key]
                assert W.shape == (len(ids), 6, identity(key)['r'])
                for ci, cid in enumerate(ids):
                    ev = result['link_evidence'][f'{key}#{cid}']
                    assert ev['identity'] and ev['valid'] and ev['fields_finite']
                    subjects.append((key, cid))
                    weights.append(np.pad(W[ci], ((0, 0), (0, maximum - W.shape[-1]))))
            coeffs = np.concatenate(weights, axis=0)
            errors2 = np.zeros((len(subjects), 6))
            for start in range(0, 63 ** 3, 2048):
                idx = np.arange(start, min(start + 2048, 63 ** 3))
                x = (np.stack(np.unravel_index(idx, (63, 63, 63)), axis=1) + 1) / 64
                G = forward(params, rotation, x)
                if start == 0:
                    # Independent existing NumPy bank evaluator, including architecture/rotation scaling.
                    vendor = DIMS / 'vendor/quad3d'
                    sys.path.insert(0, str(vendor))
                    import npcore
                    p0, T0 = npcore.load_bank(bp.parent)
                    chk = npcore.bank_np(p0, T0[:, :maximum], x[:7])
                    diff = float(np.max(np.abs(chk - G[:7])))
                    assert diff <= 1e-11
                    checks[bank] = dict(independent_bank_max_abs=diff)
                prediction = (coeffs @ G.T).reshape(len(subjects), 6, len(idx))
                for i, (_, cid) in enumerate(subjects):
                    error = prediction[i] - fields[cid][:, start:start + len(idx)]
                    errors2[i] += np.sum(error ** 2, axis=1)
            for i, (key, cid) in enumerate(subjects):
                e_t = np.sqrt(errors2[i]) / np.linalg.norm(fields[cid][0])
                records.setdefault(key, dict(case_ids=ids, per_case={}, timing_repetitions=[],
                    historical_full64_qualification=key in analysis['qualified_3d']))
                records[key]['per_case'][str(cid)] = dict(e_t=e_t.tolist(), e=float(e_t[1:].max()),
                    eps_ref=statuses[cid]['eps_ref'], eligible=statuses[cid]['eligible'])
            print(f'rescored {bank}: {len(selected)} subjects, {len(ids)} cases', flush=True)
    for key, rec in records.items():
        if key == panel['A']:
            rec['timing_repetitions'] = [dict(case=v['case'], rep=v['rep'], position=pos, seconds=v[t],
                 paired_subject=v['B'], hashes_ok=v['hashes_ok'])
                 for v in panel['invocations'] for pos, t in [('before', 'tA1'), ('after', 'tA2')]]
        else:
            rec['timing_repetitions'] = [dict(case=v['case'], rep=v['rep'], seconds=v['tB'],
                 ratio=v['ratio'], hashes_ok=v['hashes_ok']) for v in panel['invocations'] if v['B'] == key]
        assert rec['timing_repetitions'] and all(v['hashes_ok'] for v in rec['timing_repetitions'])
        assert sorted({v['case'] for v in rec['timing_repetitions']}) == ids
        rec['query_median_seconds'] = float(np.median([np.median([v['seconds'] for v in rec['timing_repetitions']
                                            if v['case'] == cid]) for cid in ids]))
        es = [v['e'] for v in rec['per_case'].values()]
        rec.update(median=float(np.median(es)), worst=max(es), errors_above_1pct=sum(v > .01 for v in es))
    out = dict(scope='Historical t3s timing population only; not a full-validation frontier',
        job_id=result['job_id'], gpu=result['gpu'], source_commit=result['commit'],
        reference_store=EXPECTED_STORE, reference_status=EXPECTED_STATUS,
        reference='E0 ST finest, conditional Richardson uncertainty', case_ids=ids,
        output_contract='mesh input, six full fields on 65-node mesh interior',
        records=records, controls=checks, sources=sources,
        limitations=['The timing population omits difficult validation cases outside its exact ids.',
                     'Full-validation accuracy/qualification is separate historical evidence.',
                     'Untimed saved coefficients are linked to timed payloads by original same-job identity gates.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, out)
    print(args.out)


if __name__ == '__main__':
    main()
