"""Per-subject timing-drift diagnosis of the closeout's archived timing records (timing fields only).

Run on the cluster: python3 - < closeout_drift.py > closeout-drift.json
Reads only result.json['timing'], ['gpu'], ['host'], ['job_id'] of the eight archived closeout panels.
No case descriptors, fields or scores are read or emitted.
"""
import json
import statistics as S

A = '/cluster/tufts/paralab/tawal01/burgers-archive/jcp-sealed-closeout-20261115/'
PAIR = {2: ('rom_lat|fib987', 'fom|PS_N48'), 3: ('rom_lat|lat4096', None)}
SEL3 = {65: 'fom|PS_N16', 129: 'fom|PS_N20', 257: 'fom|PS_N20'}
out = {}
for t in ['sealed2d256', 'sealed2d512', 'sealed2d1024', 'sealed2d2048', 'sealed2d4096',
          'sealed3d65', 'sealed3d129', 'sealed3d257']:
    r = json.load(open(A + t + '/output/result.json'))
    tm = r['timing']
    d = 2 if '2d' in t else 3
    mesh = int(t.split('d')[-1])
    pair = [PAIR[d][0], PAIR[d][1] if d == 2 else SEL3[mesh]]
    subj = {}
    for pk, ik in (('summary', 'invocations'), ('confirm_summary', 'confirm_invocations')):
        for k, v in tm[pk].items():
            kind, name = k.split('|')
            rows = [x for x in tm[ik] if x['kind'] == kind and x['name'] == name]
            reps = sorted({x['rep'] for x in rows})
            bm = [1e3 * S.median([x['seconds'] for x in rows if x['rep'] == q]) for q in reps]
            subj.setdefault(k, {})[pk] = dict(block_drift=v['block_drift'], median_ms=v['median_ms'], block_medians_ms=bm)
    failing = sorted({k for k, v in subj.items() for b in v.values() if b['block_drift'] > 1.10})
    pair_rows = {k: subj[k] for k in pair}
    pair_fold = {k: max(v['summary']['median_ms'] / v['confirm_summary']['median_ms'],
                        v['confirm_summary']['median_ms'] / v['summary']['median_ms']) for k, v in pair_rows.items()}
    out[t] = dict(job_id=r['job_id'], host=r['host'], gpu=r['gpu'], timing_burn=tm['burn'], initial_burn=tm['initial_burn'],
                  n_subjects_main=len(tm['summary']), n_subjects_confirm=len(tm['confirm_summary']),
                  panel_worst_block_drift=max(b['block_drift'] for v in subj.values() for b in v.values()),
                  subjects_failing_drift=failing,
                  failing_block_index={k: [i for pk, b in subj[k].items() for i, x in enumerate(b['block_medians_ms'])
                                           if b['block_drift'] > 1.10 and (x / b['median_ms'] > 1.10 or b['median_ms'] / x > 1.10)]
                                       for k in failing},
                  selected_pair=pair,
                  selected_pair_worst_block_drift=max(b['block_drift'] for v in pair_rows.values() for b in v.values()),
                  selected_pair_main_confirm_fold=pair_fold,
                  subjects=subj)
print(json.dumps(out, indent=1))
