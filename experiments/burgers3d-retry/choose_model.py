"""DESIGN §4 + R1-9, applied mechanically: choose the model among M0 (lane-1 bank), M1 (trw512b), M2 (trw1024b) by the
smallest max-over-meshes worst full-grid bank-validation floor at R' = 512 (cohort 923751 x 96), then the head.

    python choose_model.py --m1 <training.json of M1 or none> --m2 <training.json of M2 or none> --out checks/model-choice.json
M0's floors are taken from the `compare_bank_floors_full` record of the completed candidate jobs (same fields).
"""
import argparse
import json
from pathlib import Path

HEAD_BAR = 0.10
MESHES = ('33', '65', '129')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m1')
    ap.add_argument('--m2')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cands, notes, m0 = {}, {}, None
    for tag, p in (('M1', a.m1), ('M2', a.m2)):
        if not p or p == 'none' or not Path(p).exists():
            notes[tag] = 'excluded: job not completed by the cutoff'
            continue
        t = json.loads(Path(p).read_text())
        if not t.get('complete'):
            notes[tag] = 'excluded: training.json not complete'
            continue
        cond = t['bank'].get('condition_65')
        if cond is None or cond > 1e8 ** 0.5:            # recorded as sqrt(Gram condition); Gram threshold 1e8
            notes[tag] = f'excluded: 65-node condition {cond}'
            continue
        cands[tag] = dict(floors={n: t['floors_full'][n]['512']['worst'] for n in MESHES}, heads=t.get('heads', {}),
                          source=p)
        cb = {n: t['compare_bank_floors_full'][n]['512']['worst'] for n in MESHES}
        if m0 is not None:
            assert all(abs(m0[n] - cb[n]) <= 1e-9 * max(1., abs(cb[n])) for n in MESHES), ('M0 floors differ', m0, cb)
        m0 = cb
    assert m0 is not None, 'no completed candidate job: M0 floors unavailable on the new bank-validation cohort'
    cands['M0'] = dict(floors=m0, heads={}, source='compare_bank_floors_full')
    score = {k: max(v['floors'].values()) for k, v in cands.items()}
    chosen = min(score, key=score.get)
    head, head_err = None, None
    if chosen != 'M0' and cands[chosen]['heads']:
        hs = {k: v['validation_best_found_worst'] for k, v in cands[chosen]['heads'].items()}
        head = min(hs, key=hs.get)
        head_err = hs[head]
    out = dict(rule='min over candidates of max over meshes (33/65/129) of worst full-grid bank-validation floor at R=512',
               score=score, floors={k: v['floors'] for k, v in cands.items()}, excluded=notes, chosen=chosen,
               model_dir={'M0': 'experiments/burgers3d-span/inputs/model_R512', 'M1': 'inputs/model_M1',
                          'M2': 'inputs/model_M2'}[chosen],
               head_candidates={k: v['validation_best_found_worst'] for k, v in cands.get(chosen, {}).get('heads', {}).items()},
               head_best=head, head_best_error=head_err, head_bar=HEAD_BAR,
               head=head if head is not None and head_err <= HEAD_BAR else None)
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
