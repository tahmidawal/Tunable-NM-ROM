"""Compare replay accuracy with original QF artifacts; never compare job timing."""
from pathlib import Path
import hashlib
import json
import sys


def main():
    run = Path(sys.argv[1])
    root = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
    oldroot = root / 'worktrees/2026-08-27-b1d-poissonqf/experiments/separable-decoder/runs/b1dqf'
    comparisons = []
    for n in (128, 256, 512):
        oldfile = oldroot / f'qf_n{n}/out/sep_poisson_qf_qf_N{n}.json'
        newfile = run / 'out' / f'poisson_n{n}.json'
        old, new = (json.loads(p.read_text()) for p in (oldfile, newfile))
        assert old['config']['ckpt_sha256'] == new['config']['ckpt_sha256']
        for r in new['rows']:
            if r['method'] not in ('full', 'qf'):
                continue
            previous = next(q for q in old['rows'] if all(
                q[k] == r[k] for k in ('method', 'tau', 'cohort')))
            differences = {key: abs(r[key]-previous[key]) for key in
                           ('err_rel_l2', 'err_rel_l2_max', 'censored_frac')}
            assert differences['err_rel_l2'] < 1e-8, differences
            assert differences['err_rel_l2_max'] < 1e-8, differences
            assert differences['censored_frac'] == 0
            comparisons.append(dict(N=n, method=r['method'], tau=r['tau'],
                cohort=r['cohort'], differences=differences,
                stop_reasons_match=r['stop_reasons'] == previous['stop_reasons'],
                historical_source=str(oldfile.relative_to(root)),
                historical_sha256=hashlib.sha256(oldfile.read_bytes()).hexdigest()))
    result = dict(compared_rows=len(comparisons), rows=comparisons,
                  max_mean_error_difference=max(r['differences']['err_rel_l2'] for r in comparisons),
                  max_worst_error_difference=max(r['differences']['err_rel_l2_max'] for r in comparisons),
                  note='Accuracy and stopping fidelity only; timing from different jobs is not compared. N1024 QF is an extension.',
                  passed=True)
    (run / 'historical-accuracy-reproduction.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'},indent=2))


if __name__ == '__main__':
    main()
