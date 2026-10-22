"""Historical (A100) cost records read from their source files (no hand-typed numbers); paths returned as provenance.
These files are git-tracked on the consolidated branch (worktree 2026-10-20-jcp-consolidated)."""
import json
import re
from pathlib import Path

SEP = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs')
R3A_JSON = SEP / 'push_r3a/out/sep_burgers_r3_N256_K16_R512.json'
R3A_LOG = SEP / 'push_r3a/logs/2835788.out'
DN = SEP / 'dn256b/out'


def records():
    fit = json.loads(R3A_JSON.read_text())['train']['seconds']
    gen = float(re.search(r'gen: trajectories 576/576 .*\[(\d+)s\]', R3A_LOG.read_text()).group(1))
    ext = json.loads((DN / 'sep_coeff_N256_K16_R512.json').read_text())['total_seconds']
    full = json.loads((DN / 'hfit_dn256b_full.json').read_text())
    arms = {k: v['seconds'] for k, v in full['arms'].items()}
    outside = full['total_seconds'] - sum(arms.values())
    other = [json.loads((DN / f'hfit_dn256b_{x}.json').read_text())['total_seconds'] for x in ('f576', 'f4096')]
    return dict(gen=gen, bankfit=fit, head=ext + arms['mid'] + outside, extract=ext, mid=arms['mid'], outside=outside,
                head_selection=arms['wide'] + arms['k32'] + sum(other),
                sources=dict(bankfit=f'{R3A_JSON}:train.seconds', gen=f'{R3A_LOG} (gen line)',
                             extract=str(DN / 'sep_coeff_N256_K16_R512.json'),
                             head=str(DN / 'hfit_dn256b_full.json'),
                             head_selection=str(DN / 'hfit_dn256b_{full,f576,f4096}.json')))


def a100_validation():
    log = R3A_LOG.read_text()
    st = lambda n: float(re.search(rf'step +{n}/300000 .*\[(\d+)s\]', log).group(1))
    done = float(re.search(r'done: recon.*\[(\d+)s\]', log).group(1))
    rs = (st(15000) - st(5000)) / 10000
    rf = (st(300000) - st(295000)) / 5000
    tau = (st(295000) - st(290000)) - 5000 * rf
    T = st(1) + 289999 * rs + tau + 10000 * rf + (done - st(300000))
    rec = records()['bankfit']
    return dict(rho_sub=rs, rho_full=rf, tau=tau, projected=T, recorded=rec, rel=T / rec - 1, source=str(R3A_LOG))


if __name__ == '__main__':                       # one-off extraction -> inputs/historical.json (input of the report)
    out = Path(__file__).resolve().parent / 'inputs' / 'historical.json'
    out.write_text(json.dumps(dict(records=records(), a100_validation=a100_validation()), indent=1))
    print(out)
