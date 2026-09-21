"""Write the frozen 4096^2 evaluation configs (bh3: dev6 + hold64; bh4: fresh64) from bh2's audited selection.

    python make_eval_configs.py checks/bh2b-summary.json [selection-config]

The headline arm is read from the audited bh2 summary (group 'dev6+sel32', key 'selection.chosen'); it is
never chosen here. The arm set, FOM grid, repetitions and audit coverage are identical in both configs.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    summ = json.loads(Path(sys.argv[1]).read_text())
    sel = summ['groups']['dev6+sel32']['selection']
    head = sel['chosen']
    assert head, sel
    base = json.loads((HERE / sys.argv[2]).read_text()) if len(sys.argv) > 2 else json.loads((HERE / 'config-1024-bh2b.json').read_text())
    rungs = []
    for r in base['rungs']:                       # the selection job's full ladder, minus its dense twins
        r = dict(r, dense=False)
        r.pop('dense_cases', None)
        rungs.append(r)
    names = [f"q{r['q']}_M{r['M']}_{r['rules'][0]['name']}" for r in rungs]
    assert any(head.startswith(n + '_') for n in names), (head, names)
    fast = 'q0_M64_lat64_g0p01_fast_clip_lamcarry_pred2'
    common = dict(base)
    for k in ('selection', 'eval_draws', 'cohort_name', 'attempt', 'purpose'):
        common.pop(k, None)
    common.update(
        intervals=4096, target_chunk=4, restrict_to=256, rungs=rungs,
        fom_settings=base['fom_settings'],
        coarse_fom=[{"name": "c2048_nt1e-4_dt005", "mesh": 2048, "dt": 0.005, "ntol": 0.0001, "ltol": 1e-06},
                    {"name": "c1024_nt1e-4_dt005", "mesh": 1024, "dt": 0.005, "ntol": 0.0001, "ltol": 1e-06}],
        untimed=["fft_tight"], audit_cases=[0], audit_arms=["fft_tight", head, fast],
        reps=5, required_reps=5, burn_seconds=0.25, profile_arms=[head], reference=None,
        keep_for_reference=[], prune_each_arm=True, headline_arm=head, fast_arm=fast,
        headline_source=dict(bh2_summary=Path(sys.argv[1]).name, bh2_result_sha256=summ['result_sha256'], rule=sel['rule']))
    c3 = dict(common, attempt='bh3', eval_draws=[[7090702, 4, 'dev6'], [911702, 2, 'dev6'], [20260916, 64, 'hold64']],
              cohort_name='dev6 (opened development) + hold64 = params_draw(20260916,64) (the brief target; opened before this lane, see DESIGN A-1)',
              purpose='bh3: frozen model and frozen headline arm at 4096^2 on dev6 and hold64, paired FOM grid, 5 repetitions')
    c4 = dict(common, attempt='bh4', eval_draws=[[20260929, 64, 'fresh64']],
              cohort_name='fresh64 = params_draw(20260929,64), untouched confirmation cohort (DESIGN A-1)',
              purpose='bh4: the same frozen model and arms on the untouched fresh64 cohort, 5 repetitions')
    for name, c in (('config-4096-bh3.json', c3), ('config-4096-bh4.json', c4)):
        (HERE / name).write_text(json.dumps(c, indent=1) + '\n')
        print('wrote', name, 'headline', head, 'rungs', [(r['q'], r['M']) for r in rungs])


if __name__ == '__main__':
    main()
