"""Write the frozen runs/FRONT-2D.json for jcp-fom2d (DESIGN 6, A1.6) from checks/analysis.json and the job archives.
Run only after the 2D results audit has passed; the file is then frozen (runs/FRONT-2D.sha256); a later change is a new
versioned file (FRONT-2D-v2.json) with a dated amendment.

    python front2d.py [--version 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import analyze as AN

HERE = Path(__file__).resolve().parent


def manifest(mesh, key, an):
    """Full configuration manifest of one configuration (A1.6): rebuildable with p1common.Setting + qacc.run."""
    parts = key.split('|')
    _, r, form, sc, f = parts[:5]
    fb = len(parts) > 5 and parts[5] == 'fb'
    r, f = int(r[1:]), float(f)
    cfg = an['configurations'][key]
    dt = .005 * f
    meta, trust, src = None, None, None
    pat = f'fb{mesh}*' if fb else f'acc{mesh}*'
    for att in sorted(p_.name for p_ in AN.RUNS.glob(pat)):                # the attempt that ran this configuration
        acc = AN.out_dir(att, 'qacc')
        if acc is None:
            continue
        res = json.loads((acc / 'result.json').read_text())
        st = res['settings'].get(f'R{r}' + ('fb' if fb else ''))
        if st:
            meta, trust, src = st['meta']['main'], st['trust'], att
    dev = json.loads((HERE / 'runs' / 'MANIFEST-dev.json').read_text())
    return dict(
        mesh=mesh, Rp=r, M=4 * r, mode_list=dict(policy='frozen F_2048 prefix', prefix_M=4 * r, path=dev['modes_F2048']['path'],
                                                 sha256=dev['modes_F2048']['sha256'],
                                                 hash_M=dev['mode_lists'][f'1024|frozen|{4 * r}']['hash'] if f'1024|frozen|{4 * r}' in dev['mode_lists'] else None),
        rule=cfg['rule'], rule_meta=meta, rule_generator_sha256=dev['code']['rule_generator_sha256'],
        form=form, scheme=sc, dt=dt, factor=f, level='prod', tolerances=dict(LSPG=dev['code']['tol_of_prod_LSPG'],
                                                                               GAL=dev['code']['tol_of_prod_GAL'])[form],
        budget=600, trust=trust, init_contract=dev['code']['init_contract'],
        weighting='LSPG D = 1/(a0 + dt b0 nu lam)' if form == 'LSPG' else 'GAL Q^T R = 0, A = Q Rm',
        output_times=dev['code']['output_times'], metric_id=dev['e0_metric_id'], e0_store_version=dev['e0_store_version'],
        checkpoint_sha256=dev['checkpoint_sha256'], rotation_sha256=dev['rotation_sha256'], bank_edges=[0, 128, 256, 384, 512],
        code=dict(p1core_sha256=dev['code']['p1core_sha256'], p1common_sha256=dev['code']['p1common_sha256'],
                  qacc_sha256=dev['code']['qacc_sha256']),
        entry_point=f"P.Setting(P.Bank({mesh}, [0,128,256,384,512]), 'R{r}', {r}, '{cfg['rule']}', 'frozen', forms=('LSPG','GAL')); "
                    f"qacc.run(bank, st, 'main', '{form}', '{sc}', {dt!r}, 'prod', u0, nu)",
        source=cfg['source'], accuracy_attempt=src, fallback=fb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', type=int, default=1)
    a = ap.parse_args()
    an = json.loads((HERE / 'checks' / 'analysis.json').read_text())
    out = dict(version=a.version, lane='jcp-qualify', commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(),
               label='E0 R_ST errors are conditional Richardson estimates (E0 A10; combined A10, A11, retracted A15 carried)',
               contract='full-field 2D query (input sampled at quadrature points; full-field decode); cost = median fused-query '
                        'time / mean flanking reference A (r=384, 1024^2, LSPG-CN-R, dt0, gauss128), one H200 job per mesh',
               dominance='DESIGN 6: c_X <= 1.05 c_Y, e_X <= e_Y + tau, (c_X < c_Y/1.05 or e_X < e_Y - tau), '
                         'tau = max(2 max(eps_X, eps_Y), 0.02 max(e_X, e_Y))', meshes={})
    for mesh, F in an['fronts'].items():
        if 'candidates' not in F:
            out['meshes'][mesh] = dict(status=F.get('status'))
            continue
        T = an['timing'][str(mesh)] if str(mesh) in an['timing'] else an['timing'][mesh]
        sub_by_key = {AN.skey(s['spec']): s for s in T['subjects'].values()}
        def entry(k):
            c = an['configurations'][k]
            s = sub_by_key.get(k, {})
            return dict(config=k, manifest=manifest(int(mesh), k, an),
                        labels=dict(numerically_qualified_accuracy_job=c['numerically_qualified'], Q1=c['Q1'], Q2=c['Q2'],
                                    numerically_qualified_timing_job=s.get('timing_label', {}).get('numerically_qualified'),
                                    role=an['roles'].get(k, {}).get('role'),
                                    fronts=[x for x in ('median|c', 'worst|c') if k in F['fronts'][x]]),
                        errors=dict(median=c['stats']['median'], worst=c['stats']['worst'], n=c['stats']['n'],
                                    per_case=c['per_case']),
                        timing=dict(c_fused_over_A=s.get('c'), c_evolve_over_A=s.get('c_ev'), t_fused_s=s.get('t_fused'),
                                    t_evolve_s=s.get('t_evolve'), t_A_s=s.get('t_A'), usable=s.get('usable'),
                                    valid_fused=s.get('valid_fused'), fused=s.get('fused'), identity_hashes=s.get('identity_hashes'),
                                    triplet_evidence=s.get('triplet_evidence'), timing_attempt=T.get('attempt'),
                                    job=T['job'], gpu=T['gpu'], contract=s.get('contract')))
        out['meshes'][mesh] = dict(members=[entry(k) for k in F['members']],
                                   other_qualified=[entry(k) for k in F['candidates'] if k not in F['members']],
                                   unqualified={k: v for k, v in F['unqualified'].items()},
                                   fronts=F['fronts'])
    name = 'FRONT-2D.json' if a.version == 1 else f'FRONT-2D-v{a.version}.json'
    b = json.dumps(out, indent=1, default=str) + '\n'
    (HERE / 'runs' / name).write_text(b)
    (HERE / 'runs' / name.replace('.json', '.sha256')).write_text(hashlib.sha256(b.encode()).hexdigest() + f'  {name}\n')
    print(name, hashlib.sha256(b.encode()).hexdigest()[:16], {m: len(v.get('members', [])) for m, v in out['meshes'].items()})


if __name__ == '__main__':
    main()
