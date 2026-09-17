"""Export one EQ rule per rung as the rule set other lanes (`b-panel`) drop into their runs.

Policy (DESIGN §A4, after the draw replication `bet301`), computed by `draws.export_choice`
from the audit JSONs alone:

  (i)  the cheapest *confirmed* construction — every one of at least two independent draws of
       (q, m target, fit-state count, row scaling) meets the primary bar; the concrete rule is
       the one the timed ladder ran in `bet101` if it belongs to that construction, otherwise
       this lane's ordinary (pool-16384) draw of it;
  (ii) where no construction at the rung is confirmed (q = 128, 256): the cheapest
       primary-certified rule at an m above every m found marginal, labelled
       `certified in one draw` — a re-draw of it has NOT been measured.

Before `bet301` the set was `bet101`'s in-job choice (`rule_choice`, bar primary); those files
are removed from the directory and listed under `superseded` in PROVENANCE.json.

Each file is SHA256-verified against its job's OUTPUTS manifest (or the qrg304 import
manifest) and its nodes/weights SHA256 against the job record; the rule is re-checked
(unique interior nodes, nonnegative finite weights, count = m); PROVENANCE.json, README.md
and SHA256SUMS are written.

    python export_rules.py --status "<status line>"
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import draws as DR  # noqa: E402


def sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def read_manifest(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        h, _, name = line.partition('  ')
        out[name.strip()] = h.strip()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--j1', default=str(HERE / 'checks/bet101-audit.json'))
    p.add_argument('--j2', default=str(HERE / 'checks/bet201-audit.json'))
    p.add_argument('--j3', default=str(HERE / 'checks/bet301-audit.json'))
    p.add_argument('--out', default=str(HERE / 'certified-rules'))
    p.add_argument('--status', required=True, help='the status line written into PROVENANCE.json')
    a = p.parse_args()
    A, B, C = (json.loads(Path(x).read_text()) for x in (a.j1, a.j2, a.j3))
    jobs = [A, B, C]
    results = {j['attempt']: json.loads((HERE / 'artifacts' / j['attempt'] / 'result.json').read_text()) for j in jobs}
    manifests = {j['attempt']: read_manifest(HERE / 'artifacts' / j['attempt'] / 'OUTPUTS.sha256') for j in jobs}
    r1 = results['bet101']
    L = int(r1['intervals'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    qrg = json.loads((HERE / 'rules/qrg304/MANIFEST.json').read_text())
    qrg_by_file = {x['file']: x for x in qrg['rules']}
    by_arm = {x['arm']: x for x in A['arms']}

    DRAWS = DR.collect_draws(jobs)
    CONS = DR.constructions(DRAWS)
    lc = {c['q']: c['chosen'] for c in A['rule_choice'] if c['bar'] == 'primary' and c['chosen']}
    previous = None
    if (out / 'PROVENANCE.json').exists():
        previous = json.loads((out / 'PROVENANCE.json').read_text())

    records = []
    for q in sorted({c['q'] for c in CONS}):
        d, basis, note = DR.export_choice(CONS, DRAWS, q, lc.get(q))
        assert d is not None, f'no exportable rule at q={q}: {note}'
        c = next(c for c in CONS if c['construction'] == d['construction'])
        x = d['rule']
        if d['source'] == 'qrg304':
            name = f"rule_q{q}_{x['arm']}_m{x['m_target']}.npz"
            src = HERE / 'rules/qrg304' / name
            expected = qrg_by_file[name]['sha256']
            rec = next(z for z in r1['archived_rules'] if z['file'] == name)
            origin = dict(job_id=qrg['source_job_id'], attempt=qrg['source_attempt'],
                          commit=qrg['source_commit'], gpu=qrg['source_gpu'],
                          recertified_in=dict(job_id=A['job_id'], attempt=A['attempt']))
            tagname = f"rule_q{q}_m{x['m']}_qrg304_{x['arm']}.npz"
        else:
            att = d['attempt']
            name = f"rule_q{q}_{x['arm']}_m{x['m_target']}.npz"
            src = HERE / 'runs' / att / 'archive/output' / name
            expected = manifests[att][f'output/{name}']
            rec = next(z for z in results[att]['rules'] if z['q'] == q and z['arm'] == x['arm']
                       and z['m_target'] == x['m_target'] and not z.get('failed'))
            J = next(j for j in jobs if j['attempt'] == att)
            origin = dict(job_id=J['job_id'], attempt=att, commit=J['commit'], gpu=J['gpu'])
            tagname = f"rule_q{q}_m{x['m']}_{att}_{x['arm']}.npz"
        got = sha_bytes(src)
        assert got == expected, (name, got, expected)
        z = np.load(src)
        nodes, weights = np.asarray(z['nodes']), np.asarray(z['weights'])
        assert sha_arr(nodes) == rec['nodes_sha256'] and sha_arr(weights) == rec['weights_sha256'], name
        assert len(nodes) == x['m'] == len(weights)
        assert len(np.unique(nodes)) == len(nodes) and nodes.min() >= 0 and nodes.max() < (L - 1) ** 2
        assert np.all(weights >= 0) and np.all(np.isfinite(weights))
        assert x['certified_primary'] and not x['truncated']
        dst = out / tagname
        shutil.copy2(src, dst)
        assert sha_bytes(dst) == expected
        ran = (lc.get(q) is not None and x['arm'] == lc[q]['arm'] and x['m_target'] == lc[q]['m_target']
               and abs(x['rho_max'] - lc[q]['rho_max']) <= 1e-9)
        arm = by_arm.get(f'q{q}_eq_primary') if ran else None
        records.append(dict(
            q=q, M=x['M'], file=tagname, sha256=expected, source_file=name,
            nodes_sha256=rec['nodes_sha256'], weights_sha256=rec['weights_sha256'],
            origin=origin, source=d['source'], arm=x['arm'], construction_arm=d['arm'],
            population=x['population'], m=x['m'], m_target=x['m_target'], fit_states=x['fit_states'],
            candidate_pool=x['candidates'], design_rows=x['design_rows'], scaling=x.get('scaling'),
            compressed=x.get('compressed'), nnls_relative_fit=x['relative_fit'],
            fit_seconds=x.get('fit_seconds'), truncated=x['truncated'],
            held_out=dict(states=512, rho_max=x['rho_max'], rho_p95=x['rho_p95'], rho_median=x['rho_median'],
                          certified_primary=x['certified_primary'], certified_tight=x['certified_tight'],
                          certified_secondary=x['certified_secondary'], bars=A['bars'],
                          certified_in=dict(job_id=d['job_id'], attempt=d['attempt'])),
            construction=dict(q=q, m_target=c['m_target'], fit_states=c['fit_states'], scaling=c['scaling'],
                              draws=c['n'], draws_certifying_primary=c['certified_primary_count'],
                              draws_certifying_tight=c['certified_tight_count'],
                              rho_max_min=c['rho_min'], rho_max_median=c['rho_median'],
                              rho_max_max=c['rho_max_of_draws'], spread_ratio=c['spread_ratio'],
                              status=c['status'],
                              each_draw=[dict(label=dd['label'], job_id=dd['job_id'], m=dd['m'], rho_max=dd['rho_max'],
                                              certified_primary=dd['certified_primary']) for dd in c['draws']]),
            export_basis=basis, export_note=note,
            is_the_rule_the_timed_ladder_ran=bool(ran),
            timed_ladder_arm=(dict(arm=arm['arm'], worst_evolved_percent=arm['worst_evolved_percent'],
                                   worst_all_times_percent=arm['worst_all_times_percent'],
                                   median_gpu_ms=arm['median_gpu_ms'], converged=arm['converged'],
                                   job_id=A['job_id']) if arm else None),
            status=(c['status'] if basis == 'confirmed' else
                    f"{c['status']}: a re-draw of this construction has not been measured; the same "
                    f"recipe at m=2048 was marginal (see export_note)")))

    # remove superseded files, record them
    keep = {r_['file'] for r_ in records}
    superseded = []
    for f_ in sorted(out.glob('*.npz')):
        if f_.name not in keep:
            prev = next((r_ for r_ in (previous or {}).get('rules', []) if r_['file'] == f_.name), None)
            superseded.append(dict(file=f_.name, sha256=sha_bytes(f_), q=(prev or {}).get('q'),
                                   m=(prev or {}).get('m'), rho_max=((prev or {}).get('held_out') or {}).get('rho_max'),
                                   reason='its construction is marginal after bet301 (DESIGN §A4); still archived in '
                                          'artifacts/ and listed in the report',
                                   removed=True))
            f_.unlink()
    prov = dict(
        purpose='one empirical-quadrature rule per rung of the Burgers 256^2 correction ladder for other lanes: '
                'the cheapest CONFIRMED construction (every independent draw meets the primary bar) where one '
                'exists, else the cheapest certified single-draw rule above every m found marginal (DESIGN §A4)',
        status=a.status,
        status_note='"certified" for a single rule means: this rule met rho_max <= 0.116 on 512 held-out reachable '
                    'states. Whether the CONSTRUCTION certifies is a separate question answered per rule under '
                    '`construction.status` from every independent draw (qrg304, bet101, bet201, bet301): '
                    '`confirmed (k/k)`, `marginal at m (k/n)`, `certified in one draw`. At q = 128 and 256 no '
                    'construction is confirmed: the exported rules there are single draws at m above the marginal '
                    'm = 2048, and a re-draw could fail the bar. Carry this field with the rule.',
        policy='draws.export_choice: (i) cheapest confirmed construction, concrete rule = the one the timed ladder '
               'ran if it belongs to it, else this lane\'s ordinary pool-16384 draw; (ii) else cheapest '
               'primary-certified rule with m above every marginal m, ties by arm order std < fs64 < rhow64',
        frozen=dict(checkpoint_sha256=r1['checkpoint_sha256'], intervals=L, K=r1['K'], R=r1['R'], dt=r1['dt'],
                    tests_per_unknown=4, M='4(K+q)'),
        rho_definition=r1['rho_definition'], bars=A['bars'],
        node_convention=(f'`nodes` are int64 row-major indices into the (L-1)x(L-1) = {L-1}x{L-1} interior '
                         'grid of the L=256-interval unit square (grid position (i+1, j+1) with '
                         '(i, j) = unravel_index(node, (L-1, L-1))); `weights` are nonnegative float64. '
                         'The rule approximates Phi^T a(u) by sum_j w_j Phi(x_j) a(u)(x_j) with Phi the M '
                         'interior test modes and a(u) the upwind advection; see eqtop.make_rule / '
                         'eqtop.rho_of_field for the exact reconstruction and eqcert.fit_rule for the '
                         'driver\'s use.'),
        audits=[str(Path(x).relative_to(HERE)) for x in (a.j1, a.j2, a.j3)],
        jobs=[dict(attempt=j['attempt'], job_id=j['job_id'], commit=j['commit'], gpu=j['gpu']) for j in jobs],
        superseded=superseded, rules=records)
    (out / 'PROVENANCE.json').write_text(json.dumps(prov, indent=2) + '\n')
    lines = ['# EQ rules, one per rung, for other lanes (DESIGN §A4 policy, after the draw replication)', '',
             f'**Status: {a.status}.**', '',
             'Read `PROVENANCE.json` (`status_note`, and `construction.status` per rule) before citing a certified '
             'flag: at q = 128 and 256 the exported rule is a **single draw** of a construction whose re-draws at '
             'm = 2048 were marginal; a fresh draw of it has not been measured.', '',
             'Generated by `export_rules.py` from the three audit JSONs; every file SHA256 is verified against its '
             'source manifest and its nodes/weights SHA256 against the job record.', '',
             '| q | M | file | source | arm | fit states | m | rho_max | rho_95 | primary | tight | construction status (all draws) | draws certifying | rho_max over draws min / median / max | basis | ladder evolved % (bet101, if this rule ran) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in records:
        t = x['timed_ladder_arm'] or {}
        c = x['construction']
        lines.append(f"| {x['q']} | {x['M']} | `{x['file']}` | {x['origin']['attempt']} ({x['origin']['job_id']}) | {x['arm']} | "
                     f"{x['fit_states']} | {x['m']} | {x['held_out']['rho_max']:.4f} | {x['held_out']['rho_p95']:.4f} | "
                     f"{'yes' if x['held_out']['certified_primary'] else 'no'} | {'yes' if x['held_out']['certified_tight'] else 'no'} | "
                     f"**{c['status']}** | {c['draws_certifying_primary']}/{c['draws']} | "
                     f"{c['rho_max_min']:.4f} / {c['rho_max_median']:.4f} / {c['rho_max_max']:.4f} | {x['export_basis']} | "
                     f"{(format(t['worst_evolved_percent'], '.4f') if t else '— (not run in the timed ladder)')} |")
    if superseded:
        lines += ['', '## Superseded (removed from this directory after `bet301`)', '',
                  '| file | q | m | rho_max | reason |', '|---|---|---|---|---|']
        lines += [f"| `{s_['file']}` | {s_['q']} | {s_['m']} | {s_['rho_max']:.4f} | {s_['reason']} |" for s_ in superseded]
    lines += ['', prov['node_convention'], '', 'Verify: `sha256sum -c SHA256SUMS` in this directory.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['file']}\n" for x in records))
    print(out)
    for x in records:
        print(f"q={x['q']:3d} {x['file']:44s} fs={x['fit_states']:2d} m={x['m']:4d} rho_max={x['held_out']['rho_max']:.4f} "
              f"[{x['construction']['status']}] {x['export_basis']}")
    for s_ in superseded:
        print('superseded', s_['file'])


if __name__ == '__main__':
    main()
