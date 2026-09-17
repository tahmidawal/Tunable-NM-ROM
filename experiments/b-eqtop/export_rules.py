"""Export the cheapest primary-certified EQ rule per rung as one rule set for other lanes.

Reads the ladder job's audit (`checks/bet101-audit.json`) and its `rule_choice`, resolves each
chosen rule to its archived file (qrg304 import or the bet101 collected output), verifies the
file SHA256 against the source manifest and the nodes/weights SHA256 against the job record,
re-checks the rule's validity (unique interior nodes, nonnegative weights, count = m), copies
it under a self-describing name, and writes PROVENANCE.json + README.md.

    python export_rules.py --audit checks/bet101-audit.json --out certified-rules
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent


def sha_bytes(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_arr(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', default=str(HERE / 'checks/bet101-audit.json'))
    p.add_argument('--result', default=str(HERE / 'artifacts/bet101/result.json'))
    p.add_argument('--outputs', default=str(HERE / 'runs/bet101/archive/output'))
    p.add_argument('--manifest', default=str(HERE / 'artifacts/bet101/OUTPUTS.sha256'))
    p.add_argument('--out', default=str(HERE / 'certified-rules'))
    p.add_argument('--status', required=True, help='e.g. "provisional: bet301 (3783811) pending"')
    a = p.parse_args()
    A = json.loads(Path(a.audit).read_text())
    r = json.loads(Path(a.result).read_text())
    L = int(r['intervals'])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    outputs_manifest = {}
    for line in Path(a.manifest).read_text().splitlines():
        h, _, name = line.partition('  ')
        outputs_manifest[name.strip()] = h.strip()
    qrg = json.loads((HERE / 'rules/qrg304/MANIFEST.json').read_text())
    qrg_by_file = {x['file']: x for x in qrg['rules']}
    by_arm = {x['arm']: x for x in A['arms']}
    rules_by_key = {}
    for x in A['rules']:
        rules_by_key.setdefault((x['source'], x['q'], x['arm'], x['m_target']), x)
    records = []
    for choice in A['rule_choice']:
        if choice['bar'] != 'primary' or not choice['chosen']:
            continue
        q, c = choice['q'], choice['chosen']
        if c['source'] == 'archived':
            name = f"rule_q{q}_{c['arm']}_m{c['m_target']}.npz"
            src = HERE / 'rules/qrg304' / name
            expected = qrg_by_file[name]['sha256']
            rec = next(x for x in r['archived_rules'] if x['file'] == name)
            audit_rule = rules_by_key[('qrg304', q, c['arm'], c['m_target'])]
            origin = dict(job_id=qrg['source_job_id'], attempt=qrg['source_attempt'],
                          commit=qrg['source_commit'], gpu=qrg['source_gpu'],
                          recertified_in=dict(job_id=A['job_id'], attempt=A['attempt']))
            tagname = f"rule_q{q}_m{c['m']}_qrg304_{c['arm']}.npz"
        else:
            name = f"rule_q{q}_{c['arm']}_m{c['m_target']}.npz"
            src = Path(a.outputs) / name
            expected = outputs_manifest[f'output/{name}']
            rec = next(x for x in r['rules'] if x['q'] == q and x['arm'] == c['arm']
                       and x['m_target'] == c['m_target'] and not x.get('failed'))
            audit_rule = rules_by_key[('this_job', q, c['arm'], c['m_target'])]
            origin = dict(job_id=A['job_id'], attempt=A['attempt'], commit=A['commit'], gpu=A['gpu'])
            tagname = f"rule_q{q}_m{c['m']}_{A['attempt']}_{c['arm']}.npz"
        got = sha_bytes(src)
        assert got == expected, (name, got, expected)
        z = np.load(src)
        nodes, weights = np.asarray(z['nodes']), np.asarray(z['weights'])
        assert sha_arr(nodes) == rec['nodes_sha256'] and sha_arr(weights) == rec['weights_sha256'], name
        assert len(nodes) == c['m'] == len(weights)
        assert len(np.unique(nodes)) == len(nodes) and nodes.min() >= 0 and nodes.max() < (L - 1) ** 2
        assert np.all(weights >= 0) and np.all(np.isfinite(weights))
        assert abs(audit_rule['rho_max'] - c['rho_max']) <= 1e-12 * max(1., c['rho_max'])
        assert audit_rule['certified_primary'] and not audit_rule['truncated']
        dst = out / tagname
        shutil.copy2(src, dst)
        assert sha_bytes(dst) == expected
        arm = by_arm.get(f'q{q}_eq_primary') or {}
        records.append(dict(
            q=q, M=audit_rule['M'], file=tagname, sha256=expected, source_file=name,
            nodes_sha256=rec['nodes_sha256'], weights_sha256=rec['weights_sha256'],
            origin=origin, source=c['source'], arm=c['arm'], population=audit_rule['population'],
            m=c['m'], m_target=c['m_target'], fit_states=audit_rule['fit_states'],
            candidate_pool=audit_rule['candidates'], design_rows=audit_rule['design_rows'],
            scaling=audit_rule.get('scaling'), compressed=audit_rule.get('compressed'),
            nnls_relative_fit=audit_rule['relative_fit'], fit_seconds=audit_rule.get('fit_seconds'),
            truncated=audit_rule['truncated'],
            held_out=dict(states=512, rho_max=audit_rule['rho_max'], rho_p95=audit_rule['rho_p95'],
                          rho_median=audit_rule['rho_median'], certified_primary=audit_rule['certified_primary'],
                          certified_tight=audit_rule['certified_tight'],
                          certified_secondary=audit_rule['certified_secondary'], bars=A['bars'],
                          certified_in=dict(job_id=A['job_id'], attempt=A['attempt'])),
            timed_ladder_arm=(dict(arm=arm['arm'], worst_evolved_percent=arm['worst_evolved_percent'],
                                   worst_all_times_percent=arm['worst_all_times_percent'],
                                   median_gpu_ms=arm['median_gpu_ms'], converged=arm['converged'],
                                   job_id=A['job_id']) if arm else None),
            status=a.status))
    prov = dict(
        purpose='the cheapest primary-certified empirical-quadrature rule per rung of the Burgers 256^2 '
                'correction ladder, chosen in job bet101 from the qrg304 archive and bet101\'s own fits '
                '(ties by arm order std < fs64 < rhow64, archived first at equal m)',
        status=a.status,
        status_note='every rule is one draw of (candidate pool, fit-state subset); bet201 showed the same '
                    'construction moving rho_max 0.11x-2.3x under a second draw and flipping certification at '
                    'q=64, m=1024. Until bet301 (the pre-registered replication) lands, "certified" means '
                    '"this draw met the bar on 512 held-out reachable states", not "the construction does".',
        frozen=dict(checkpoint_sha256=r['checkpoint_sha256'], intervals=L, K=r['K'], R=r['R'], dt=r['dt'],
                    tests_per_unknown=4, M='4(K+q)'),
        rho_definition=r['rho_definition'], bars=A['bars'],
        node_convention=(f'`nodes` are int64 row-major indices into the (L-1)x(L-1) = {L-1}x{L-1} interior '
                         'grid of the L=256-interval unit square (grid position (i+1, j+1) with '
                         '(i, j) = unravel_index(node, (L-1, L-1))); `weights` are nonnegative float64. '
                         'The rule approximates Phi^T a(u) by sum_j w_j Phi(x_j) a(u)(x_j) with Phi the M '
                         'interior test modes and a(u) the upwind advection; see eqtop.make_rule / '
                         'eqtop.rho_of_field for the exact reconstruction and eqcert.fit_rule for the '
                         'driver\'s use.'),
        audit=str(Path(a.audit).relative_to(HERE)), result=str(Path(a.result).relative_to(HERE)),
        rules=records)
    (out / 'PROVENANCE.json').write_text(json.dumps(prov, indent=2) + '\n')
    lines = ['# Certified EQ rules, one per rung (cheapest primary-certified)', '',
             f'**Status: {a.status}.** See `PROVENANCE.json` (`status_note`) before citing a certified flag.', '',
             'Generated by `export_rules.py` from `checks/bet101-audit.json`; every file SHA256 is verified '
             'against its source manifest and its nodes/weights SHA256 against the job record.', '',
             '| q | M | file | source | arm | fit states | m | rho_max | rho_95 | primary | tight | evolved % (ladder) | GPU ms (ladder) |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for x in records:
        t = x['timed_ladder_arm'] or {}
        lines.append(f"| {x['q']} | {x['M']} | `{x['file']}` | {x['origin']['attempt']} ({x['origin']['job_id']}) | {x['arm']} | "
                     f"{x['fit_states']} | {x['m']} | {x['held_out']['rho_max']:.4f} | {x['held_out']['rho_p95']:.4f} | "
                     f"{'yes' if x['held_out']['certified_primary'] else 'no'} | {'yes' if x['held_out']['certified_tight'] else 'no'} | "
                     f"{t.get('worst_evolved_percent', float('nan')):.4f} | {t.get('median_gpu_ms', float('nan')):.1f} |")
    lines += ['', prov['node_convention'], '',
              'Verify: `sha256sum -c SHA256SUMS` in this directory.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    (out / 'SHA256SUMS').write_text(''.join(f"{x['sha256']}  {x['file']}\n" for x in records))
    print(out)
    for x in records:
        print(f"q={x['q']:3d} {x['file']:44s} fs={x['fit_states']:2d} m={x['m']:4d} rho_max={x['held_out']['rho_max']:.4f}")


if __name__ == '__main__':
    main()
