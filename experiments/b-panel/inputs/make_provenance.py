"""Write `inputs/PROVENANCE.json`: where every frozen input of this lane came from.

Nothing here is typed by hand. Every SHA256 is recomputed from the file in this directory and
checked against the line the SOURCE job's own `OUTPUTS.sha256` recorded for it, so a frozen
input that drifted from its archive cannot pass. The b-speed kernels are checked against the
Git blob at the commit they were taken from.

    python inputs/make_provenance.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QTD = ROOT.parent / '2026-09-16-q-trajdirs/experiments/q-trajdirs/artifacts/qtd02'
QRG = ROOT / 'experiments/q-ridge/artifacts/qrg304'
SPEED_COMMIT = 'b3f9ecf928805fd512e15ff47d15863bc0d3d268'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def outputs_table(archive):
    rows = {}
    for line in (archive / 'OUTPUTS.sha256').read_text().splitlines():
        h, name = line.split('  ', 1)
        rows[name] = h
    return rows


def main():
    out = dict(note=('frozen inputs of the b-panel lane; every SHA256 below is recomputed here and '
                     'asserted equal to the source job\'s own OUTPUTS.sha256 entry'), files={})

    # --- the dense ladder's directions (qtd02, job 3757505) --------------------------
    qtd_out = outputs_table(QTD)
    qtd_res = json.loads((QTD / 'result.json').read_text())
    f = HERE / 'directions_qtd02.npz'
    h = sha(f)
    assert h == qtd_out['output/directions_old.npz'], (h, qtd_out['output/directions_old.npz'])
    ds = qtd_res['direction_sets']['old']
    out['files']['directions_qtd02.npz'] = dict(
        sha256=h, source_job=qtd_res['job_id'], source_attempt='qtd02',
        source_commit=(QTD / 'COMMIT.txt').read_text().strip(),
        source_archive_sha256=json.loads((QTD / 'archive.json').read_text())['sha256'],
        source_path='output/directions_old.npz', columns=ds['columns'],
        available_rank=ds['available_rank'], prefix_sha256=ds['prefix_sha256'],
        field_orthonormal_prefix_sha256=ds['field_orthonormal_prefix_sha256'],
        directions_sha256=ds['directions_sha256'],
        rule=ds['rule'], gpu=qtd_res['gpu'],
        role=('C_q for every q > 0 arm in this lane, in BOTH meshes: the frozen model is '
              '(bank, head, C) and C is transferred, not refitted'))

    # --- the certified EQ rules (qrg304, job 3768168) ---------------------------------
    qrg_out = outputs_table(QRG)
    qrg_res = json.loads((QRG / 'result.json').read_text())
    choice = {c['q']: c for c in qrg_res['rule_choice']}
    rules = {x['q']: x for x in qrg_res['rules'] if x['population'] == 'reachable'}
    by_key = {(x['q'], x['m_target']): x for x in qrg_res['rules'] if x['population'] == 'reachable'}
    for p in sorted((HERE / 'rules').glob('rule_q*_reachable_m*.npz')):
        name = f'output/{p.name}'
        h = sha(p)
        assert h == qrg_out[name], (p.name, h, qrg_out[name])
        q = int(p.name.split('_')[1][1:])
        m = int(p.name.split('_m')[1].split('.')[0])
        info = by_key[(q, m)]
        c = choice[q]
        out['files'][f'rules/{p.name}'] = dict(
            sha256=h, source_job=qrg_res['job_id'], source_attempt='qrg304',
            source_commit=(QRG / 'COMMIT.txt').read_text().strip(),
            source_archive_sha256=json.loads((QRG / 'archive.json').read_text())['sha256'],
            source_path=name, q=q, M=info['M'], m=info['m'], m_target=info['m_target'],
            population='reachable', fitter=info['fitter'], relative_fit=info['relative_fit'],
            truncated=info['truncated'], rho_max=info['certification']['rho_max'],
            rho_p95=info['certification']['rho_p95'], rho_bar=info['rho_bar'],
            certified_primary=info['certified_primary'],
            certified_secondary=info['certified_secondary'],
            nodes_sha256=info['nodes_sha256'], weights_sha256=info['weights_sha256'],
            chosen_by_qrg304=(c['chosen_m'] == m), qrg304_basis=(c['basis'] if c['chosen_m'] == m else None),
            gpu=qrg_res['gpu'])

    # --- the b-speed kernels ----------------------------------------------------------
    for name in ('fast.py', 'ladders.py'):
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                                        f'{SPEED_COMMIT}:experiments/b-speed/{name}'])
        local = (HERE.parent / 'speed' / name).read_bytes()
        assert blob == local, name
        out['files'][f'speed/{name}'] = dict(
            sha256=hashlib.sha256(local).hexdigest(), source_commit=SPEED_COMMIT,
            source_path=f'experiments/b-speed/{name}', source_branch='exp/2026-09-16-b-speed',
            role='the parity-gated optimised q = 0 kernel (arm L4), copied byte for byte')
    (HERE / 'PROVENANCE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('WROTE', HERE / 'PROVENANCE.json', len(out['files']), 'files verified')


if __name__ == '__main__':
    main()
