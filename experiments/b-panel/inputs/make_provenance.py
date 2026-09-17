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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QTD = ROOT.parent / '2026-09-16-q-trajdirs/experiments/q-trajdirs/artifacts/qtd02'
QRG = ROOT / 'experiments/q-ridge/artifacts/qrg304'
EQTOP = ROOT.parent / '2026-09-17-b-eqtop/experiments/b-eqtop/certified-rules'
SPEED_COMMIT = 'b3f9ecf928805fd512e15ff47d15863bc0d3d268'


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


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
                     'asserted equal to the source job\'s own OUTPUTS.sha256 entry (qtd02, qrg304) or to the '
                     'exporting lane\'s SHA256SUMS and PROVENANCE.json (b-eqtop, rules-eqtop/)'), files={})

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

    # --- b-eqtop's exported rule set (DESIGN A8: the `eqtop` arms of bpn301) --------------
    # Verified three ways: the file SHA256 against b-eqtop's SHA256SUMS AND its PROVENANCE.json,
    # and the nodes / weights SHA256 recomputed here against the per-rule record. The
    # construction status (confirmed / marginal / certified in one draw) is carried verbatim so
    # every table row and caption can show it: at q = 128 and 256 the exported rules are single
    # draws of constructions whose m = 2048 re-draws were marginal (b-eqtop DESIGN A4).
    et = json.loads((EQTOP / 'PROVENANCE.json').read_text())
    sums = dict(line.split('  ', 1)[::-1] for line in (EQTOP / 'SHA256SUMS').read_text().splitlines() if line.strip())
    et_head = subprocess.check_output(['git', '-C', str(EQTOP), 'rev-parse', 'HEAD'], text=True).strip()
    et_dirty = subprocess.check_output(['git', '-C', str(EQTOP), 'status', '--porcelain', str(EQTOP)], text=True).strip()
    assert not et_dirty, f'b-eqtop certified-rules directory has uncommitted changes:\n{et_dirty}'
    by_file = {r['file']: r for r in et['rules']}
    superseded = {x['sha256']: x for x in et.get('superseded', [])}
    local_qrg = {sha(p_): p_.name for p_ in (HERE / 'rules').glob('*.npz')}
    for p in sorted((HERE / 'rules-eqtop').glob('*.npz')):
        rec = by_file[p.name]
        h = sha(p)
        assert h == rec['sha256'] == sums[p.name], (p.name, h, rec['sha256'], sums.get(p.name))
        z = np.load(p)
        nodes, w = np.asarray(z['nodes'], dtype=int), np.asarray(z['weights'], dtype=float)
        assert sha_array(nodes) == rec['nodes_sha256'] and sha_array(w) == rec['weights_sha256'], p.name
        assert len(nodes) == rec['m'] and rec['M'] == 4 * (16 + rec['q']), p.name
        ho = rec['held_out']
        assert ho['bars']['primary'] == et['bars']['primary']
        out['files'][f'rules-eqtop/{p.name}'] = dict(
            sha256=h, source_lane='b-eqtop', source_job=rec['origin']['job_id'], source_attempt=rec['origin']['attempt'],
            source_commit=rec['origin']['commit'], source_gpu=rec['origin']['gpu'],
            source_path=f'experiments/b-eqtop/certified-rules/{p.name}', source_worktree_head=et_head,
            recertified_in=rec['origin'].get('recertified_in'), certified_in=ho.get('certified_in'),
            q=rec['q'], M=rec['M'], m=rec['m'], m_target=rec['m_target'], population=rec['population'],
            construction_arm=rec['construction_arm'], fit_states=rec['fit_states'], candidate_pool=rec['candidate_pool'],
            design_rows=rec['design_rows'], scaling=rec['scaling'], compressed=rec['compressed'],
            relative_fit=rec['nnls_relative_fit'], truncated=rec['truncated'], held_out_states=ho['states'],
            rho_max=ho['rho_max'], rho_p95=ho['rho_p95'], rho_median=ho['rho_median'], rho_bar=et['bars']['primary'],
            certified_primary=ho['certified_primary'], certified_secondary=ho['certified_secondary'],
            certified_tight=ho['certified_tight'], tight_bar=et['bars']['tight'],
            construction_status=rec['construction']['status'], construction_draws=rec['construction']['draws'],
            draws_certifying_primary=rec['construction']['draws_certifying_primary'],
            rho_max_over_draws=dict(min=rec['construction']['rho_max_min'], median=rec['construction']['rho_max_median'],
                                    max=rec['construction']['rho_max_max']),
            export_basis=rec['export_basis'], export_note=rec['export_note'], status_note=rec['status'],
            same_file_as_qrg304_rule=local_qrg.get(h),
            nodes_sha256=rec['nodes_sha256'], weights_sha256=rec['weights_sha256'],
            eqtop_set_status=et['status'], eqtop_status_note=et['status_note'])
    # what b-eqtop's replication says about the qrg304 files this lane carries as `eqcert`
    for key, entry in out['files'].items():
        if not key.startswith('rules/'):
            continue
        h = entry['sha256']
        if h in {r['sha256'] for r in et['rules']}:
            rec = next(r for r in et['rules'] if r['sha256'] == h)
            entry.update(construction_status=rec['construction']['status'], construction_draws=rec['construction']['draws'],
                         construction_assessed_by='b-eqtop (same file, exported there as well)')
        elif h in superseded:
            entry.update(construction_status='marginal (b-eqtop superseded list)', construction_assessed_by='b-eqtop',
                         construction_note=superseded[h]['reason'])
        else:
            entry.update(construction_status=None, construction_assessed_by=None,
                         construction_note='single qrg304 draw; its construction was not re-drawn under this file hash in b-eqtop')

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
