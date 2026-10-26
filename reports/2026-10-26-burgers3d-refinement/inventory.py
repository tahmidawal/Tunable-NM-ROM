"""Reconcile historical Burgers 3D evidence without changing its scientific predicates.

CPU-only: reads explicit unsealed score files/metadata, never traverses reference arrays.
Summary records are not solver invocations. Scored observations retain their own populations;
qualification from another observation is cited separately and never promoted to a new pass.
"""
from __future__ import annotations

import argparse
import collections
import functools
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
LANES = {
    'quadrature': ('2026-10-01-quadrature-burgers3d', 'quadrature-burgers3d'),
    'references': ('2026-10-06-jcp-references', 'jcp-references'),
    'combined': ('2026-10-13-jcp-combined', 'jcp-combined'),
    'speed': ('2026-10-14-jcp-speed3d', 'jcp-speed3d'),
    'qualify': ('2026-10-18-jcp-qualify', 'jcp-qualify'),
    'dims': ('2026-10-18-jcp-dims', 'jcp-dims'),
    'budget': ('2026-10-18-jcp-budget', 'jcp-budget'),
    'pod': ('2026-10-06-jcp-hr-baselines', 'jcp-hr-baselines'),
    'robust': ('2026-10-19-jcp-robust', 'jcp-robust'),
}
BANK_HASH = {
    'M2': '6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9',
    'W1024': '86094f6884e55643240197e560f7dc1924bac3885aed79b20faf2e54446aa036',
}
EXPECTED_STORE = '3ec3b5492748a162c9d77e7c09cdbe1485d449a31bd73a29175b4a0e4efc66a0'
EXPECTED_STATUS = 'af7d824e94b42afb83753d6b11a0c4d71407cff0923ebee9c245eb8d51f81005'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=32)
def read(path):
    p = Path(path)
    if p.suffix == '.gz':
        with gzip.open(p, 'rt') as f:
            return json.load(f)
    return json.loads(p.read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def clean(value):
    """JSON null means a nonfinite archived value; never silently coerce it to zero."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def lane_dirs(root):
    return {k: root / 'worktrees' / wt / 'experiments' / exp for k, (wt, exp) in LANES.items()}


def identity(key, record=None):
    record = record or {}
    p = key.split('|')
    # Explicit M labels are part of the identity, not stripped from the record key.
    match = re.match(r'^(M2_512|W(\d+))', p[0])
    if match is None:
        return {'key': key, 'parse_status': 'unsupported'}
    r = 512 if p[0].startswith('M2') else int(match.group(2))
    rule = p[1]
    rm = re.match(r'([a-z]+)(\d+)', rule)
    m = None
    if rm:
        family, n = rm.group(1), int(rm.group(2))
        if family == 'gl':
            m = n ** 3
        elif family in ('lat', 'mc', 'hal', 'sobol'):
            m = n
        elif family in ('smol', 'smolyak'):
            m = {8: 2559, 9: 6271, 10: 15103, 11: 35839, 12: 83967, 13: 194559}.get(n)
    solver_i, nodes, contract = 2, 65, 'mesh'
    if len(p) > 3 and p[2] in ('mf', 'mesh'):
        contract = p[2]
        if not re.fullmatch(r'n\d+', p[3]):
            return {'key': key, 'parse_status': 'unsupported mesh tag'}
        nodes, solver_i = int(p[3][1:]), 4
    typ = p[solver_i] if len(p) > solver_i else None
    scheme_i = solver_i + (2 if typ == 'ad' else 1)
    bank = 'M2' if p[0].startswith('M2') else 'W1024'
    return dict(key=key, setting=p[0], bank=bank, bank_sha256=BANK_HASH[bank], r=r,
                M=record.get('M'), rule=rule, m=m, solver=typ,
                scheme=p[scheme_i] if len(p) > scheme_i else None,
                dt_label=p[scheme_i + 1] if len(p) > scheme_i + 1 else None,
                policy=p[scheme_i + 2:] if len(p) > scheme_i + 2 else [],
                mesh_nodes=nodes, query_input_contract=contract, scoring_nodes_per_axis=63,
                parse_status='parsed', mode_list_identity='see pinned raw run config/tables')


def refs(dirs):
    d = dirs['references']
    sys.path.insert(0, str(d))
    import refstore as rs
    assert Path(rs.__file__).parent == d
    assert rs.store_version() == EXPECTED_STORE, 'reference-store pin changed'
    assert sha(d / 'store/STATUS.json') == EXPECTED_STATUS, 'reference-status pin changed'
    statuses = {}
    for cohort, count, seed in [('dev4', 4, 923651), ('val64', 64, 923801), ('test32', 32, 923901)]:
        entries = {str(i): rs.reference_status(3, cohort, i, 'ST') for i in range(count)}
        statuses[cohort] = dict(seed=seed, selection_allowed=cohort != 'test32',
                                cases=entries, eligible=sum(v['eligible'] for v in entries.values()))
    return dict(store_version=EXPECTED_STORE, status_sha256=EXPECTED_STATUS, kind='ST',
                field='finest', uncertainty='conditional Richardson estimate, not a bound',
                output_times=[0, .05, .1, .15, .2, .25], scoring_lattice='63^3 interior, x slowest',
                denominator='norm of reference initial field', cohorts=statuses)


def resolve_run(lane, run, attempt):
    """Known old archive paths moved to slim. Select only by identity, not latest mtime."""
    p = Path(run)
    candidates = [p if p.is_absolute() else lane / p]
    old = str(candidates[0])
    if '/archive/' in old:
        candidates.append(Path(old.replace('/archive/', '/slim/')))
    for base in [lane / 'runs' / attempt / 'slim/output', lane / 'runs' / attempt / 'archive/output']:
        if base.is_dir():
            candidates.append(base)
            candidates.extend(sorted(base.glob('*')))
    found = []
    for q in candidates:
        if (q / 'result.json').is_file():
            found.append(q.resolve())
    found = list(dict.fromkeys(found))
    if not found:
        return None
    # An archive/output parent and its nested driver should not both have distinct result payloads.
    hashes = {sha(q / 'result.json') for q in found}
    if len(hashes) != 1:
        raise ValueError(f'ambiguous archived result for {lane.name}/{attempt}: {found}')
    return found[0]


def coefficients(run, key):
    if run is None:
        return dict(status='missing-local-run')
    res = read(run / 'result.json')
    candidates = []
    mode = res.get('mode')
    if mode:
        candidates.append(run / f'W_{mode}.npz')
    candidates += sorted(run.glob('W_*.npz'))
    for p in dict.fromkeys(candidates):
        if not p.is_file():
            continue
        with np.load(p, allow_pickle=False) as z:
            if key in z.files:
                ids = z.get('cases|' + key)
                # Read the NPY header, not every saved coefficient array, just to obtain shape.
                with zipfile.ZipFile(p) as archive:
                    with archive.open(key + '.npy') as member:
                        version = np.lib.format.read_magic(member)
                        reader = {(1, 0): np.lib.format.read_array_header_1_0,
                                  (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                        if reader is None:
                            raise ValueError(f'unsupported NPY header version {version}: {p}/{key}')
                        shape, _, _ = reader(member)
                return dict(status='available', path=str(p), array_key=key,
                            shape=list(shape), case_ids=None if ids is None else ids.tolist(),
                            all_internal_states=('wall|' + key) in z.files)
    # Adaptive outputs use separate AD_<key>.npz rather than the shared W archive.
    return dict(status='not-in-shared-W-archive', run=str(run))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--out', type=Path, default=HERE / 'data')
    args = ap.parse_args()
    dirs = lane_dirs(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {}

    def source(p, role):
        p = Path(p)
        if str(p) not in provenance:
            provenance[str(p)] = dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size, roles=[])
        entry = provenance[str(p)]
        if role not in entry['roles']:
            entry['roles'].append(role)
        return str(p)

    reference = refs(dirs)
    source(dirs['references'] / 'store/MANIFEST.sha256', 'reference store manifest')
    source(dirs['references'] / 'store/STATUS.json', 'reference eligibility')
    dims_path = dirs['dims'] / 'checks/analysis-3d.json'
    speed_path = dirs['speed'] / 'checks/analysis-grid.json'
    dims, speed = read(dims_path), read(speed_path)
    qualify_path = dirs['qualify'] / 'runs/EXPORT.json'
    qualify = read(qualify_path)['threeD']['tested']
    source(qualify_path, 'earlier 3D Q2 rounds')
    source(dims_path, 'qualification summary')
    source(speed_path, 'screening summary')
    summaries = []
    for lane, items, src in [('dims', dims['configurations'], dims_path), ('speed', speed['configs'], speed_path)]:
        for key, value in items.items():
            q1 = value.get('Q1_pass') if lane == 'dims' else value.get('eligible')
            q2 = value.get('Q2', {}).get('label', 'unverified')
            qualified = bool(value.get('qualified', False)) if lane == 'dims' else None
            summaries.append(dict(summary_id=f'{lane}|{key}', source=str(src), lane=lane,
                                  identity=identity(key, value), Q1_reported=q1, Q2_reported=q2,
                                  qualified_reported=qualified, payload=clean(value),
                                  status='historical aggregate, not a solver invocation',
                                  note='speed eligibility predates Q2; never treat as full qualification'))
    for key, value in dims['rounds'].items():
        screening = speed['configs'].get(key, {})
        summaries.append(dict(summary_id=f'dims-round|{key}', source=str(dims_path), lane='dims-round',
                              identity=identity(key), Q1_reported=screening.get('eligible'),
                              Q2_reported=value.get('label', 'missing'),
                              qualified_reported=key in dims['qualified_3d'], payload=clean(value),
                              Q1_source=str(speed_path), status='historical Q2 round with cited Q1'))
    for key, label in qualify.items():
        summaries.append(dict(summary_id=f'qualify|{key}', source=str(qualify_path), lane='qualify',
                              identity=identity(key), Q1_reported=speed['configs'].get(key, {}).get('eligible'),
                              Q2_reported=label, qualified_reported=None,
                              status='earlier Q2 evidence; retained even if a later round exists'))

    observations, cases_out, skips, raw_seen = [], [], [], {}
    coeff_cache = {}
    for lane_name in ['speed', 'dims', 'combined']:
        lane = dirs[lane_name]
        for p in sorted((lane / 'checks').glob('scores-*.json')):
            if lane_name == 'combined' and '3d_e0' not in p.name:
                continue
            doc = read(p)
            seed = doc.get('seed', doc.get('registry_entry', {}).get('cohort_seed'))
            cohort = doc.get('cohort', doc.get('registry_entry', {}).get('cohort'))
            if seed not in (923651, 923801, 923901):
                skips.append(dict(source=str(p), reason='not approved 3D cohort schema'))
                continue
            attempt = p.name[7:].replace('.json', '').replace('-3d_e0', '')
            run = resolve_run(lane, doc.get('run', ''), attempt)
            rp, raw = (None, {}) if run is None else (run / 'result.json', read(run / 'result.json'))
            source(p, 'archived E0 per-case score')
            if rp:
                source(rp, 'raw invocation/configuration/timing metadata')
                raw_seen[str(rp)] = dict(path=str(rp), lane=lane_name, job_id=raw.get('job_id'),
                                         gpu=raw.get('gpu'), commit=raw.get('commit'),
                                         config=raw.get('config'), mode=raw.get('mode'), complete=raw.get('complete'))
            scores_to_process = doc.get('scores', {})
            if lane_name == 'combined':
                grouped = {}
                for fullkey, sc in scores_to_process.items():
                    base, cid_text = fullkey.rsplit('|', 1)
                    if not cid_text.isdigit() or not isinstance(sc, dict):
                        raise ValueError(f'unsupported combined case key: {p}/{fullkey}')
                    cid = int(cid_text)
                    norm = sc.get('ref', sc.get('ST'))
                    if not isinstance(norm, dict) or 'worst' not in norm:
                        raise ValueError(f'unsupported combined score: {p}/{fullkey}')
                    ref_case = reference['cohorts'][cohort]['cases'][str(cid)]
                    grouped.setdefault(base, {})[str(cid)] = dict(e=norm['worst'], e_t=norm['per_time'],
                        eligible=ref_case['eligible'], eps_ref=ref_case['eps_ref'], original_score_key=fullkey)
                scores_to_process = grouped
            for key, sc in scores_to_process.items():
                if not isinstance(sc, dict):
                    continue
                if not sc or not all(str(k).isdigit() for k in sc):
                    skips.append(dict(source=str(p), key=key, reason='unsupported score mapping'))
                    continue
                records, key_config = sc, key
                ids = sorted(int(j) for j in records)
                errs, eligible = [], []
                oid = f'{lane_name}|{attempt}|{key}'
                for cid in ids:
                    entry = records[str(cid)]
                    rs = reference['cohorts'][cohort]['cases'][str(cid)]
                    if bool(entry.get('eligible')) != rs['eligible']:
                        raise ValueError(f'eligibility mismatch {oid}/{cid}')
                    if entry.get('eps_ref') is not None and not math.isclose(float(entry['eps_ref']), rs['eps_ref'], rel_tol=1e-10):
                        raise ValueError(f'uncertainty mismatch {oid}/{cid}')
                    ev = entry.get('e')
                    finite = ev is not None and math.isfinite(float(ev))
                    if finite and rs['eligible']:
                        errs.append(float(ev)); eligible.append(cid)
                    cases_out.append(clean(dict(observation_id=oid, case_id=cid, cohort=cohort,
                                               seed=seed, reference_kind='ST', reference_eligible=rs['eligible'],
                                               finite=finite, e=ev, eps_ref=rs['eps_ref'], e_t=entry.get('e_t'),
                                               source=str(p), score_key=entry.get('original_score_key', key))))
                ck = (str(run), key_config)
                if ck not in coeff_cache:
                    coeff_cache[ck] = coefficients(run, key_config)
                coef = coeff_cache[ck]
                if coef.get('path'):
                    source(Path(coef['path']), 'saved coefficient trajectories')
                ids_match = coef.get('case_ids') is None or ids == sorted(coef['case_ids'])
                if not ids_match:
                    raise ValueError(f'coefficient/score population mismatch {oid}')
                authority = dims['configurations'].get(key_config)
                qualifier = ('dims|' + key_config) if authority is not None else (
                    'dims-round|' + key_config if key_config in dims['rounds'] else (
                        'qualify|' + key_config if key_config in qualify else None))
                ident = identity(key_config, authority)
                setting_meta = raw.get('settings_M', {}).get(ident.get('setting'), {})
                setting_meta = setting_meta or raw.get('settings', {}).get(ident.get('setting'), {})
                ident['metadata_validation'] = {}
                for raw_name, field in [('Rp', 'r'), ('M', 'M'), ('bank', 'bank')]:
                    if raw_name in setting_meta:
                        previous = ident.get(field)
                        if previous is not None and previous != setting_meta[raw_name]:
                            raise ValueError(f'raw identity disagrees with key/qualification: {oid}/{field}')
                        ident[field] = setting_meta[raw_name]
                        ident['metadata_validation'][field] = 'raw run setting'
                if setting_meta.get('mode_hash'):
                    ident['mode_list_sha256'] = setting_meta['mode_hash']
                ident['trust_radius'] = setting_meta.get('trust')
                if isinstance(raw.get('mesh'), int):
                    if ident.get('mesh_nodes') != raw['mesh']:
                        raise ValueError(f'mesh identity disagrees: {oid}')
                    ident['metadata_validation']['mesh_nodes'] = 'raw run mesh'
                rule_meta = setting_meta.get('rules', {}).get(ident.get('rule'), {})
                if 'm' in rule_meta:
                    if ident.get('m') is not None and ident['m'] != rule_meta['m']:
                        raise ValueError(f'point-count identity disagrees: {oid}')
                    ident['m'] = rule_meta['m']
                    ident['metadata_validation']['m'] = 'raw run rule'
                observations.append(clean(dict(observation_id=oid, lane=lane_name, attempt=attempt,
                    config_key=key_config, identity=ident, seed=seed, cohort=cohort,
                    identity_validation='raw setting fields checked where present; remaining fields inferred/unknown',
                    case_ids=ids, eligible_ids=eligible, n_case_records=len(ids), n_scored=len(errs),
                    registered_population_check='not reconstructed; exact saved score/coefficient ids checked',
                    median=float(np.median(errs)) if errs else None, worst=max(errs) if errs else None,
                    errors_above_1pct=sum(e > .01 for e in errs), source=str(p),
                    result=str(rp) if rp else None, job_id=doc.get('job_id', raw.get('job_id')),
                    source_commit=raw.get('commit'), gpu=raw.get('gpu'), coefficients=coef,
                    score_reference_store=doc.get('store_version', doc.get('registry_entry', {}).get('store_version')),
                    qualification_evidence=qualifier,
                    qualification_scope='cited historical aggregate; not a new invocation-level pass',
                    posthoc_rule_label=doc.get('rule_selection_label'),
                    timing_status='see same-invocation raw timing; no cross-job seconds constructed',
                    selection_allowed=cohort != 'test32')))

    # Source inventory covers all requested lanes; unsupported schemas remain explicit sources.
    for lane_name, lane in dirs.items():
        for pattern in ['report.md', 'DESIGN.md', 'checks/*3d*.json', 'checks/analysis-pm.json',
                        'checks/VERDICT-c3d64.json', 'checks/*c3d64.json',
                        'checks/FROZEN-STAGEB-c3d64.json', 'runs/EXPORT.json', 'audits/*.md']:
            for p in sorted(lane.glob(pattern)):
                if p.is_file() and p.stat().st_size <= 50 * 1024 ** 2:
                    source(p, f'{lane_name}: supporting evidence; not automatically normalised')
    nominal_qualified = sum(len(v) for v in dims['coverage'].values())
    report = dict(schema_version=1, scope='3D historical aggregates and E0-scored saved observations',
                  base_commit='93f399faad67f94ea2b56fe25f13003143e64694',
                  source_lane_heads={k: subprocess.check_output(['git', '-C', str(d), 'rev-parse', 'HEAD'], text=True).strip()
                                     for k, d in dirs.items()},
                  summary_counts=dict(speed=len(speed['configs']), dims=len(dims['configurations']),
                      dims_qualified=sum(bool(v['qualified']) for v in dims['configurations'].values()),
                      dims_rounds=len(dims['rounds']), dims_qualified_union=len(dims['qualified_3d']),
                      qualify_earlier_rounds=len(qualify),
                      dims_nominal_width_qualified=nominal_qualified, observations=len(observations),
                      scored_cases=len(cases_out)),
                  reference=reference, summaries=summaries, observations=observations,
                  unnormalised_sources=skips, invocation_metadata=list(raw_seen.values()),
                  limitations=['Not an exhaustive normalisation of every historical lane schema.',
                               'Historical aggregates are retained separately from scored observations.',
                               'Old POD hyper-reduction Q2 was reported, not gated.',
                               'No new performance or full-population qualification is established.',
                               'No cross-job seconds or mixed-population frontier is constructed.'])
    write(args.out / 'inventory.json', clean(report))
    with (args.out / 'inventory.json.gz').open('wb') as fout:
        with gzip.GzipFile(fileobj=fout, mode='wb', mtime=0) as gz:
            gz.write(json.dumps(clean(report), allow_nan=False, sort_keys=True).encode())
    # Deterministic gzip header, preserving exact per-case traceability without a giant table in prose.
    with (args.out / 'cases.jsonl.gz').open('wb') as fout:
        with gzip.GzipFile(fileobj=fout, mode='wb', mtime=0) as gz:
            for row in cases_out:
                gz.write((json.dumps(row, allow_nan=False, sort_keys=True) + '\n').encode())
    write(args.out / 'SOURCES.json', sorted(provenance.values(), key=lambda x: x['path']))
    print(json.dumps(report['summary_counts'], sort_keys=True))


if __name__ == '__main__':
    main()
