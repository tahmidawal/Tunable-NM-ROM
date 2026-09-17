"""Draw-to-draw bookkeeping shared by the report generator, the rule export, the self-audit and
the lab entry (DESIGN §A2, §A4). One place computes, from the audit JSONs alone:

- every *draw* of every rule *construction*: a construction is (q, m target, fit-state count,
  row scaling); a draw is one untruncated fitted rule of it, from any job (the `qrg304`
  archived reachable rules count as draws of the incumbent `std` construction on the 8192
  pool; this lane's ordinary arms share one pool per rung and draw their own fit-state subset;
  the `bet301` replication arms redraw both pool and subset from a dedicated stream);
- how many draws certify on each bar, the min / median / max of rho_max over the draws;
- the status label §A2 assigns to the construction at that m:
    `confirmed (k/k)`            every draw certifies and there are at least two draws
    `marginal at m=... (k/n)`    some draws certify and some do not
    `certified in one draw`      a single certified draw, construction not replicated
    `not certified (0/n)`        no draw certifies
- the exported rule per rung under the §A4 policy (see `export_choice`).

Nothing here reads a result.json or a rule file; only the audits.
"""
from __future__ import annotations

import numpy as np

ARM_ORDER = {'std': 0, 'fs64': 1, 'rhow64': 2}


def construction_of(rule):
    """(q, m_target, fit_states, scaling) with the qrg304 `reachable` arm folded into `std`."""
    return (rule['q'], rule['m_target'], rule['fit_states'], rule.get('scaling') or 'row')


def arm_of(rule):
    """The construction arm: qrg304 `reachable` and the `reprowincumbent…` replication arms are
    the incumbent `std` construction; `reprow64…` replication arms are `fs64`."""
    a = rule['arm']
    if a == 'reachable' or a.startswith('reprowincumbent'):
        return 'std'
    if a.startswith('reprow64'):
        return 'fs64'
    return a


def collect_draws(jobs):
    """Every untruncated reachable-population rule from every audit, one record per draw.

    `jobs` is a list of audit dicts (in priority order; the archived qrg304 rules are taken
    from the first audit that carries them)."""
    draws, seen = [], set()
    for j in jobs:
        for x in j['rules']:
            if x['population'] != 'reachable' or x['truncated']:
                continue
            if x['source'] == 'qrg304':
                k = ('qrg304', x['q'], x['arm'], x['m_target'])
                if k in seen:
                    continue
                seen.add(k)
                label = f"qrg304 {x['arm']} (pool 8192)"
            elif x.get('seed_offset') is not None:
                label = f"{j['attempt']} draw {x['seed_offset']}"
            else:
                label = f"{j['attempt']} {x['arm']}"
            draws.append(dict(
                construction=construction_of(x), q=x['q'], m_target=x['m_target'], m=x['m'],
                fit_states=x['fit_states'], scaling=x.get('scaling') or 'row', arm=arm_of(x),
                source_arm=x['arm'], source=x['source'], attempt=j['attempt'], job_id=j['job_id'],
                commit=j['commit'], pool=x['candidates'], replication=x.get('seed_offset') is not None,
                seed_offset=x.get('seed_offset'), draw_seed=x.get('draw_seed'), label=label,
                rho_max=x['rho_max'], rho_p95=x['rho_p95'], rho_median=x['rho_median'],
                relative_fit=x['relative_fit'], certified_primary=bool(x['certified_primary']),
                certified_tight=bool(x['certified_tight']),
                certified_secondary=bool(x['certified_secondary']), rule=x))
    return draws


def status_label(n, k, m):
    if n == 0:
        return 'no draw'
    if k == n and n >= 2:
        return f'confirmed ({k}/{n})'
    if k == n == 1:
        return 'certified in one draw'
    if k == 0:
        return f'not certified (0/{n})'
    return f'marginal at m={m} ({k}/{n})'


def constructions(draws, bar='primary'):
    """One record per construction with its draws, counts, spread and status."""
    groups = {}
    for d in draws:
        groups.setdefault(d['construction'], []).append(d)
    out = []
    for key, ds in sorted(groups.items()):
        q, m_target, fs, scaling = key
        v = np.array([d['rho_max'] for d in ds], float)
        k = sum(d[f'certified_{bar}'] for d in ds)
        kt = sum(d['certified_tight'] for d in ds)
        n_rep = sum(d['replication'] for d in ds)
        k_rep = sum(d['certified_primary'] for d in ds if d['replication'])
        out.append(dict(
            construction=key, q=q, m_target=m_target, fit_states=fs, scaling=scaling,
            arm=ds[0]['arm'] if len({d['arm'] for d in ds}) == 1 else '/'.join(sorted({d['arm'] for d in ds})),
            draws=ds, n=len(ds), certified_primary_count=int(k), certified_tight_count=int(kt),
            replication_draws=int(n_rep), replication_certified_primary=int(k_rep),
            rho_min=float(v.min()), rho_median=float(np.median(v)), rho_max_of_draws=float(v.max()),
            rho_mean=float(v.mean()), rho_std=(float(v.std(ddof=1)) if len(v) > 1 else None),
            spread_ratio=float(v.max() / v.min()),
            status=status_label(len(ds), int(k), m_target),
            status_tight=status_label(len(ds), int(kt), m_target),
            confirmed=bool(k == len(ds) and len(ds) >= 2),
            marginal=bool(0 < k < len(ds))))
    return out


def status_of_rule(cons, rule):
    """The construction status of one rule (audit record) — what its certified flag is worth."""
    key = construction_of(rule)
    c = next((c for c in cons if c['construction'] == key), None)
    return c['status'] if c else 'no draw'


def sort_key(d):
    """Cheapest first: m, then this lane's pool-16384 draw before qrg304's, then arm order."""
    return (d['m'], 0 if d['source'] != 'qrg304' else 1, ARM_ORDER.get(d['arm'], 9))


def export_choice(cons, draws, q, ladder_choice=None):
    """The §A4 export policy for one rung.

    (i) the cheapest (smallest m) *confirmed* construction (>= 2 draws, all primary-certified);
        the concrete rule is the one the timed ladder ran if it belongs to that construction,
        else this lane's ordinary (non-replication, pool-16384) draw, else the cheapest draw;
    (ii) if no construction at the rung is confirmed: the cheapest primary-certified single-draw
         rule whose m exceeds every m at which a construction of the rung was found marginal,
         labelled `certified in one draw`; ties by arm order.
    Returns (draw, basis, note)."""
    cq = [c for c in cons if c['q'] == q]
    confirmed = sorted([c for c in cq if c['confirmed']], key=lambda c: c['m_target'])
    if confirmed:
        c = confirmed[0]
        ds = sorted(c['draws'], key=sort_key)
        pick = None
        if ladder_choice is not None:
            pick = next((d for d in ds if d['source_arm'] == ladder_choice['arm']
                         and d['m_target'] == ladder_choice['m_target']
                         and abs(d['rho_max'] - ladder_choice['rho_max']) <= 1e-9), None)
        if pick is None:
            pick = next((d for d in ds if not d['replication'] and d['source'] != 'qrg304'), None)
        if pick is None:
            pick = ds[0]
        return pick, 'confirmed', (f"cheapest confirmed construction: {c['arm']} m={c['m_target']}, "
                                   f"{c['fit_states']} states, {c['certified_primary_count']}/{c['n']} draws certify")
    marginal_m = [c['m_target'] for c in cq if c['marginal']]
    floor = max(marginal_m) if marginal_m else 0
    cands = sorted([d for d in draws if d['q'] == q and d['certified_primary'] and d['m'] > floor
                    and not d['replication']], key=sort_key)
    if not cands:
        return None, 'none', f"no confirmed construction and no certified rule above m={floor}"
    d = cands[0]
    c = next(c for c in cq if c['construction'] == d['construction'])
    return d, 'single-draw', (f"no confirmed construction at this rung; constructions marginal at m="
                              f"{sorted(set(marginal_m))}; cheapest certified rule above that m, "
                              f"status '{c['status']}'")
