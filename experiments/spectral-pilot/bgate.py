"""spectral-pilot Part B reference gate and allowance, recomputed from the stored differences (DESIGN A2.5, A3.4, A4),
with the SW2 check for B (the gate must reject a perturbed difference set). PILOT: small, not a result."""
from __future__ import annotations

import math


def order_gate(d1, d2, d3):
    """d1 = |u128-u256|, d2 = |u256-u512|, d3 = |u512-u1024| (max over cases/times, i/64 lattice)."""
    if not all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (d1, d2, d3)):
        return dict(gate=False, reason='non-finite or non-positive differences')
    if not d1 > d2 > d3:
        return dict(gate=False, reason='differences not strictly decreasing')
    p1, p2 = math.log2(d1 / d2), math.log2(d2 / d3)
    ok = p2 >= 1 and abs(p1 - p2) <= 0.5
    return dict(gate=ok, p1=p1, p2=p2, e_ref=d3 / (2 ** p2 - 1) if ok else None)


def _fin(x, pos=False):
    return isinstance(x, (int, float)) and math.isfinite(x) and (x > 0 if pos else x >= 0)


def allowance(ref):
    o = (ref.get('order') or {}).get('lat64') or {}
    ds = [o.get('d128_256'), o.get('d256_512'), o.get('d512_1024')]
    tg = ref.get('time_gate') or {}
    dts = [tg.get('d1024_4cases'), tg.get('d512_12cases')]
    t_ok = all(_fin(x) for x in dts)
    sw2 = sw2_fixture()
    if not all(_fin(x, pos=True) for x in ds) or not t_ok:
        return dict(U=None, mode='unavailable (space or time evidence missing or non-finite)', order=None,
                    delta_t=None, SW2_fired=sw2)
    g = order_gate(*ds)
    delta_t = max(dts) if t_ok else None
    if g['gate'] and t_ok:
        U, mode = g['e_ref'] + delta_t, 'conditional physical allowance'
    elif t_ok:
        U, mode = delta_t, 'numerical reference only (order gate failed); physical accuracy unqualified'
    else:
        U, mode = None, 'unavailable'
    return dict(U=U, mode=mode, order=g, delta_t=delta_t, SW2_fired=sw2)


def sw2_fixture():
    """SW2 for B on fixed fixtures: equal differences (no convergence) and a mismatched-order set must be rejected;
    a clean second-order set must be accepted."""
    return bool((not order_gate(1e-3, 1e-3, 1e-3)['gate']) and (not order_gate(4e-3, 1e-3, 5e-4)['gate'])
                and order_gate(4e-3, 1e-3, 2.5e-4)['gate'])
