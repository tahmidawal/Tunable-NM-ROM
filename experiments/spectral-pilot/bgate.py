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


def allowance(ref):
    o = ref['order']['lat64']
    g = order_gate(o['d128_256'], o['d256_512'], o['d512_1024'])
    tg = ref['time_gate']
    dts = [tg['d1024_4cases'], tg['d512_12cases']]
    t_ok = all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in dts)
    delta_t = max(dts) if t_ok else None
    if g['gate'] and t_ok:
        U, mode = g['e_ref'] + delta_t, 'conditional physical allowance'
    elif t_ok:
        U, mode = delta_t, 'numerical reference only (order gate failed); physical accuracy unqualified'
    else:
        U, mode = None, 'unavailable (time gate missing or non-finite)'
    # SW2 for B: a difference set whose finest difference is inflated 5x must fail the order gate
    sw2 = not order_gate(o['d128_256'], o['d256_512'], 5 * o['d512_1024'])['gate']
    return dict(U=U, mode=mode, order=g, delta_t=delta_t, SW2_fired=sw2)
