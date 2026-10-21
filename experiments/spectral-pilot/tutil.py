"""spectral-pilot: call-level interleaved timing of a candidate set (DESIGN A1.5, A5). PILOT: small, not a result."""
from __future__ import annotations

import time

import jax


def call(fn, x):
    t = time.perf_counter()
    jax.block_until_ready(fn(x))
    return time.perf_counter() - t


def roundrobin(cands, cases, reps, prune=3.0):
    """cands: list of (key, method, fn). Warm-up (compile) every candidate, then one call each on cases[0]
    (alternating order as given); a candidate is pruned if that single time exceeds `prune` x the best single time of
    its method (A5 heuristic). The kept candidates are then timed call-by-call in rounds: for each repetition, for each
    case, one call of every kept candidate in turn. Returns (times {key: [[t per rep] per case]}, one {key: t},
    pruned [key])."""
    for _, _, fn in cands:
        jax.block_until_ready(fn(cases[0]))
    one = {k: call(fn, cases[0]) for k, _, fn in cands}
    best = {}
    for k, m, _ in cands:
        best[m] = min(best.get(m, float('inf')), one[k])
    kept = [(k, m, fn) for k, m, fn in cands if one[k] <= prune * best[m]]
    pruned = [k for k, m, _ in cands if one[k] > prune * best[m]]
    times = {k: [[] for _ in cases] for k, _, _ in kept}
    for _ in range(reps):
        for ci, x in enumerate(cases):
            for k, _, fn in kept:
                times[k][ci].append(call(fn, x))
    return times, one, pruned
