"""spectral-pilot: call-level interleaved timing of a candidate set (DESIGN A1.5, A5). PILOT: small, not a result."""
from __future__ import annotations

import time

import jax


def call(fn, x):
    t = time.perf_counter()
    jax.block_until_ready(fn(x))
    return time.perf_counter() - t


def _rounds(cands, cases, reps):
    times = {k: [[] for _ in cases] for k, _, _ in cands}
    for _ in range(reps):
        for ci, x in enumerate(cases):
            for k, _, fn in cands:
                times[k][ci].append(call(fn, x))
    return times


def _med(ts):
    s = sorted(sorted(x)[len(x) // 2] for x in ts)
    return s[len(s) // 2]


def roundrobin(cands, cases, reps, prune=3.0):
    """cands: list of (key, method, fn). (1) warm-up (compile) every candidate and time one call each on cases[0];
    (2) fully time, call-level interleaved, the best-single-call candidate of each method; (3) prune a candidate only
    if its single call exceeds `prune` x its method's best FULLY-timed median (A5); (4) fully time every kept
    candidate in call-level interleaved rounds (fresh times for all, including those of step 2).
    Returns (times {key: [[t per rep] per case]}, one {key: t}, pruned [key])."""
    for _, _, fn in cands:
        jax.block_until_ready(fn(cases[0]))
    one = {k: call(fn, cases[0]) for k, _, fn in cands}
    leaders = {}
    for c in cands:
        if c[1] not in leaders or one[c[0]] < one[leaders[c[1]][0]]:
            leaders[c[1]] = c
    lt = _rounds(list(leaders.values()), cases, reps)
    full_best = {m: _med(lt[c[0]]) for m, c in leaders.items()}
    lead_keys = {c[0] for c in leaders.values()}                  # fully timed leaders are always kept (audit 4)
    kept = [c for c in cands if c[0] in lead_keys or one[c[0]] <= prune * full_best[c[1]]]
    pruned = [c[0] for c in cands if c not in kept]
    times = _rounds(kept, cases, reps)
    times_leaders = {str(k): v for k, v in lt.items()}
    roundrobin.last = dict(leader_times=times_leaders, full_best=full_best)
    return times, one, pruned
