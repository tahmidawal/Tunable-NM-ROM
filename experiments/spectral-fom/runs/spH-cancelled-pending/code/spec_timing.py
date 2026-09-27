"""spectral-fom: the shared A-B-A timing harness (one allocation per mesh).

Design (fixed before any cluster result; DESIGN.md section T):
  phase romA1  every ROM subject, `repetitions` reps x cases, randomised order within each case
  phase spec   every spectral-FOM subject, same reps x cases, randomised
  phase romA2  identical to romA1
Between phases: device sync, `cooldown_seconds` sleep, a fixed `phase_dummy_seconds` 384^2 matmul kernel.
Before every invocation: `burn_seconds` of the same dummy kernel (GPU clocks up), gc.collect(), the GPU
UUID guard; the Python garbage collector is disabled around the timed call.

Gates (limit 1.10):
  drift      per ROM subject, median(romA2) / median(romA1) in [1/1.10, 1.10]
  neighbour  per subject and phase, median(after a long predecessor) / median(after a short one) <= 1.10
             (predecessor = the invocation immediately before, same phase).  Two subjects in the phase:
             long/short = the slower/faster of {the other subject, itself}.  Three or more: long = the top third
             of the OTHER subjects by phase median, short = the bottom third.  >= 3 samples on each side.
Timing statistic: ROM = median over romA1 u romA2; spectral = median over its phase.
GPU time = `fused_device_seconds`: from after the synchronised host->device input copy to
block_until_ready of the outputs (the paper's GPU-query scope); `total_seconds` adds both copies.
"""
from __future__ import annotations

import ctypes
import gc
import time
import uuid

import numpy as np
import jax
import jax.numpy as jnp


def gpu_uuid():
    cuda = ctypes.CDLL('libcuda.so.1')
    assert cuda.cuInit(0) == 0
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    raw = (ctypes.c_ubyte * 16)()
    fn = getattr(cuda, 'cuDeviceGetUuid_v2', cuda.cuDeviceGetUuid)
    assert fn(ctypes.byref(raw), dev) == 0
    return 'GPU-' + str(uuid.UUID(bytes=bytes(raw)))


_BURN = {}


def burn(seconds):
    if 'fn' not in _BURN:
        _BURN['a'] = jnp.ones((384, 384), dtype=jnp.float64) * 0.001
        _BURN['fn'] = jax.jit(lambda x: x @ x + 0.0001)
    a, fn = _BURN['a'], _BURN['fn']
    fn(a).block_until_ready()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        fn(a).block_until_ready()


def query(fn, host_input):
    """host array in -> host f64 output(s) out.  fn(device_input) -> array or tuple (first = field)."""
    start = time.perf_counter()
    x = jax.device_put(host_input)
    x.block_until_ready()
    t0 = time.perf_counter()
    out = fn(x)
    jax.block_until_ready(out)
    t1 = time.perf_counter()
    out = jax.device_get(out)
    end = time.perf_counter()
    field = out[0] if isinstance(out, tuple) else out
    return np.asarray(field, dtype=np.float64), dict(
        fused_device_seconds=t1 - t0, total_seconds=end - start,
        input_seconds=t0 - start, output_seconds=end - t1), out


class ABA:
    def __init__(self, cfg, uuid0, record, log=print, save=lambda: None):
        self.cfg, self.uuid0, self.record, self.log, self.save = cfg, uuid0, record, log, save
        self.invocations, self.breaks = [], []
        self.begin = time.perf_counter()

    def phase_break(self, label):
        jax.block_until_ready(jnp.zeros(()))
        time.sleep(self.cfg['cooldown_seconds'])
        burn(self.cfg['phase_dummy_seconds'])
        self.breaks.append(dict(before=label, at_seconds=time.perf_counter() - self.begin))

    def run_phase(self, label, subjects, cases, reps, rng):
        prev = None
        for rep in range(reps):
            for case in cases:
                for i in rng.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    gc.collect()
                    burn(self.cfg['burn_seconds'])
                    assert gpu_uuid() == self.uuid0
                    gc.disable()
                    try:
                        field, row, raw = sub['call'](case)
                    finally:
                        gc.enable()
                    inv = self.record(sub, case, rep, label, field, row, raw)
                    inv.update(name=sub['name'], role=sub['role'], case=int(case), rep=rep, phase=label,
                               previous=prev,
                               fused_device_seconds=row['fused_device_seconds'],
                               total_seconds=row['total_seconds'])
                    self.invocations.append(inv)
                    prev = sub['name']
            self.log(f'PHASE {label} rep {rep} t={time.perf_counter() - self.begin:.1f}s')
            self.save()

    def run(self, rom, spec, cases, reps, seed):
        rng = np.random.default_rng(seed)
        self.phase_break('romA1')
        self.run_phase('romA1', rom, cases, reps, rng)
        self.phase_break('spec')
        self.run_phase('spec', spec, cases, reps, rng)
        self.phase_break('romA2')
        self.run_phase('romA2', rom, cases, reps, rng)
        return self.gates(rom, spec)

    def med(self, name, phases):
        v = [x['fused_device_seconds'] for x in self.invocations if x['name'] == name and x['phase'] in phases]
        return float(np.median(v)), len(v)

    def gates(self, rom, spec):
        out = self._gates(rom, spec, normalise=False)
        out['neighbour_case_normalised'] = self._gates(rom, spec, normalise=True)['neighbour']
        return out

    def _gates(self, rom, spec, normalise):
        """normalise=True (DESIGN amendment T1): every time is divided by the median of the same subject on the same
        case in the same phase before the neighbour medians are taken, so a case-dependent cost (iteration counts
        differ by case) cannot masquerade as an order effect."""
        lim = self.cfg['neighbour_limit']
        drift = []
        for s in rom:
            a1, _ = self.med(s['name'], ('romA1',))
            a2, _ = self.med(s['name'], ('romA2',))
            drift.append(dict(name=s['name'], romA1_median=a1, romA2_median=a2, ratio=a2 / a1))
        drift_ok = all(1 / lim <= r['ratio'] <= lim for r in drift)
        rows = []
        for label, subs in (('romA1', rom), ('romA2', rom), ('spec', spec)):
            if len(subs) < 2:
                continue
            inv = [x for x in self.invocations if x['phase'] == label]
            meds = {s['name']: self.med(s['name'], (label,))[0] for s in subs}
            cmed = {}
            if normalise:
                for x in inv:
                    cmed.setdefault((x['name'], x['case']), []).append(x['fused_device_seconds'])
                cmed = {k: float(np.median(v)) for k, v in cmed.items()}
            tval = (lambda x: x['fused_device_seconds'] / cmed[(x['name'], x['case'])]) if normalise else (lambda x: x['fused_device_seconds'])
            for s in subs:
                others = {k: v for k, v in meds.items() if k != s['name']}
                mine = [x for x in inv if x['name'] == s['name']]
                if len(others) == 1:
                    # two subjects: "after the other" vs "after itself", oriented slow over fast
                    (other,) = others
                    if meds[other] >= meds[s['name']]:
                        slow, fast = [other], [s['name']]
                    else:
                        slow, fast = [s['name']], [other]
                else:
                    # >= 3 subjects: predecessors among the other subjects, top third vs bottom third by median
                    order_ = sorted(others, key=others.get)
                    k3 = max(1, len(order_) // 3)
                    fast, slow = order_[:k3], order_[-k3:]
                hi = [tval(x) for x in mine if x['previous'] in slow]
                lo = [tval(x) for x in mine if x['previous'] in fast]
                slow, lo_name = ','.join(slow), ','.join(fast)
                if len(hi) >= 3 and len(lo) >= 3:
                    rows.append(dict(name=s['name'], phase=label, long_predecessor=slow, short_predecessor=lo_name,
                                     after_long_median=float(np.median(hi)), after_short_median=float(np.median(lo)),
                                     n_long=len(hi), n_short=len(lo),
                                     ratio=float(np.median(hi) / np.median(lo))))
                else:
                    rows.append(dict(name=s['name'], phase=label, insufficient=True, n_long=len(hi), n_short=len(lo)))
        nb_ok = all(r.get('ratio', 0.0) <= lim for r in rows) and not any(r.get('insufficient') for r in rows)
        return dict(drift=dict(rows=drift, limit=lim, passed=bool(drift_ok)),
                    neighbour=dict(rows=rows, limit=lim, passed=bool(nb_ok)))

    def timings(self, rom, spec):
        out = {}
        for s in rom:
            m, k = self.med(s['name'], ('romA1', 'romA2'))
            a1, _ = self.med(s['name'], ('romA1',))
            a2, _ = self.med(s['name'], ('romA2',))
            tot = [x['total_seconds'] for x in self.invocations if x['name'] == s['name']]
            out[s['name']] = dict(role=s['role'], median_ms=m * 1e3, romA1_ms=a1 * 1e3, romA2_ms=a2 * 1e3,
                                  samples=k, total_median_ms=float(np.median(tot)) * 1e3)
        for s in spec:
            m, k = self.med(s['name'], ('spec',))
            tot = [x['total_seconds'] for x in self.invocations if x['name'] == s['name']]
            out[s['name']] = dict(role=s['role'], median_ms=m * 1e3, samples=k,
                                  total_median_ms=float(np.median(tot)) * 1e3)
        return out
