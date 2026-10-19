"""jcp-robust E5 timing job: one GPU, one job, P-cost then P-cross (DESIGN 4 as amended by A1.11, A1.12, A2.8, A3.8, A4.1).

  0  source check: every copied file (e5/PROVENANCE.json) and the lane p1core.py that P1's p1common imports must hash to
     the recorded sha256, else the job stops before touching the GPU; GPU-type check against the config; record
     (nvidia-smi name, uuid, driver, CUDA, SM/memory clocks, application clocks, power limit/draw, temperature; jax,
     jaxlib, plugin versions; JAX_ENABLE_X64, JAX_DEFAULT_MATMUL_PRECISION, XLA_FLAGS; SLURM job id; host).
  1  P-cost: P1's `p1time.main` UNCHANGED (e5/p1time.py, byte-identical copy), run on a derived config (subset of P1c:
     R128 LSPG-BE dt0 fib4181 frozen and R384 LSPG-CN-R dt0 gauss128 frozen at 256^2 / 1024^2 [+ 4096^2 when the config
     says so, H200 only]; A = R384 gauss128 frozen CN-R 1024^2; P1's doubled-work control = C-E5.1). Output in
     <out>/pcost/ in P1's format (result.json, S_*.npz).
  2  record between panels.
  3  P-cross at 1024^2: c2time's sections 0 (bank, settings, inputs), 1 (scoring runs, identities), warm, 4 (stage-2
     triplets) and the T-dbl control, with the FOM comparators PINNED from fom2d's FROZEN-SELECTION-1024.json (sha256
     checked) by the selection keys '<ROM key>@1024|H|union|joint' (== '|H|PS|joint', timed once) and
     '<ROM key>@1024|H|CC|same_mesh'. No stage 0 / stage 1 (no re-selection). Output in <out>/pcross/ in c2time's format.
  4  record at end; <out>/result.json embeds both panel results; COMPLETE.

    python e5time.py --config configs/e5<attempt>.json --out <dir>
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE / 'e5'
REPO = HERE.parents[1]
# e5/ first: P1's own p1common (the lane's p1common carries jcp-dims edits). p1core is NOT copied: the lane's
# p1core.py is byte-identical to P1's (verified below) and its path logic needs to live in the lane dir.
# cohorts.py (copied into e5/) resolves its two search paths relative to its own location; give the real ones here.
for _i, _p in enumerate((E5, REPO / 'experiments' / 'mr-burgers2d', HERE / 'vendor' / 'quad3d' / 'vendor' / 'burgers3d-span')):
    if str(_p) not in sys.path:
        sys.path.insert(_i, str(_p))
os.environ.setdefault('JCPREF_META', str(E5 / 'e0meta'))

import e5common as EC  # noqa: E402

SRC_OK, SRC_ROWS = EC.verify_sources()
if not SRC_OK:
    print('E5 SOURCE CHECK FAILED', [r for r in SRC_ROWS if not r['ok']], flush=True)
    sys.exit(3)

import numpy as np  # noqa: E402
import p1common as P  # noqa: E402
import p1time as P1T  # noqa: E402
import c2time as C2T  # noqa: E402
import fomrun2 as FR  # noqa: E402
import refstore as RS  # noqa: E402
import cohorts as CH  # noqa: E402
import jax  # noqa: E402

assert Path(P.__file__).resolve().parent == E5, P.__file__
assert Path(P.T2.__file__).resolve() == HERE / 'p1core.py', P.T2.__file__
assert Path(P1T.P.__file__).resolve() == Path(P.__file__).resolve()
assert P1T.BAND == EC.BAND and C2T.BAND == EC.BAND


# ------------------------------------------------------------------ P-cross (c2time sections 0, 1, warm, 4, T-dbl) ----

def pinned_comparators(xc, fz):
    """{rom label: {suffix: FOM key}} from the frozen file; A1.11 / A2.8: union == PS joint (timed once)."""
    N = int(xc['mesh'])
    out = {}
    for r in xc['roms']:
        lab = r['label']
        out[lab] = {}
        for suf in xc['selection_suffixes']:
            g = fz['selection'][f'{lab}@{N}{suf}']
            assert g['status'] == 'selected' and g['selected'], (lab, suf, g.get('status'))
            out[lab][suf] = g['selected']
        ps = fz['selection'][f'{lab}@{N}|H|PS|joint']['selected']
        assert ps == out[lab]['|H|union|joint'], ('A1.11: union joint must equal PS joint', lab, ps)
    return out


def pcross(xc, out, records, rec):
    out.mkdir(parents=True, exist_ok=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    N = int(xc['mesh'])
    bsec = float(xc.get('burn', 0.05))
    reps = int(xc.get('reps', 3))
    fzb = (HERE / xc['frozen_selection']['path']).read_bytes()
    assert EC.sha_bytes(fzb) == xc['frozen_selection']['sha256'], 'FROZEN-SELECTION sha256 differs from the config'
    fz = json.loads(fzb)
    assert int(fz['mesh']) == N
    pinned = pinned_comparators(xc, fz)
    assert pinned == xc['expected_comparators'], (pinned, xc['expected_comparators'])
    fkeys = sorted({k for v in pinned.values() for k in v.values()})
    sel_used = {f'{lab}@{N}{suf}': fz['selection'][f'{lab}@{N}{suf}'] for lab in pinned for suf in xc['selection_suffixes']}
    rep = dict(config=xc, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=P.gpu_info(),
               jax_version=jax.__version__, store_version=RS.store_version(), checkpoint_sha256=P.QS.sha_file(P.T2.CKPT),
               mesh=N, subjects={}, prep={}, stage0={}, stage1=[], selection=sel_used, pinned=pinned,
               frozen_selection_sha256=EC.sha_bytes(fzb), stage2=[], controls={}, memory={}, gates={}, complete=False,
               e5_note='E5: comparators pinned from fom2d FROZEN-SELECTION-1024.json (H200-selected); no stage 0 / '
                       'stage 1, no re-selection; stage-2 triplets and identities as c2time')
    save = lambda: (out / 'result.json').write_text(json.dumps(C2T.clean(rep), indent=1) + '\n')
    co = {c: CH.Cohort(2, c) for c in ('dev6', 'val32')}
    cases = [(c, int(j), co[c].D[int(j)]) for c, j in xc['cases']]
    cid = lambda c: f'{c[0]}|{c[1]}'

    # ---- 0. bank, ROM settings, inputs (c2time section 0) ----
    bank = P.Bank(N, xc['edges'], nrb=xc.get('bank_blocks'))
    rep['memory']['bank'] = dict(P.mem(), bank_bytes=bank.bytes, build_seconds=bank.build_seconds)
    print('XBANK', N, el(), P.mem(), flush=True)
    settings = {}
    forms_needed = {}
    for r_ in [xc['A']] + xc['roms']:
        forms_needed.setdefault((int(r_['Rp']), r_['rule'], r_.get('policy', 'frozen')), set()).add(r_.get('form', 'LSPG'))

    def setting(sp):
        k = (int(sp['Rp']), sp['rule'], sp.get('policy', 'frozen'))
        if k not in settings:
            fs = tuple(['LSPG'] + sorted(f for f in forms_needed[k] if f != 'LSPG'))
            settings[k] = P.Setting(bank, f'{k}', k[0], k[1], k[2], forms=fs, nrec=0)
        return settings[k]

    roms = {xc['A']['label']: xc['A']}
    for r in xc['roms']:
        assert r['label'] not in roms, r['label']
        roms[r['label']] = r
    U0 = {}
    for c in cases:
        u0n = P.Q.e.initial(N, c[2])
        U0[cid(c)] = (jax.numpy.asarray(u0n), float(c[2][4]))
    foms = FR.FOMs()
    XS = {}

    def xkey(k):
        sp = FR.parse(k)
        return ('fd', sp[2]) if sp[0] == 'fd' else ('ps', sp[1])

    def xget(k, c):
        kk = (xkey(k), cid(c))
        if kk not in XS:
            XS[kk] = FR.prepare(FR.parse(k), c[2])
        return XS[kk][0]
    for k in fkeys:
        for c in cases:
            xget(k, c)
        tp = [XS[(xkey(k), cid(c))][1] for c in cases]
        rep['prep'][k] = dict(median=float(np.median(tp)), max=float(np.max(tp)), shared_key=list(xkey(k)))
    rep['memory']['inputs'] = dict(P.mem(), n_inputs=len(XS))

    def rom_call(lb, c):
        sp = roms[lb]
        st = setting(sp)
        st.check('main', bank)
        u0, nu = U0[cid(c)]
        form = sp.get('form', 'LSPG')
        assert int(round(.25 / float(sp['dt']))) <= st.nstep, ('per-step evidence buffer too short', sp['dt'])
        q = st.q0[form]
        d = st.data[('main', form)]
        sch = st.sched(form, sp['scheme'], float(sp['dt']), sp.get('level', 'prod'))
        cold = st.cold
        return lambda: q['fused'](u0, nu, d, cold, sch)

    def rom_ident(o):
        s_ = o['stats']
        return P.sha(np.concatenate([np.asarray(o['W']).ravel(), np.asarray(o['w0']).ravel(),
                                     np.asarray(s_['step_reason'], float), np.asarray(s_['step_it'], float),
                                     np.asarray(P._reduce(o['fields']))]))

    def fom_call(k, c):
        return foms.callable(FR.parse(k), xget(k, c), float(c[2][4]), ('mesh', N))

    def fom_ident(F):
        return P.sha(np.concatenate([np.asarray(F[:, ::N // 256, ::N // 256]).ravel(), np.asarray(P._reduce(F))]))

    call = lambda lb, c: rom_call(lb, c) if lb in roms else fom_call(lb, c)
    ident = lambda lb, o: rom_ident(o) if lb in roms else fom_ident(o)
    fields = lambda lb, o: o['fields'] if lb in roms else o

    # ---- 1. scoring runs, identities (c2time section 1; target_e / ladder_e known for dev6 only) ----
    REF = {}
    for lb in list(roms) + fkeys:
        rows = []
        for c in cases:
            t, o = C2T.sync_time(call(lb, c))
            REF[(lb, cid(c))] = ident(lb, o)
            F = np.asarray(fields(lb, o)[:, ::N // 256, ::N // 256])
            s = RS.score(F, 2, c[0], c[1], 'ST')
            want = (roms[lb].get('target_e') or {}).get(cid(c)) if lb in roms else xc['foms'][lb]['ladder_e'].get(cid(c))
            row = dict(case=cid(c), e=s['e'], eligible=s['eligible'], first_seconds=t, want=want,
                       repro=None if want is None else bool(abs(s['e'] - want) <= C2T.TOL_REPRO),
                       finite=bool(np.isfinite(F).all()))
            if lb in roms:
                row['valid'] = P.run_valid(o)
            rows.append(row)
        rep['subjects'][lb] = dict(kind='rom' if lb in roms else 'fom', spec=roms.get(lb) or dict(key=lb), scoring=rows,
                                   repro_ok=all(r['repro'] is not False for r in rows),
                                   repro_checked=sum(r['repro'] is not None for r in rows))
        print('SCORED', lb, rep['subjects'][lb]['repro_ok'], el(), flush=True)
    ndev = sum(1 for c in cases if c[0] == 'dev6')
    rep['gates']['G-RR'] = all(rep['subjects'][lb]['repro_ok'] and rep['subjects'][lb]['repro_checked'] == ndev
                               and all(r_.get('valid') is True for r_ in rep['subjects'][lb]['scoring'])
                               for lb in roms if lb != xc['A']['label'])
    rep['gates']['A-valid'] = all(r_.get('valid') is True for r_ in rep['subjects'][xc['A']['label']]['scoring'])
    rep['gates']['G-FR'] = all(rep['subjects'][k]['repro_ok'] and rep['subjects'][k]['repro_checked'] == ndev for k in fkeys)
    rep['gates']['scope'] = 'repro checked on the dev6 cases (fom2d target_e / ladder_e exist for dev6 only)'
    rep['memory']['scoring'] = P.mem()
    save()

    VETO = set()

    def check(lb, c, o):
        ok = ident(lb, o) == REF[(lb, cid(c))]
        if not ok:
            VETO.add(lb)
        return ok

    A = xc['A']['label']

    def timed_A(c):
        fn = rom_call(A, c)
        C2T.burn(bsec)
        t, o = C2T.sync_time(fn)
        return t, check(A, c, o)

    def triplet(lb, c, r_, stage):
        fnB = call(lb, c)

        def attempt(_a):
            ta1, ha1 = timed_A(c)
            C2T.burn(bsec)
            tb, ob = C2T.sync_time(fnB)
            hb = check(lb, c, ob)
            ta2, ha2 = timed_A(c)
            return dict(B=lb, case=cid(c), rep=r_, stage=stage, tA1=ta1, tA2=ta2, tB=tb, hashes_ok=bool(ha1 and ha2 and hb))
        return EC.aba_triplet(attempt)

    for lb in list(roms) + fkeys:                     # warm every subject on every case once (first-call effects)
        for c in cases:
            o = jax.block_until_ready(call(lb, c)())
            check(lb, c, o)
            del o
    records.append(rec('pcross-after-warm'))

    # ---- 4. stage 2 (fresh triplets, randomised) ----
    rng = np.random.default_rng(int(xc.get('seed', 1)))
    s2 = [lb for lb in roms if lb != A] + fkeys
    order = [(lb, ci, r_) for lb in s2 for ci in range(len(cases)) for r_ in range(reps)]
    rng.shuffle(order)
    for i, (lb, ci, r_) in enumerate(order):
        rep['stage2'].append(triplet(lb, cases[ci], r_, 2))
        if (i + 1) % 24 == 0:
            save()
            print('STAGE2', i + 1, len(order), el(), flush=True)
    rep['memory']['stage2'] = P.mem()
    save()

    # ---- 5. T-dbl (c2time control; descriptive in E5, C-E5.1 is P1's doubled-work control in P-cost) ----
    dbl = []
    for c in [x for x in cases if [x[0], x[1]] in [list(y) for y in xc.get('controls', {}).get('doubled_cases', [])]]:
        fnA = rom_call(A, c)
        for r_ in range(reps):
            C2T.burn(bsec)
            t1, o1 = C2T.sync_time(fnA)
            C2T.burn(bsec)
            t0 = time.perf_counter()
            oa = jax.block_until_ready(fnA())
            ob = jax.block_until_ready(fnA())
            tb = time.perf_counter() - t0
            C2T.burn(bsec)
            t2, o2 = C2T.sync_time(fnA)
            dbl.append(dict(case=cid(c), rep=r_, t1=t1, t_double=tb, t2=t2, ratio=tb / (.5 * (t1 + t2)), drift=t2 / t1,
                            identity_ok=all(check(A, c, o_) for o_ in (o1, oa, ob, o2))))
    med = float(np.median([d['ratio'] for d in dbl])) if dbl else float('nan')
    rep['controls']['T-dbl'] = dict(rows=dbl, median=med, fires=bool(1.8 <= med <= 2.2), role='descriptive in E5')
    rep['compile_log'] = foms.compile_log
    rep['compile_events'] = foms.compile_events
    rep['veto'] = sorted(VETO)
    rep['memory']['end'] = P.mem()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('PCROSS COMPLETE', el(), flush=True)
    del bank, settings, XS, foms
    return rep


# ------------------------------------------------------------------ job ----

def pcost_config(cfg):
    pc = dict(cfg['pcost'], attempt=f"{cfg['attempt']}-pcost", driver='p1time')
    if cfg.get('allow_cpu_smoke'):
        pc['allow_cpu_smoke'] = True
    return pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cb = Path(a.config).read_bytes()
    cfg = json.loads(cb)
    assert cfg['driver'] == 'e5time', cfg.get('driver')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    P.check_env(cfg)
    if '4096' in cfg['pcost']['meshes']:
        assert cfg['gpu_type'] == 'H200' and cfg.get('allow_4096') is True, 'A1.11: 4096^2 only on H200'
    if cfg.get('jcpref_store'):
        os.environ.setdefault('JCPREF_STORE', cfg['jcpref_store'])
    begin = time.perf_counter()
    # executing device (job audit E5 item 3): JAX's device ordinal -> CUDA driver UUID / PCI bus id / CUDA_VISIBLE_DEVICES
    # -> exactly one nvidia-smi row; that UUID is used for the type check and for every telemetry record
    dev = jax.devices()[0]
    ordinal = getattr(dev, 'local_hardware_id', None)
    ordinal = dev.id if ordinal is None else ordinal
    rows, smi_err = EC.smi_rows()
    device = EC.resolve_device(rows, ordinal, os.environ.get('CUDA_VISIBLE_DEVICES'),
                               EC.cuda_uuid(ordinal) if jax.default_backend() == 'gpu' else None,
                               getattr(dev, 'pci_bus_id', None), os.environ.get('CUDA_DEVICE_ORDER'))
    device.update(smi_error=smi_err, jax_device=str(dev), jax_device_kind=dev.device_kind)
    uuid = device.get('uuid')
    rec = lambda tag: EC.env_record(tag, uuid)
    records = [rec('start')]
    tprob = EC.telemetry_problems(records[0], uuid)
    gok, gname = EC.gpu_type_ok(cfg['gpu_type'], device.get('row'))
    top = dict(attempt=cfg['attempt'], config=cfg, config_sha256=hashlib.sha256(cb).hexdigest(), gpu_type=cfg['gpu_type'],
               device=device, device_uuid=uuid, telemetry_start_problems=tprob,
               gpu_type_ok=gok, gpu_name=gname, backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
               jax_version=jax.__version__, x64=bool(jax.config.jax_enable_x64), source_check=SRC_ROWS,
               commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'), records=records,
               pcost=None, pcross=None, complete=False)
    save = lambda: (out / 'result.json').write_text(json.dumps(C2T.clean(top), indent=1) + '\n')
    save()
    if not cfg.get('allow_cpu_smoke') and not (device['resolved'] and gok and not tprob):
        print('E5 DEVICE / TYPE / TELEMETRY CHECK FAILED', cfg['gpu_type'], gname, device.get('reason'), tprob, flush=True)
        sys.exit(4)

    # ---- P-cost: P1's p1time.main, unchanged ----
    pdir = out / 'pcost'
    pdir.mkdir(exist_ok=True)
    (pdir / 'config.json').write_text(json.dumps(pcost_config(cfg), indent=1) + '\n')
    argv0 = sys.argv
    sys.argv = ['p1time.py', '--config', str(pdir / 'config.json'), '--out', str(pdir)]
    try:
        P1T.main()
    finally:
        sys.argv = argv0
    top['pcost'] = json.loads((pdir / 'result.json').read_text())
    gc.collect()
    records.append(rec('between-panels'))
    save()
    print('E5 PCOST DONE', round(time.perf_counter() - begin, 1), flush=True)

    # ---- P-cross ----
    top['pcross'] = pcross(cfg["pcross"], out / "pcross", records, rec)
    records.append(rec('end'))
    top['telemetry_end_problems'] = EC.telemetry_problems(records[-1], uuid)
    top['elapsed_seconds'] = round(time.perf_counter() - begin, 1)
    top['complete'] = bool(top['pcost'].get('complete') and top['pcross'].get('complete'))
    save()
    if top['complete']:
        (out / 'COMPLETE').write_text('complete\n')
    print('E5TIME COMPLETE', top['complete'], top['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
