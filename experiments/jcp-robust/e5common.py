"""jcp-robust E5 shared, JAX-free helpers (DESIGN 4, A1.11, A1.12, A2.8, A3.7-A3.9, A4.1).

  * the drift band and the A-B-A retry loop used by the P-cross driver (c2time's stage-2 rule, factored so the CPU unit
    test C-E5.3 runs the very function the driver runs); the P-cost panel runs P1's p1time unchanged, whose inline band
    is checked against this one by source inspection (`p1time_band_is_ours`);
  * hardware / software records (nvidia-smi clocks, power, temperature; jax, jaxlib and plugin versions; env flags);
  * source verification against e5/PROVENANCE.json (every copied file, plus the lane p1core.py that P1's p1common
    imports, must still hash to the recorded sha256 before a job does anything).

No JAX import here: this module is used by e5analyze.py and e5_tests.py on CPU.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as IM
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E5 = HERE / 'e5'
BAND = (1 / 1.1, 1.1)          # p1time.BAND and c2time.BAND (identical literal in both sources)
MAX_TRIES = 3                  # one attempt + <= 2 retries (p1time and c2time)


def drift_ok(drift):
    """The drift band of p1time / c2time: t_A2 / t_A1 inside [1/1.1, 1.1] (closed)."""
    return bool(BAND[0] <= drift <= BAND[1])


def aba_triplet(run_attempt, max_tries=MAX_TRIES):
    """c2time stage-2 triplet rule. `run_attempt(attempt)` performs one A-B-A attempt and returns a dict with at least
    tA1, tA2, tB, hashes_ok. Drift outside the band -> repeated (<= 2 retries); the last attempt decides validity, and
    a hash failure in ANY attempt invalidates. Field names and arithmetic are c2time's."""
    tries = []
    for attempt in range(max_tries):
        d = dict(run_attempt(attempt))
        ta1, ta2, tb = float(d['tA1']), float(d['tA2']), float(d['tB'])
        d.update(attempt=attempt, drift=ta2 / ta1, ratio=tb / (.5 * (ta1 + ta2)), hashes_ok=bool(d['hashes_ok']))
        tries.append(d)
        if drift_ok(d['drift']):
            break
    return dict(tries[-1], valid=bool(drift_ok(tries[-1]['drift']) and all(t_['hashes_ok'] for t_ in tries)),
                retries=len(tries) - 1, drifts=[t_['drift'] for t_ in tries], attempts=tries)


def p1time_band_is_ours(src_text):
    """p1time keeps its band inline; confirm by source that it is this band and this retry count (C-E5.3 support)."""
    return bool('BAND = (1 / 1.1, 1.1)' in src_text and 'for attempt in range(3):' in src_text
                and 'if BAND[0] <= drift <= BAND[1]:' in src_text)


# ------------------------------------------------------------------ records ----

def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=60).strip()
    except Exception as e:  # noqa: BLE001
        return f'ERROR: {e}'


SMI_QUERY = ('index,pci.bus_id,uuid,name,memory.total,driver_version,clocks.sm,clocks.mem,clocks.applications.graphics,'
             'clocks.applications.memory,power.limit,power.draw,temperature.gpu')
SMI_KEYS = tuple(SMI_QUERY.split(','))
TELEMETRY_MANDATORY = ('name', 'uuid', 'driver_version', 'clocks.sm', 'clocks.mem', 'power.limit', 'temperature.gpu')
TELEMETRY_MAY_BE_UNSUPPORTED = ('clocks.applications.graphics', 'clocks.applications.memory', 'power.draw')
_UNSUP = ('[n/a]', 'n/a', '[not supported]', 'not supported')      # nvidia-smi's explicit 'no such capability' only;
#                                                                     '[Unknown Error]' and query errors are failures


def _unsupported(v):
    return str(v).strip().lower() in _UNSUP


def parse_smi(text):
    """`nvidia-smi --query-gpu=SMI_QUERY --format=csv,noheader` -> list of {field: value} (raw strings)."""
    rows = []
    for line in (text or '').splitlines():
        if not line.strip():
            continue
        vals = [v.strip() for v in line.split(',')]
        if len(vals) != len(SMI_KEYS):
            raise ValueError(f'nvidia-smi row has {len(vals)} fields, expected {len(SMI_KEYS)}: {line!r}')
        rows.append(dict(zip(SMI_KEYS, vals)))
    return rows


def smi_rows(uuid=None):
    """(rows, error). With uuid: only that device (nvidia-smi --id=<uuid>)."""
    cmd = ['nvidia-smi', f'--query-gpu={SMI_QUERY}', '--format=csv,noheader'] + ([f'--id={uuid}'] if uuid else [])
    try:
        return parse_smi(subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=60)), None
    except Exception as e:  # noqa: BLE001
        return [], f'ERROR: {e}'


def cuda_uuid(ordinal):
    """UUID of CUDA device `ordinal` (the visible-device ordinal JAX uses) from the CUDA driver, 'GPU-...' as nvidia-smi
    prints it; None if libcuda is unavailable."""
    try:
        import ctypes
        lib = ctypes.CDLL('libcuda.so.1')
        if lib.cuInit(0) != 0:
            return None
        dev = ctypes.c_int()
        if lib.cuDeviceGet(ctypes.byref(dev), int(ordinal)) != 0:
            return None
        buf = (ctypes.c_ubyte * 16)()
        fn = getattr(lib, 'cuDeviceGetUuid_v2', None) or lib.cuDeviceGetUuid
        if fn(buf, dev) != 0:
            return None
        h = bytes(buf).hex()
        return f'GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}'
    except Exception:  # noqa: BLE001
        return None


def _bus(b):
    try:
        dom, rest = str(b).lower().split(':', 1)
        return f'{int(dom, 16):08x}:{rest}'
    except Exception:  # noqa: BLE001
        return str(b).lower()


def resolve_device(rows, ordinal, cvd=None, cu_uuid=None, pci_bus_id=None, device_order=None):
    """Bind the EXECUTING device (JAX device ordinal) to one nvidia-smi row (job audit E5 item 3). Evidence, in order:
    the CUDA driver UUID of that ordinal; JAX's PCI bus id; the CUDA_VISIBLE_DEVICES entry at that ordinal (a UUID, or an
    index only under CUDA_DEVICE_ORDER=PCI_BUS_ID); a single visible GPU. Conflicting evidence or no unique match ->
    resolved=False."""
    out = dict(resolved=False, ordinal=ordinal, cvd=cvd, cuda_uuid=cu_uuid, pci_bus_id=pci_bus_id, evidence={})
    by_uuid = {r['uuid']: r for r in rows}
    cands = {}
    if cu_uuid:
        cands['cuda-driver uuid of the JAX ordinal'] = cu_uuid if cu_uuid in by_uuid else 'NO-ROW'
    if pci_bus_id:
        m = [r['uuid'] for r in rows if _bus(r['pci.bus_id']) == _bus(pci_bus_id)]
        cands['JAX pci bus id'] = m[0] if len(m) == 1 else 'NO-ROW'
    if cvd and ordinal is not None:
        ent = [x.strip() for x in str(cvd).split(',') if x.strip()]
        if 0 <= int(ordinal) < len(ent):
            e = ent[int(ordinal)]
            if e.startswith(('GPU-', 'MIG-')):
                cands['CUDA_VISIBLE_DEVICES uuid'] = e if e in by_uuid else 'NO-ROW'
            elif e.isdigit() and device_order == 'PCI_BUS_ID':
                m = [r['uuid'] for r in rows if r['index'] == e]
                cands['CUDA_VISIBLE_DEVICES index (PCI_BUS_ID order)'] = m[0] if len(m) == 1 else 'NO-ROW'
    out['evidence'] = cands
    vals = set(cands.values())
    if 'NO-ROW' in vals or len(vals) > 1:
        out['reason'] = 'conflicting or unmatched device evidence'
        return out
    if len(vals) == 1:
        u = vals.pop()
        out.update(resolved=True, uuid=u, method=' + '.join(sorted(cands)), row=by_uuid[u])
        return out
    if len(rows) == 1:
        out.update(resolved=True, uuid=rows[0]['uuid'], method='single visible GPU', row=rows[0])
        return out
    out['reason'] = 'no device evidence and several visible GPUs'
    return out


def smi_record(uuid=None):
    """Telemetry of the executing device: its nvidia-smi row (application clocks / power draw may be 'unsupported',
    recorded explicitly), the raw error if the query failed, and the driver's CUDA version."""
    rows, err = smi_rows(uuid)
    row = None
    if not err:
        m = [r for r in rows if uuid is None or r['uuid'] == uuid]
        row = dict(m[0]) if len(m) == 1 else None
        if row is None:
            err = f'no unique nvidia-smi row for {uuid}'
    if row:
        for k in TELEMETRY_MAY_BE_UNSUPPORTED:
            if _unsupported(row.get(k, '')):
                row[k] = 'unsupported'
    head = _run(['nvidia-smi'])
    m = re.search(r'CUDA Version:\s*([0-9.]+)', head)
    return dict(row=row, error=err, n_rows=len(rows), cuda_driver_version=m.group(1) if m else None)


def telemetry_problems(rec, uuid):
    """Fail-closed completeness of one record (job audit E5 item 7). Empty list = complete."""
    if not rec:
        return ['record missing']
    p = []
    smi = rec.get('nvidia_smi') or {}
    row = smi.get('row') or {}
    if smi.get('error'):
        p.append(f"nvidia-smi: {smi['error']}")
    for k in TELEMETRY_MANDATORY:
        v = str(row.get(k, '')).strip()
        if not v or _unsupported(v) or 'error' in v.lower():
            p.append(f'{k} missing or not a value ({v!r})')
    for k in TELEMETRY_MAY_BE_UNSUPPORTED:
        v = str(row.get(k, '')).strip()
        if v != 'unsupported' and (not v or _unsupported(v) or 'error' in v.lower()):
            p.append(f'{k} neither a value nor explicitly unsupported ({v!r})')
    if not uuid or row.get('uuid') != uuid:
        p.append(f"record uuid {row.get('uuid')!r} is not the executing device {uuid!r}")
    ver = rec.get('versions') or {}
    for k in ('jax', 'jaxlib'):
        if not ver.get(k):
            p.append(f'{k} version missing')
    for k in ('jax_cuda_plugin', 'jax_cuda_pjrt'):    # a GPU job cannot run without the CUDA plugin: 'not installed' fails
        if not ver.get(k) or ver.get(k) == 'not installed':
            p.append(f'{k}: {ver.get(k)!r}')
    if not re.fullmatch(r'[0-9]+(\.[0-9]+)*', str(smi.get('cuda_driver_version') or '')):
        p.append(f"CUDA driver version missing ({smi.get('cuda_driver_version')!r})")
    env = rec.get('env') or {}
    if env.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest':
        p.append(f"JAX_DEFAULT_MATMUL_PRECISION={env.get('JAX_DEFAULT_MATMUL_PRECISION')!r}")
    if str(env.get('JAX_ENABLE_X64')).lower() not in ('true', '1'):
        p.append(f"JAX_ENABLE_X64={env.get('JAX_ENABLE_X64')!r}")
    return p


CUDA_PLUGINS = ('jax-cuda12-plugin', 'jax-cuda13-plugin')
CUDA_PJRT = ('jax-cuda12-pjrt', 'jax-cuda13-pjrt')


def versions(dists=None):
    """Installed versions from import metadata (names normalised to lower case with '-'); plus the explicit fields
    jax_cuda_plugin / jax_cuda_pjrt = 'name==version' of whichever is installed, or 'not installed' when the metadata
    shows none."""
    out = {}
    for d in (IM.distributions() if dists is None else dists):
        n = (d.metadata.get('Name') or '').lower().replace('_', '-')
        if n.startswith('jax') or n.startswith('nvidia-cuda') or n.startswith('nvidia-cudnn') or n in ('flax', 'numpy', 'scipy'):
            out[n] = d.version
    for key, names in (('jax_cuda_plugin', CUDA_PLUGINS), ('jax_cuda_pjrt', CUDA_PJRT)):
        hit = [f'{n}=={out[n]}' for n in names if out.get(n)]
        out[key] = ', '.join(hit) if hit else 'not installed'
    return dict(sorted(out.items()))


def env_record(tag, uuid=None):
    return dict(tag=tag, device_uuid=uuid, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                env={k: os.environ.get(k) for k in ('JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION', 'XLA_FLAGS',
                                                    'XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX_PLATFORMS', 'JCPREF_STORE',
                                                    'JCPREF_META', 'SOURCE_COMMIT')},
                nvidia_smi=smi_record(uuid), versions=versions())


def gpu_type_ok(expected, row):
    """expected in {'A100-80G', 'H200', 'L40S'}; row = the EXECUTING device's nvidia-smi row (resolve_device)."""
    if not row:
        return False, 'executing device not resolved'
    name, memtot = row.get('name', ''), row.get('memory.total', '')
    try:
        mib = float(str(memtot).split()[0])
    except (ValueError, IndexError):
        mib = 0.
    ok = {'A100-80G': 'A100' in name and mib >= 79000, 'H200': 'H200' in name, 'L40S': 'L40S' in name}.get(expected, False)
    return bool(ok), f'{name} {memtot} {row.get("uuid")}'


# ------------------------------------------------------------------ provenance ----

def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def verify_sources():
    """Every file listed in e5/PROVENANCE.json must hash to its recorded sha256 (copied files and the lane files P1's
    code imports). Returns (ok, rows)."""
    prov = json.loads((E5 / 'PROVENANCE.json').read_text())
    rows = []
    for f in prov['files']:
        p = HERE / f['path']
        got = sha_file(p) if p.exists() else None
        rows.append(dict(path=f['path'], want=f['sha256'], got=got, ok=got == f['sha256']))
    return all(r['ok'] for r in rows), rows


# ------------------------------------------------------------------ execution inventory (certify.py) ----

_NAMED = {'dev6': [f'dev6|{i}' for i in range(6)], 'dev2': ['dev6|0', 'dev6|1'],
          'sub16': [f'dev6|{i}' for i in range(6)] + [f'val32|{i}' for i in range(10)]}


def _cases(which):
    return list(_NAMED[which]) if isinstance(which, str) else [f'{c}|{int(i)}' for c, i in which]


def _p1_label(sp):
    return f"{sp['mesh']}|R{sp['Rp']}|{sp['rule']}|{sp.get('policy', 'frozen')}|{sp['scheme']}"


def inv_e5time(cfg):
    """Execution inventory of one e5time config (code audit 3 item 12). Keys are tuples
    ('e5time', panel, subject, kind, case, rep) with rep = -1 for untimed runs:

      P-cost (p1time, unchanged): per unique subject (A + subjects, deduplicated by label as p1time does) and case: the
        untimed scoring run ('score'), its 'tight' and 'tighter' replays, and 'qp' / 'qpp' when the subject carries those
        Q2 roles; one split/fused 'equivalence' run on the first case; per (subject, case, rep) one A-B-A 'triplet' slot
        and one 'fused-triplet' slot (a slot covers its <= 2 retries); per doubled subject, case, rep one 'doubled' slot.
      P-cross (c2time sections 1, warm, 4, T-dbl): per subject (A, ROMs, pinned FOMs) and case one 'score' and one 'warm'
        run; per (ROM or FOM subject, case, rep) one 'triplet' slot; per T-dbl case and rep one 'tdbl' slot.
    """
    keys = []
    pc = cfg['pcost']
    pcases = _cases(pc['cases'])
    subs = {}
    for sp in [pc['A']] + list(pc['subjects']):
        subs.setdefault(_p1_label(sp), sp)
    for lb, sp in subs.items():
        for i, c in enumerate(pcases):
            kinds = ['score', 'tight', 'tighter'] + [k for k in ('qp', 'qpp') if sp.get(k)] + (['equivalence'] if i == 0 else [])
            keys += [('e5time', 'pcost', lb, k, c, -1) for k in kinds]
    reps = int(pc['reps'])
    for sp in pc['subjects']:
        for c in pcases:
            for r in range(reps):
                keys += [('e5time', 'pcost', _p1_label(sp), 'triplet', c, r), ('e5time', 'pcost', _p1_label(sp), 'fused-triplet', c, r)]
    for sp in pc.get('doubled', []):
        for c in _cases(sp.get('cases', pc['cases'])):
            for r in range(reps):
                keys.append(('e5time', 'pcost', _p1_label(sp), 'doubled', c, r))
    xc = cfg['pcross']
    xcases = _cases(xc['cases'])
    foms = sorted({k for d in xc['expected_comparators'].values() for k in d.values()})
    roms = [r['label'] for r in xc['roms']]
    xreps = int(xc['reps'])
    for s in [xc['A']['label']] + roms + foms:
        for c in xcases:
            keys += [('e5time', 'pcross', s, 'score', c, -1), ('e5time', 'pcross', s, 'warm', c, -1)]
    for s in roms + foms:
        for c in xcases:
            keys += [('e5time', 'pcross', s, 'triplet', c, r) for r in range(xreps)]
    for c in _cases(xc.get('controls', {}).get('doubled_cases', [])):
        keys += [('e5time', 'pcross', xc['A']['label'], 'tdbl', c, r) for r in range(xreps)]
    return keys


def klass(key):
    """Timing class of an inventory key: 'e5time|<panel>|<subject>|<kind>' (per-subject, so 4096^2 P-cost subjects and
    each FOM comparator get their own measured seconds; a class absent from the smoke timings is reported missing)."""
    _, panel, subj, kind = key[:4]
    return f'e5time|{panel}|{subj}|{kind}'


# ------------------------------------------------------------------ registered E5 populations (code audit 4) ----
# Independent of the job's own config: DESIGN 4.1, A1.11, A2.8 (comparators), A2.9 (smk4: dev6|0-4 x 2 reps). Keyed by
# the e5time config attempt. Analysis requires every job's evidence to match its attempt's contract exactly.

P1_REF_PREFIX = 'output/p1time_time/'           # P1c archive layout (runs/time/archive/OUTPUTS.sha256)
E5_FAMS = ('R128|fib4181|frozen|BE', 'R384|gauss128|frozen|CNR')
E5_RP = {'R128|fib4181|frozen|BE': 128, 'R384|gauss128|frozen|CNR': 384}
E5_A = '1024|R384|gauss128|frozen|CNR'
E5_ROMS = ('ckpt:18f0266ae6f0|R128|fib4181|frozen|BE|0.005|LSPG|prod',
           'ckpt:18f0266ae6f0|R384|gauss128|frozen|CNR|0.005|LSPG|prod')
E5_PINNED = {E5_ROMS[0]: {'|H|union|joint': 'ps|N96|0.0125', '|H|CC|same_mesh': 'fd|CC|L1024|0.0125|t2'},
             E5_ROMS[1]: {'|H|union|joint': 'ps|N96|0.0125', '|H|CC|same_mesh': 'fd|CC|L1024|0.005|t1'}}
_DEV5 = tuple(f'dev6|{i}' for i in range(5))
_DEV6 = tuple(f'dev6|{i}' for i in range(6))
_X16 = _DEV6 + tuple(f'val32|{i}' for i in range(10))


def make_contract(gpu_type, meshes, pcost_cases, pcost_reps, pcross_cases, pcross_reps, roms=E5_ROMS, pinned=None):
    pinned = pinned or E5_PINNED
    subs = tuple(f'{L}|{f}' for f in E5_FAMS for L in meshes)
    return dict(gpu_type=gpu_type, meshes=tuple(meshes), pcost_subjects=subs, scored=tuple(sorted(set(subs) | {E5_A})),
                pcost_cases=tuple(pcost_cases), pcost_reps=int(pcost_reps),
                doubled=(f'1024|{E5_FAMS[0]}', E5_A), roms=tuple(roms), pinned=pinned,
                foms=tuple(sorted({k for d in pinned.values() for k in d.values()})),
                pcross_cases=tuple(pcross_cases), pcross_reps=int(pcross_reps))


E5_CONTRACT = {
    'e5smk4': make_contract('A100-80G', (256, 1024), _DEV5, 2, _DEV5, 2),
    'e5smk1': make_contract('H200', (256, 1024), _DEV5, 2, _DEV5, 2),
    'e5prod_a100': make_contract('A100-80G', (256, 1024), _DEV6, 3, _X16, 3),
    'e5prod_h200': make_contract('H200', (256, 1024, 4096), _DEV6, 3, _X16, 3),
}


def population_check(rows, keyf, required):
    """Every required key present exactly once and no unregistered key. Returns (ok, detail)."""
    seen = {}
    for x in rows:
        k = keyf(x)
        seen[k] = seen.get(k, 0) + 1
    req = set(required)
    missing = sorted(map(str, req - set(seen)))
    dup = sorted(str(k) for k, n in seen.items() if n > 1)
    extra = sorted(map(str, set(seen) - req))
    ok = bool(req) and not missing and not dup and not extra
    return ok, dict(required=len(req), missing=missing[:20], n_missing=len(missing), duplicates=dup[:20], extra=extra[:20])
