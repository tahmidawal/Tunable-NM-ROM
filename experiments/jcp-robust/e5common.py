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
SMI_FIELDS = ('name,uuid,driver_version,clocks.sm,clocks.mem,clocks.applications.graphics,'
              'clocks.applications.memory,power.limit,power.draw,temperature.gpu')


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


def smi_record():
    q = _run(['nvidia-smi', f'--query-gpu={SMI_FIELDS}', '--format=csv'])
    mem = _run(['nvidia-smi', '--query-gpu=name,uuid,memory.total', '--format=csv,noheader'])
    head = _run(['nvidia-smi'])
    m = re.search(r'CUDA Version:\s*([0-9.]+)', head)
    return dict(query=q, memory=mem, cuda_driver_version=m.group(1) if m else None)


def versions():
    out = {}
    for d in IM.distributions():
        n = (d.metadata.get('Name') or '').lower()
        if n.startswith('jax') or n.startswith('nvidia-cuda') or n.startswith('nvidia-cudnn') or n in ('flax', 'numpy', 'scipy'):
            out[n] = d.version
    return dict(sorted(out.items()))


def env_record(tag):
    return dict(tag=tag, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), host=socket.gethostname(),
                slurm_job_id=os.environ.get('SLURM_JOB_ID'), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                env={k: os.environ.get(k) for k in ('JAX_ENABLE_X64', 'JAX_DEFAULT_MATMUL_PRECISION', 'XLA_FLAGS',
                                                    'XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX_PLATFORMS', 'JCPREF_STORE',
                                                    'JCPREF_META', 'SOURCE_COMMIT')},
                nvidia_smi=smi_record(), versions=versions())


def gpu_type_ok(expected, smi_memory_lines):
    """expected in {'A100-80G', 'H200', 'L40S'}; smi lines 'name, uuid, NNN MiB' (p1common.gpu_info format)."""
    if not smi_memory_lines:
        return False, 'no nvidia-smi output'
    name, _, memtot = (s.strip() for s in smi_memory_lines[0].split(',')[:3])
    mib = float(memtot.split()[0]) if memtot and memtot.split()[0].replace('.', '').isdigit() else 0.
    if expected == 'A100-80G':
        ok = 'A100' in name and mib >= 79000
    elif expected == 'H200':
        ok = 'H200' in name
    elif expected == 'L40S':
        ok = 'L40S' in name
    else:
        ok = False
    return bool(ok), f'{name} {memtot}'


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
