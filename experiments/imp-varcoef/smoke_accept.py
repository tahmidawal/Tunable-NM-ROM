"""Smoke acceptance (DESIGN §10, A1.19, A2.19, A3.19, A4.19): every must-fail control fired and every must-pass check passed
in the real-size smoke; production staging requires checks/smoke-accept.json with accepted = true.

    python smoke_accept.py --job runs/<smoke job> --out checks/smoke-accept.json
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def J(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    o = Path(a.job) / 'output'
    imp = J(o / 'checks_impl_smk-impl/result.json')
    ref = J(o / 'vref_smk-ref/result.json')
    lad = J(o / 'vladder_smk-ladder/result.json')
    tim = J(o / 'vtime_smk-time/result.json')
    log = '\n'.join(p.read_text() for p in (Path(a.job) / 'logs').glob('*.out'))
    gs = json.loads(subprocess.check_output([sys.executable, str(HERE / 'gate.py'), 'selftest']))
    c = dict(
        backend_gpu=bool('jax_backend=gpu' in log and imp['backend'] == 'gpu' and ref['backend'] == 'gpu'
                         and lad['backend'] == 'gpu' and tim['backend'] == 'gpu'),
        x64_highest=bool('x64+highest ok' in log),
        impl_real_size=bool(imp.get('small') is False),
        timing_pilots_ok=bool(tim.get('pilots_ok') is True and tim.get('pilot_foms')),
        impl_checks_pass=bool(imp['all_pass']),
        impl_controls_fire=bool(imp['controls_fire']),
        ref_controls_fire=bool(ref['controls'] and ref['controls'].get('all_fire') is True),
        ref_complete=bool(ref.get('complete')),
        ladder_complete=bool(lad.get('complete')),
        timing_complete=bool(tim.get('complete')),
        timing_controls_fire=bool(all(v.get('fires') for v in tim['controls'].values())
                                  and {'T-dbl', 'T-fault', 'T-id'} <= set(tim['controls'])),
        timing_gates=bool(tim['gates'].get('A-valid') and tim['gates'].get('G-RR') and tim['gates'].get('G-FR')),
        timing_h200=bool('H200' in str(tim.get('gpu'))),
        gate_selftest=bool(gs['all_fire']),
        all_done=bool('ALL-DONE' in log),
    )
    out = dict(job=str(a.job), checks=c, accepted=all(c.values()),
               sources={k: hashlib.sha256(Path(p).read_bytes()).hexdigest() for k, p in
                        dict(impl=o / 'checks_impl_smk-impl/result.json', ref=o / 'vref_smk-ref/result.json',
                             ladder=o / 'vladder_smk-ladder/result.json', timing=o / 'vtime_smk-time/result.json').items()})
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
