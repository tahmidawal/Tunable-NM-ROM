"""Assemble g2d's source trees from pinned commits of the source lanes (git show, byte-exact).

    /home/tahmid/Dev/.venv/bin/python scripts/assemble.py

Writes src/burgers/<repo path> and src/heat/<repo path> and COPIED-FROM.json (commit, path, sha256).
Every file is taken from a commit blob, never from a working copy.
"""
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
HERE = Path(__file__).resolve().parents[1]

C = {
    'speed': 'fb4a9ff7',     # exp/2026-09-23-burgers2d-speed HEAD (Table 1 Burgers 512^2 row: job 4241035)
    'bank': '340627ca',      # exp/2026-09-23-burgers-bank-knob HEAD (Table 1 Burgers 4096^2 row: job 4197473)
    'cmp': '07c0e3e8',       # exp/2026-09-23-burgers-compare-hires HEAD (Burgers operator machinery)
    'hcmp': '9a5baf9a',      # exp/2026-09-23-heat-compare-hires HEAD (heat operator machinery + panel)
    'hbk': 'cc92aca8',       # exp/2026-09-23-heat-bank-knob HEAD (Table 1 heat rows: job 4197350)
}
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
BURGERS = {
    'speed': ['experiments/burgers2d-speed/b2speed.py', 'experiments/burgers2d-speed/b2fast.py',
              'experiments/burgers2d-speed/config-512.json',
              'experiments/burgers-bank-knob/bkfast.py', 'experiments/burgers-bank-knob/inputs/rotation_R512.npz',
              'experiments/burgers-repanel/xfast.py', 'experiments/hires-burgers/hops.py',
              'experiments/hires-burgers/hfast.py',
              'experiments/b-panel/speed/fast.py', 'experiments/b-panel/speed/ladders.py',
              'experiments/b-panel/inputs/directions_qtd02.npz',
              'experiments/b-panel/inputs/rules-eqtop/rule_q0_m1024_qrg304_reachable.npz',
              'experiments/b-ladder-top/topfix.py', 'experiments/cheap-corrections/varpro.py',
              'experiments/head-ablation/arms.py', 'experiments/head-ablation/ladder.py',
              'experiments/head-ablation/ablation.py', 'experiments/mr-burgers2d/engines.py',
              'experiments/mr-burgers2d/iterative_paths.py', 'experiments/mr-burgers2d/accuracy_paths.py',
              'experiments/separable-decoder/sep_common.py', CHECKPOINT],
    'bank': ['experiments/burgers-bank-knob/bankknob.py', 'experiments/burgers-bank-knob/config-h64a.json',
             'experiments/burgers-bank-knob/config-4096.json'],
    'cmp': ['experiments/burgers-compare-hires/opdata.py',
            'experiments/burgers-compare-hires/inputs/pinned/train-index.json',
            'experiments/burgers-compare-hires/inputs/pinned/validation-index.json',
            'experiments/burgers-compare-hires/operators-1024.json',
            'experiments/burgers-compare-hires/operators-2048.json'] +
           [f'experiments/burgers-compare-hires/ops/{f}' for f in
            ('train.py', 'dataset.py', 'model.py', 'families.py', 'spectral_conv_f64.py', 'NEURALOPERATOR-LICENSE',
             'worker.py', 'optime.py')] +
           [f'experiments/burgers-compare-hires/inputs/opconfigs/{f}.json' for f in
            ('fno-large', 'unet-refine', 'tsol-refine', 'don-small', 'smoke-fno', 'smoke-unet', 'smoke-tsol', 'smoke-don')],
}
HEAT = {
    'hcmp': ['experiments/heat-compare-hires/panel.py', 'experiments/heat-compare-hires/podfac.py',
             'experiments/heat-compare-hires/audit_panel.py', 'experiments/heat-compare-hires/summarize.py',
             'experiments/heat-compare-hires/configs/pn4096.json', 'experiments/heat-compare-hires/configs/pn2048.json',
             'experiments/heat-compare-hires/configs/smoke64.json'] +
            [f'experiments/heat-compare-hires/ops/{f}' for f in
             ('train_ops.py', 'heatops.py', 'families.py', 'spectral_conv_f64.py', 'NEURALOPERATOR-LICENSE')] +
            [f'experiments/heat-compare-hires/configs/ops/{f}.json' for f in ('fno', 'unet', 'transolver', 'deeponet')] +
            ['experiments/hires-heat/core.py', 'experiments/hires-heat/inputs/wide2d/bank.pkl',
             'experiments/hires-heat/inputs/wide2d/head_K8.pkl', 'experiments/hires-heat/inputs/wide2d/training.json',
             'experiments/hires-heat/inputs/wide2d/SHA256SUMS'],
    'hbk': ['experiments/heat-bank-knob/hbk_core.py', 'experiments/heat-bank-knob/hbk_run.py',
            'experiments/heat-bank-knob/core.py', 'experiments/heat-bank-knob/prep_2d.npz',
            'experiments/heat-bank-knob/prep_2d.json', 'experiments/heat-bank-knob/configs/h2d.json',
            'experiments/heat-bank-knob/make_prep.py',
            'experiments/heat-bank-knob/inputs/wide2d/SHA256SUMS', 'experiments/heat-bank-knob/inputs/wide2d/bank.pkl',
            'experiments/heat-bank-knob/inputs/wide2d/head_K8.pkl', 'experiments/heat-bank-knob/inputs/wide2d/training.json'],
}


def show(commit, path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', f'{commit}:{path}'])


def main():
    record = {}
    for tree, spec in (('burgers', BURGERS), ('heat', HEAT)):
        for key, paths in spec.items():
            full = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', C[key]], text=True).strip()
            for pth in paths:
                try:
                    data = show(full, pth)
                except subprocess.CalledProcessError:
                    print('MISSING', key, pth)
                    continue
                dest = HERE / 'src' / tree / pth
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                record[f'src/{tree}/{pth}'] = dict(commit=full, path=pth, sha256=hashlib.sha256(data).hexdigest(),
                                                   bytes=len(data), modified=False)
    (HERE / 'COPIED-FROM.json').write_text(json.dumps(record, indent=1) + '\n')
    print(len(record), 'files')


if __name__ == '__main__':
    main()
