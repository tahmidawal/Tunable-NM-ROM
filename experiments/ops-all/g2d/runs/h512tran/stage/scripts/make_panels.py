"""g2d expanded mandate: generate the full-ladder panel configs, operator manifests, job bodies and checkpoint staging
(hard links, no extra disk) for every cell. Run locally; writes configs/pl<n>.json, configs/ops-pl<n>.json,
configs/phl<n>_hbk.json, configs/phl<n>_ops.json, scripts/jobs/pl<n>.sh, scripts/jobs/phl<n>.sh, ckpt/<job>/...
"""
import hashlib
import json
import os
from pathlib import Path

G = Path(__file__).resolve().parents[1]
WT = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')
NS = '/cluster/tufts/paralab/tawal01/opsall_20260924/g2d'
BK = G / 'src/burgers/experiments/burgers-bank-knob'
CMP = WT / '2026-09-23-burgers-compare-hires/experiments/burgers-compare-hires'
OPT = WT / '2026-09-22-ops-timing-panel/experiments/ops-timing-panel/runs/opt201/opsckpt'
HCMP = WT / '2026-09-23-heat-compare-hires/experiments/heat-compare-hires/runs'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def link(src, job, rel):
    dest = G / 'ckpt' / job / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        os.link(src, dest)
    return f'ckpt/{rel}'


def remote(tjob, arm):
    return dict(name=arm, path=f'{NS}/{tjob}/out/train/{arm}/best.pt', result_json=f'{NS}/{tjob}/out/train/{arm}/result.json',
                copy=False, source=f'g2d training job {tjob}')


def burgers():
    local = {
        256: {n: (OPT / f'{n}.pt', 'ops-timing-panel opt201 (job 4181372)') for n in
              ('fno-large', 'unet-small', 'unet-medium', 'unet-refine', 'unet-large', 'tsol-small', 'tsol-medium', 'tsol-refine',
               'tsol-large', 'don-small', 'don-medium', 'don-refine', 'don-large')},
        1024: {n: (CMP / f'runs/t1024/archive/out/{n}/best.pt', 'burgers-compare-hires t1024 (job 4196040)')
               for n in ('unet-refine', 'fno-large', 'tsol-refine', 'don-small')},
        2048: {'fno-large': (CMP / 'runs/p2048b/archive/out/fno-large/best.pt', 'burgers-compare-hires p2048b (job 4207177, salvaged)'),
               'unet-refine': (CMP / 'runs/t2048/archive/out/unet-refine/best.pt', 'burgers-compare-hires t2048 (job 4196041)'),
               'tsol-refine': (CMP / 'runs/t2048/archive/out/tsol-refine/best.pt', 'burgers-compare-hires t2048 (job 4196041)'),
               'don-small': (CMP / 'runs/t2048s/archive/out/don-small/best.pt', 'burgers-compare-hires t2048s (job 4206643)')},
    }
    new = {256: [('t256ub64', 'unet-b64'), ('t256ff32', 'fno-w96f32')],
           512: [('t512ur', 'unet-refine'), ('t512fl', 'fno-large'), ('t512tr', 'tsol-refine'), ('t512ds', 'don-small'),
                 ('t512ul', 'unet-large'), ('t512ub64', 'unet-b64'), ('t512tl', 'tsol-large'), ('t512ff32', 'fno-w96f32')],
           1024: [(f't1024{k}', a) for k, a in (('ul', 'unet-large'), ('ub64', 'unet-b64'), ('tl', 'tsol-large'), ('ff32', 'fno-w96f32'))],
           2048: [(f't2048{k}', a) for k, a in (('ul', 'unet-large'), ('ub64', 'unet-b64'), ('tl', 'tsol-large'), ('ff32', 'fno-w96f32'))],
           4096: [('b4096tc', 'unet-refine'), ('t4096ts', 'tsol-refine'), ('t4096dn', 'don-small')]}
    out = {}
    for L in (256, 512, 1024, 2048, 4096):
        job = f'pl{L}'
        cfg = json.loads((BK / f'config-{L}.json').read_text())
        if L == 4096:
            h64 = json.loads((BK / 'config-h64a.json').read_text())
            cfg.update(bank_blocks=h64['bank_blocks'], timed_full_sha_every=h64['timed_full_sha_every'], timed_skip_score=True)
        cfg['attempt'] = job
        cfg['arms'] = [a for a in cfg['arms'] if a.get('family') in ('a', 'b')]
        cfg['parity_pairs'] = []
        cfg['keep_parent'] = False
        cfg['skip_certificates'] = True
        cfg['untimed_fom'] = ['fft_tight']
        cfg['audit_arms'] = ['fft_tight']
        cfg['purpose'] = (f'g2d ops-all expanded panel {L}^2: burgers-bank-knob config-{L}.json (bk{L}b), arms of families (a) head '
                          'and (b) span only (the setting ladder), every FOM setting timed except the fft_tight reference, '
                          'certificates skipped (bk jobs certified the same arms); operators timed after it in the same allocation')
        (G / 'configs' / f'{job}.json').write_text(json.dumps(cfg, indent=1) + '\n')
        ops = []
        for n, (p, src) in local.get(L, {}).items():
            rel = link(p, job, f'{n}.pt')
            ops.append(dict(name=n, path=rel, sha256=sha(p), copy=False, source=src))
        for tj, arm in new.get(L, []):
            ops.append(remote(tj, arm))
        (G / 'configs' / f'ops-{job}.json').write_text(json.dumps(dict(mesh=L, operators=ops), indent=1) + '\n')
        frac = '0.95' if L == 4096 else '0.90'
        (G / 'scripts/jobs' / f'{job}.sh').write_text(f'''# g2d expanded Burgers panel {L}^2 (DESIGN.md "Expanded mandate")
set -uo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION={frac}
cd experiments/burgers-bank-knob
"$PY" bankknob.py --config "$ROOT/configs/{job}.json" --checkpoint ../separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \\
  --rotation inputs/rotation_R512.npz --inputs ../b-panel/inputs --out "$OUT/drv" || echo "DRIVER FAILED"
[ -f "$OUT/drv/opcohort/index.json" ] || {{ echo "NO OPCOHORT"; exit 1; }}
cd ../burgers-compare-hires
mkdir -p "$OUT/drv/optiming" "$OUT/drv/fields"
"$PY" "$ROOT/scripts/stage_ops.py" "$ROOT/configs/ops-{job}.json" "$ROOT/opckpt" > "$ROOT/opckpt.txt"
cp "$ROOT/opckpt/staging.json" "$OUT/drv/staging.json"
NAMES=""
while read -r name path; do
  NAMES="$NAMES $name"
  "$PY" ops/optime.py --checkpoint "$path" --index "$OUT/drv/opcohort/index.json" --out "$OUT/drv/optiming" \\
    --fields "$OUT/drv/fields" --name "$name" --role "g2d expanded {L}^2" --repetitions 5 --burn-in 20 < /dev/null || echo "OPERATOR FAILED $name"
  "$PY" "$ROOT/scripts/opscore.py" "$OUT/drv" "$name"
done < "$ROOT/opckpt.txt"
true
rm -rf "$ROOT/opckpt" "$ROOT/ckpt" "$OUT/drv/opcohort"/*.npz "$OUT/drv/fields"
''')
        out[job] = [tj for tj, _ in new.get(L, [])]
    return out


def heat():
    hbk = json.loads((G / 'src/heat/experiments/heat-bank-knob/configs/h2d.json').read_text())
    pn = json.loads((G / 'src/heat/experiments/heat-compare-hires/configs/pn2048.json').read_text())
    local = {1024: ('tr1024a', 'heat-compare-hires tr1024a (job 4196355)'), 2048: ('tr2048', 'heat-compare-hires tr2048 (job 4196062)')}
    new = {1024: [('h1024ul', 'unet-large', 'unet'), ('h1024ub64', 'unet-b64', 'unet'), ('h1024ff32', 'fno-w96f32', 'fno')],
           2048: [('h2048ul', 'unet-large', 'unet'), ('h2048ub64', 'unet-b64', 'unet'), ('h2048ff32', 'fno-w96f32', 'fno')],
           4096: [('h4096u2', 'unet', 'unet'), ('h4096t2', 'transolver', 'transolver')],
           8192: [('h8192tran', 'transolver', 'transolver')]}
    for m in (256, 512):
        new[m] = [(f'h{m}{k}', a, f) for k, a, f in (('fno', 'fno', 'fno'), ('unet', 'unet', 'unet'), ('tran', 'transolver', 'transolver'),
                                                      ('deep', 'deeponet', 'deeponet'), ('ul', 'unet-large', 'unet'), ('ff32', 'fno-w96f32', 'fno'))]
    out = {}
    for n in (256, 512, 1024, 2048, 4096):
        job = f'phl{n}' if n != 4096 else 'phl4096b'
        c = dict(hbk, meshes=[n], cohorts=[['heldout_sealed_791099', 791099, 16]], ladder=[128, 96, 64, 48, 32, 16],
                 q_by_Rp={'128': [0], '96': [], '64': [], '48': [], '32': [], '16': []}, families={'cn': {'init': 'moments'}},
                 parity={'meshes': [], 'q': []}, profile={'meshes': []})
        c['g2d_note'] = ('heat-bank-knob h2d.json at one mesh, held-out cohort only (Table 1 cohort), every span R\' of the lane (linear '
                         'CN rungs) + the head-only arm nmrom_R128_q0_cn, CN family; full CN-CG grid; DST exact reference')
        (G / 'configs' / f'{job}_hbk.json').write_text(json.dumps(c, indent=1) + '\n')
        ops = {}
        if n in local:
            src, note = local[n]
            base = HCMP / src / 'pull/out'
            link(base / 'provenance.json', job, f'{src}/provenance.json')
            for fam in ('fno', 'unet', 'transolver', 'deeponet'):
                link(base / fam / 'best.pt', job, f'{src}/{fam}/best.pt')
                ops[f'op_{fam}'] = dict(checkpoint=f'{NS}/{job}/ckpt/{src}/{fam}/best.pt', family=fam, source=note,
                                        sha256=sha(base / fam / 'best.pt'))
        for tj, name, fam in new[n]:
            ops[f'op_{name}'] = dict(checkpoint=f'{NS}/{tj}/out/tr/{name}/best.pt', family=fam, source=f'g2d training job {tj}')
        p = dict(pn, mesh=n, operators_first=True, qm_first=False, fom_order=[], coarse_intervals=[], retime={}, operators=ops)
        for k in ('nmrom', 'pod', 'qm'):
            p.pop(k, None)
        p['g2d_note'] = 'heat-compare-hires panel.py with only operator blocks (+ DST control, sentinels); NM-ROM/FOM by hbk_run.py in the same job'
        (G / 'configs' / f'{job}_ops.json').write_text(json.dumps(p, indent=1) + '\n')
        (G / 'scripts/jobs' / f'{job}.sh').write_text(f'''# g2d expanded heat panel {n}^2
set -uo pipefail
cd experiments/heat-bank-knob
"$PY" hbk_run.py --config "$ROOT/configs/{job}_hbk.json" --out "$OUT/hbk" || echo "HBK FAILED"
rm -f "$OUT"/hbk/fields_*.npz
cd ../heat-compare-hires
export XLA_PYTHON_CLIENT_PREALLOCATE=false   # JAX must not reserve the GPU before the PyTorch operators (phl4096 OOM)
"$PY" panel.py --config "$ROOT/configs/{job}_ops.json" --out "$OUT/ops" || echo "OPS PANEL FAILED"
rm -rf "$OUT/ops/fields" "$ROOT/ckpt"
''')
        out[job] = [tj for tj, _, _ in new[n]]
    return out


if __name__ == '__main__':
    deps = {**burgers(), **heat()}
    (G / 'configs' / 'panel-deps.json').write_text(json.dumps(deps, indent=1) + '\n')
    print(json.dumps(deps))
