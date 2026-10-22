"""imp-fronts: register a trained arm bank as rom/inputs/banks/s<label>/ (separate-file variant of the unchanged
register_banks.py, which keys the directory by the integer seed; every arm is trained with seed 1, so the label must differ
from the seed or the arm would overwrite K1 = inputs/banks/s1). CPU only, no JAX.

  production   python register_arm.py <archive dir> <seed> <label>
               as register_banks.py (SEED-COMPLETE, OUTPUTS.sha256 / LARGE.sha256 verification, copy, BANK.json) with
               output inputs/banks/s<label>/ and BANK.json fields label, arm, seed. label must match [a-z][a-z0-9]*.
  pulled dir   python register_arm.py --seed-dir <pulled run_seed.sh output dir> <seed> <label> --job-log <job .out log>
                                     [--root <dir>]
               production provenance required (code audit B-5): run_record.json with mode == 'production' and seed == the
               CLI seed (smoke / dry-run outputs also write SEED-COMPLETE and are refused), the training job's log ending
               in ALL-DONE, no EXHAUSTED.json beside the output, SEED-COMPLETE, every file in OUTPUTS.sha256 that is present
               verifies, ckpt + rotation present and listed; --root redirects the inputs/banks/ destination (tests only).
  smoke        python register_arm.py --runtime <seed output dir> <dest dir> <label>
               copies stageB/sep_hfit_dense_mid_N256_dense.pkl and rotation/rotation_R512.npz of a bank trained inside
               the same smoke job into <dest dir>/{ckpt.pkl, rotation_R512.npz} and writes BANK.json with
               runtime_hashed=true (smoke only: the label must start with 'smk').
"""
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CK, ROT = Path('stageB') / 'sep_hfit_dense_mid_N256_dense.pkl', Path('rotation') / 'rotation_R512.npz'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_bank(src_dir, dst, meta):
    dst.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(src_dir / CK, dst / 'ckpt.pkl')
    shutil.copyfile(src_dir / ROT, dst / 'rotation_R512.npz')
    meta = dict(meta, ckpt='ckpt.pkl', rotation='rotation_R512.npz',       # canonical relative file names (floors_b.py)
                ckpt_sha256=sha(dst / 'ckpt.pkl'), rotation_sha256=sha(dst / 'rotation_R512.npz'),
                source_ckpt_sha256=sha(src_dir / CK), source_rotation_sha256=sha(src_dir / ROT))
    assert meta['ckpt_sha256'] == meta['source_ckpt_sha256'] and meta['rotation_sha256'] == meta['source_rotation_sha256']
    rj = src_dir / 'rotation' / 'rotation_R512.json'
    if rj.exists():
        meta['rotation_json'] = json.loads(rj.read_text())
    rr = src_dir / 'run_record.json'
    meta['run_record'] = json.loads(rr.read_text()) if rr.exists() else {}
    (dst / 'BANK.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')
    return meta


def main():
    a = sys.argv[1:]
    if a and a[0] == '--runtime':
        src, dst, label = Path(a[1]), Path(a[2]), a[3]
        assert label.startswith('smk'), 'runtime registration is for smoke banks only'
        m = write_bank(src, dst, dict(label=label, runtime_hashed=True, source=str(src)))
        print(label, m['ckpt_sha256'][:12], m['rotation_sha256'][:12])
        return
    if a and a[0] == '--seed-dir':
        d, seed, label = Path(a[1]), a[2], a[3]
        root = Path(a[a.index('--root') + 1]) if '--root' in a else HERE
        assert re.fullmatch(r'[a-z][a-z0-9]*', label), 'arm labels are lower-case and never numeric (numeric = seed banks)'
        assert (d / 'SEED-COMPLETE').exists(), f'{d}: SEED-COMPLETE missing (training incomplete)'
        rr = d / 'run_record.json'
        assert rr.exists(), f'{d}: run_record.json missing (no production provenance)'
        rec0 = json.loads(rr.read_text())
        assert rec0.get('mode') == 'production', f"run_record mode {rec0.get('mode')!r}: only production outputs register"
        assert int(rec0.get('seed', -1)) == int(seed), f"run_record seed {rec0.get('seed')} != CLI seed {seed}"
        assert '--job-log' in a, 'the training job log (ALL-DONE) is required'
        log = Path(a[a.index('--job-log') + 1]).read_text().strip().splitlines()
        assert log and log[-1].strip() == 'ALL-DONE', 'training job log does not end in ALL-DONE'
        for up in (d, d.parent, d.parent.parent):
            assert not (up / 'EXHAUSTED.json').exists(), f'{up}/EXHAUSTED.json: training exhausted (arm NOT RUN)'
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet', '--ignore-missing'], cwd=d, check=True)
        listed = {l.split(None, 1)[1].strip() for l in (d / 'OUTPUTS.sha256').read_text().splitlines()}
        for f in (CK, ROT):
            assert str(f) in listed and (d / f).exists(), f'{f}: not present / not listed in OUTPUTS.sha256'
        rec = json.loads((d / 'run_record.json').read_text()) if (d / 'run_record.json').exists() else {}
        m = write_bank(d, root / 'inputs' / 'banks' / f's{label}', dict(label=label, arm=label, seed=int(seed), source=str(d),
                                                                         job_id=rec.get('job_id')))
        print(label, m['ckpt_sha256'][:12], m['rotation_sha256'][:12])
        return
    arch, seed, label = Path(a[0]), a[1], a[2]
    assert re.fullmatch(r'[a-z][a-z0-9]*', label), 'arm labels are lower-case and never numeric (numeric = seed banks)'
    d = arch / 'output' / f'train2d_s{seed}'
    assert (d / 'SEED-COMPLETE').exists(), f'seed {seed} incomplete'
    subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet', '--ignore-missing'], cwd=d, check=True)
    large = {f.strip(): h for h, f in (l.split(None, 1) for l in (arch / 'LARGE.sha256').read_text().splitlines())}
    for h, f in (l.split(None, 1) for l in (d / 'OUTPUTS.sha256').read_text().splitlines()):
        f = f.strip()
        if not (d / f).exists():
            assert large.get(f'output/train2d_s{seed}/{f}') == h, f'{f}: neither local nor archived with this sha256'
    job = json.loads((arch / 'JOBSPEC.json').read_text()).get('attempt')
    jid = sorted(p.stem for p in (arch / 'logs').glob('*.out'))[-1]
    m = write_bank(d, HERE / 'inputs' / 'banks' / f's{label}',
                   dict(label=label, arm=label, seed=int(seed), job=job, job_id=jid,
                        source_ckpt=str((d / CK).relative_to(arch)), source_rotation=str((d / ROT).relative_to(arch))))
    print(label, m['ckpt_sha256'][:12], m['rotation_sha256'][:12])


if __name__ == '__main__':
    main()
