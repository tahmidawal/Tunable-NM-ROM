"""jcp-robust: register the retrained 2D banks of job e2t2d as inputs/banks/s<seed>/ (DESIGN 3.2, A1.6, A2.10). CPU only.

For each seed: the emitted dense mid checkpoint (Stage B) and the head-based rotation are copied from the collected archive
(both < 100 MB, pulled locally), their sha256 recorded in BANK.json together with the job id, the training record (Stage A
steps, mid steps_done, whether the head-fit time cap fired -> recipe deviation, device) and the rotation diagnostics.
Refuses a seed whose SEED-COMPLETE is missing or whose OUTPUTS.sha256 does not verify.

    python register_banks.py <e2t2d archive dir> <seed> [<seed> ...]
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    arch = Path(sys.argv[1])
    job = json.loads((arch / 'JOBSPEC.json').read_text())['attempt']
    jid = sorted(p.stem for p in (arch / 'logs').glob('*.out'))[-1]
    for s in sys.argv[2:]:
        d = arch / 'output' / f'train2d_s{s}'
        assert (d / 'SEED-COMPLETE').exists(), f'seed {s} incomplete'
        # files > 100 MB stay in the cluster archive (brief 16): verify the rest locally and require every missing file
        # to be listed with the same sha256 in the archive's LARGE.sha256
        subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet', '--ignore-missing'], cwd=d, check=True)
        large = {f.strip(): h for h, f in (l.split(None, 1) for l in (arch / 'LARGE.sha256').read_text().splitlines())}
        for h, f in (l.split(None, 1) for l in (d / 'OUTPUTS.sha256').read_text().splitlines()):
            f = f.strip()
            if not (d / f).exists():
                assert large.get(f'output/train2d_s{s}/{f}') == h, f'{f}: neither local nor archived with this sha256'
        ck = d / 'stageB' / 'sep_hfit_dense_mid_N256_dense.pkl'
        rot = d / 'rotation' / 'rotation_R512.npz'
        rec = json.loads((d / 'run_record.json').read_text()) if (d / 'run_record.json').exists() else {}
        out = HERE / 'inputs' / 'banks' / f's{s}'
        out.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(ck, out / 'ckpt.pkl')
        shutil.copyfile(rot, out / 'rotation_R512.npz')
        meta = dict(seed=int(s), job=job, job_id=jid, ckpt_sha256=sha(out / 'ckpt.pkl'), rotation_sha256=sha(out / 'rotation_R512.npz'),
                    source_ckpt=str(ck.relative_to(arch)), source_rotation=str(rot.relative_to(arch)), run_record=rec,
                    rotation_json=json.loads((d / 'rotation' / 'rotation_R512.json').read_text()))
        assert meta['ckpt_sha256'] == sha(ck) and meta['rotation_sha256'] == sha(rot)
        (out / 'BANK.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')
        print(s, meta['ckpt_sha256'][:12], meta['rotation_sha256'][:12])


if __name__ == '__main__':
    main()
