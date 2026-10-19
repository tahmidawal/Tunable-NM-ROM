"""jcp-robust: register a retrained 3D W1024-recipe bank of job e2t3d<seed> as inputs/banks3d/s<seed>/ (DESIGN 3.3, A9.2,
A11; job audit 3D-1 #10). CPU only.

bank.pkl (> 100 MB) stays in the cluster archive (brief section 16); its sha256 is taken from the job's OUTPUTS.sha256 and
cross-checked against the archive's LARGE.sha256. BANK.json records bank_sha256, archive_path (consumed by
make_configs_e2_3d.bank_of -> config bank_registry + cluster_inputs, and by make_jobs.e2run3d_jobs -> job-spec
cluster_inputs), the run record (selected step, device peak, process MaxRSS) and the A9.2 memory-gate inputs.
With --fetch the bank is also copied to inputs/banks3d/s<seed>/bank.pkl (git-untracked; sha256-verified) for local
bank-aware scoring (e2score3.py). Refuses a seed whose SEED-COMPLETE is missing or whose checksums do not verify.

    python register_banks3d.py <seed> [--fetch]
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = '/cluster/tufts/paralab/tawal01/burgers-archive/jcp-robust'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 24), b''):
            h.update(blk)
    return h.hexdigest()


def main():
    s = sys.argv[1]
    att = f'e2t3d{s}'
    arch = HERE / 'runs' / att / 'archive'
    rel = f'output/train3d_s{s}'
    d = arch / rel
    assert (d / 'SEED-COMPLETE').exists(), f'seed {s} incomplete'
    lines = dict(reversed(l.split(None, 1)) for l in (d / 'OUTPUTS.sha256').read_text().splitlines())
    lines = {k.strip(): v for k, v in lines.items()}
    bank_sha = lines['bank.pkl']
    chk = subprocess.run(['sha256sum', '-c', 'OUTPUTS.sha256', '--quiet', '--ignore-missing'], cwd=d)
    assert chk.returncode == 0, 'OUTPUTS.sha256 does not verify'
    large = dict(reversed(l.split(None, 1)) for l in (arch / 'LARGE.sha256').read_text().splitlines())
    large = {k.strip(): v for k, v in large.items()}
    assert large.get(f'{rel}/bank.pkl') == bank_sha, 'archive LARGE.sha256 disagrees with OUTPUTS.sha256 for bank.pkl'
    rec = json.loads((d / 'run_record.json').read_text())
    out = HERE / 'inputs' / 'banks3d' / f's{s}'
    out.mkdir(parents=True, exist_ok=True)
    jid = sorted(p.stem for p in (arch / 'logs').glob('*.out'))[-1] if (arch / 'logs').exists() else None
    meta = dict(seed=int(s), job=att, job_id=jid, bank_sha256=bank_sha, archive_path=f'{ARCHIVE}/{att}/{rel}/bank.pkl',
                run_record=rec, training_json_sha256=sha(d / 'training.json'))
    if '--fetch' in sys.argv:
        dst = out / 'bank.pkl'
        subprocess.run(['scp', '-q', f'tufts-login:{meta["archive_path"]}', str(dst)], check=True)
        assert sha(dst) == bank_sha, 'fetched bank does not verify'
        meta['local_copy'] = 'inputs/banks3d/s%s/bank.pkl (untracked; sha256 above)' % s
    (out / 'BANK.json').write_text(json.dumps(meta, indent=1, default=str) + '\n')
    print(s, bank_sha[:12], meta['archive_path'])


if __name__ == '__main__':
    main()
