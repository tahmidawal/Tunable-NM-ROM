"""jcp-references: draw the SECRET seeds of the new sealed / OOD cohorts and the custody key (DESIGN A1.11, A2.9).

Run ONCE on the cluster login node (no GPU, NumPy only):
    python make_seeds.py --custody /cluster/tufts/paralab/tawal01/jcp-ref/custody --sealed /cluster/tufts/paralab/tawal01/jcp-ref/sealed

- key: 32 random bytes as one 64-char hex line -> <custody>/sealed.key (mode 0400); refuses to overwrite.
- seeds: secrets.randbits(31) per new cohort; a draw is rejected and redrawn ONLY if it collides with a historical
  seed, a public check seed, or another new seed (the redraw count is recorded); never replaced for any other reason.
- writes <sealed>/SEEDS.json.enc (AES-256, plaintext never on disk) and prints ONLY the plaintext and ciphertext
  sha256 and the redraw count (to be committed in checks/sealed_seeds.json). The seeds themselves are never printed.
"""
import argparse
import json
import os
import secrets
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sealio as SI  # noqa: E402

HISTORICAL = {7090702, 911702, 20260927, 20260916, 923651, 923801, 923901}
PUBLIC = {5550001, 5550002}
NEW = ['2d/sealed128', '2d/ood32', '3d/sealed64', '3d/ood16']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--custody', required=True)
    ap.add_argument('--sealed', required=True)
    a = ap.parse_args()
    cust, sealed = Path(a.custody), Path(a.sealed)
    cust.mkdir(parents=True, exist_ok=True)
    sealed.mkdir(parents=True, exist_ok=True)
    os.chmod(cust, 0o700)
    key = cust / 'sealed.key'
    out = sealed / 'SEEDS.json.enc'
    if key.exists() or out.exists():
        raise SystemExit('refusing to overwrite an existing key or seed file')
    fd = os.open(key, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    os.write(fd, (secrets.token_bytes(32).hex() + '\n').encode())
    os.close(fd)
    seeds, redraws, used = {}, 0, set(HISTORICAL | PUBLIC)
    for name in NEW:
        while True:
            s = secrets.randbits(31)
            if s not in used:
                break
            redraws += 1
        used.add(s)
        seeds[name] = s
    data = json.dumps(seeds, sort_keys=True).encode()
    blob = SI.encrypt(data, key)
    assert SI.decrypt(blob, key) == data
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    os.write(fd, blob)
    os.close(fd)
    print(json.dumps(dict(seeds_plain_sha256=SI.sha(data), seeds_cipher_sha256=SI.sha(blob), redraws=redraws,
                          cohorts=NEW, key_path=str(key), seeds_path=str(out))))


if __name__ == '__main__':
    main()
