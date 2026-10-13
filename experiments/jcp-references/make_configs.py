"""jcp-references: write every job config (one file per job; one cluster directory per job). Re-runnable; the set of
configs is committed. Sealed configs carry only the cluster paths of the custody key and the encrypted seed file and
the committed hashes (checks/sealed_seeds.json); never a seed."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
C = HERE / 'configs'
NS = '/cluster/tufts/paralab/tawal01/jcp-ref'


def w(name, cfg):
    (C / f'{name}.json').write_text(json.dumps(cfg, indent=1) + '\n')


def chunks(n, k):
    return [list(range(i, min(n, i + k))) for i in range(0, n, k)]


C.mkdir(exist_ok=True)
w('gates', dict(mode='gates', dim=2))
w('calib2d', dict(mode='calib', dim=2))
w('calib3d', dict(mode='calib', dim=3))
FD = {2: dict(dev6=(6, 6), val32=(32, 16), test64=(64, 16)), 3: dict(dev4=(4, 4), val64=(64, 32), test32=(32, 32))}
for d, cohs in FD.items():
    for coh, (n, k) in cohs.items():
        for i, cs in enumerate(chunks(n, k)):
            w(f'fd{d}d{coh}p{i}', dict(mode='fd', dim=d, cohort=coh, cases=cs, kinds=['ST', 'S']))
w('ps2d', dict(mode='ps', dim=2, cohort='val32', cases=[0, 1, 2, 3, 4, 5, 19, 26], attempt=0))
w('ps3d', dict(mode='ps', dim=3, cohort='val64', cases=[0, 1, 2, 3, 5, 33, 43, 47], attempt=0))
seeds = HERE / 'checks' / 'sealed_seeds.json'
if seeds.exists():
    s = json.loads(seeds.read_text())
    SEAL = {2: dict(sealed128=(128, 32), ood32=(32, 32)), 3: dict(sealed64=(64, 32), ood16=(16, 16))}
    for d, cohs in SEAL.items():
        for coh, (n, k) in cohs.items():
            for i, cs in enumerate(chunks(n, k)):
                w(f'fd{d}d{coh}p{i}', dict(mode='fd', dim=d, cohort=coh, cases=cs, kinds=['ST', 'S'],
                                          key=f'{NS}/custody/sealed.key', seeds_file=f'{NS}/sealed/SEEDS.json.enc',
                                          seeds_plain_sha256=s['seeds_plain_sha256'],
                                          seeds_cipher_sha256=s['seeds_cipher_sha256']))
print(sorted(p.name for p in C.glob('*.json')))
