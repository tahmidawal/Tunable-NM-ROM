"""A12 (COMMON-BRIEF 16): fetch the offloaded jcp-qualify archive files that G-B3 / G-L1 compare against -- only the
R_R<r>_<form>_<scheme>_<factor>.npz files of the members configured in this lane's 2D b-job configs for one mesh --
from /cluster/tufts/paralab/tawal01/burgers-archive/jcp-qualify/<run>/ into a scratch cache that mirrors that layout.
ana2d.archive_f257 reads the cache (env QUALIFY_CACHE) and verifies every file against jcp-qualify's committed
archive.sha256-listing. Delete the cache after the analysis of that mesh.

    python fetch_qualify.py --mesh 1024 --cache <scratch dir>
"""
import argparse
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
QRUNS = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-qualify/experiments/jcp-qualify/runs')
REMOTE = '/cluster/tufts/paralab/tawal01/burgers-archive/jcp-qualify'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--cache', required=True)
    a = ap.parse_args()
    names = set()
    for cf in sorted((HERE / 'configs').glob(f'b{a.mesh}*.json')):
        for m in json.loads(cf.read_text()).get('members', []):
            names.add(f"R_R{int(m['Rp'])}_{m['form']}_{m['scheme']}_{float(m['factor']):g}.npz")
    want = []
    for lst in sorted(QRUNS.glob(f'*{a.mesh}/archive.sha256-listing')):
        run = lst.parent.name
        for line in lst.read_text().splitlines():
            if line.strip():
                rel = line.split(None, 1)[1].strip().lstrip('./')
                if Path(rel).name in names:
                    want.append(f'{run}/{rel}')
    files = Path(a.cache) / 'files.txt'
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    files.write_text('\n'.join(want) + '\n')
    subprocess.run(['rsync', '-a', '--files-from', str(files), f'tufts-login:{REMOTE}/', a.cache + '/'], check=True)
    print('fetched', len(want), 'files for', len(names), 'members at mesh', a.mesh)


if __name__ == '__main__':
    main()
