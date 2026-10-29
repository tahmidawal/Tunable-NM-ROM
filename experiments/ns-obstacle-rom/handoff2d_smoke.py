"""Release rejection only: no fabricated full admission or native/data execution."""
import json,tempfile,subprocess,sys
from pathlib import Path
from unittest.mock import patch
import handoff2d as h


def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);plan=root/'plan.json';release=root/'release.json'
        plan.write_text(json.dumps({'schema':'ns-obstacle-fd1-handoff-plan-v1','split':'validation','cases':[]}))
        release.write_text(json.dumps({'schema':'ns-obstacle-fd1-handoff-release-v1','source_sha256':'wrong',
            'plan_sha256':h.fe.sha_file(plan),'split':'validation','single_execution':True}))
        with patch.object(h,'read_bound',side_effect=AssertionError('unreleased source accessed')):
            try:h.verify(plan,release)
            except ValueError:pass
            else:raise AssertionError('mismatched release accepted')
    code="import handoff_primary_worker as w,sys,json; names=json.loads((w.m.NATIVE/'SOURCE.json').read_text())['files']; assert not any(name[:-3] in sys.modules for name in names)"
    subprocess.run([sys.executable,'-c',code],cwd=Path(h.__file__).parent,check=True)
    print('PASS fresh native-worker namespace; handoff source mismatch rejected before input access; no admission/data synthesized')


if __name__=='__main__':main()
