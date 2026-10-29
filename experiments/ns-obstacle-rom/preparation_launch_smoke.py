"""No native work: isolated source closure and rejected launcher provenance."""
from pathlib import Path
import json, os, shutil, subprocess, sys, tempfile
import build_preparation_launch as builder
import preparation_launch2d as launcher


def run():
    with tempfile.TemporaryDirectory(prefix='preparation-source-', dir=builder.ROOT/'staging') as temporary:
        root = Path(temporary)
        names = builder.source_closure()
        for name in names:
            destination = root/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(builder.ROOT/name, destination)
        command = 'import prepare2d,sys; print(prepare2d.preparation_digest()); assert "ngsolve" not in sys.modules and "jax" not in sys.modules'
        result = subprocess.run([sys.executable, '-c', command], cwd=root,
            env=dict(os.environ, PYTHONPATH=str(root)), capture_output=True, text=True, check=True)
        assert result.stdout.strip() == builder.preparation_digest()
        (root/'bad-release.json').write_text('{}')
        (root/'bad-inventory.json').write_text('{}')
        try:
            launcher.verify(root/'bad-release.json', root/'bad-inventory.json')
        except ValueError:
            pass
        else:
            raise AssertionError('unbound release accepted')
    return {'source_closure_files':len(names), 'isolated_digest_equal':True,
            'no_native_or_jax_import':True, 'unbound_release_rejected':True,
            'actual_training_or_preparation_executed':False}


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
