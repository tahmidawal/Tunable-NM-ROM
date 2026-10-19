"""jcp-regimes write / read boundary (DESIGN A5, code audits 1-2). Every lane driver that writes calls check_write on the
exact file path just before writing it; the resolved path (symlinks followed, including descendant links) must lie under
the lane's cluster namespace inside a Slurm job, or under /tmp/ for a local CPU functional run, and never inside the
E0 lane or any jcp-ref directory."""
import os
from pathlib import Path

LANE_NS = '/cluster/tufts/paralab/tawal01/jcp-regimes/'
FORBIDDEN = ('/jcp-ref/', 'jcp-references', '/sealed/', '/custody/')


class BoundaryError(SystemExit):
    pass


def allowed_root():
    return LANE_NS if os.environ.get('SLURM_JOB_ID') else '/tmp/'


def check_write(path):
    r = os.path.realpath(str(path))
    parent = os.path.realpath(os.path.dirname(r) or '.')
    for x in (r, parent):
        if any(f in x + '/' for f in FORBIDDEN):
            raise BoundaryError(f'refusing to write into a forbidden location: {x}')
        if not (x + '/').startswith(allowed_root()):
            raise BoundaryError(f'refusing to write outside {allowed_root()}: {x}')
    return Path(path)


def check_relative(src):
    """A configured relative source path: no absolute path, no '..' component."""
    p = Path(src)
    if p.is_absolute() or '..' in p.parts:
        raise BoundaryError(f'source must be a plain relative path: {src}')
    return p
