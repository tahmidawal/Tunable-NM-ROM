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


def check_dir(path):
    """A directory about to be created: every existing ancestor resolves inside the allowed root (no symlink escape)."""
    p = Path(path)
    for q in [p] + list(p.parents):
        if q.exists() or q.is_symlink():
            check_write(Path(os.path.realpath(str(q))) / '.probe')
            break
    check_write(p / '.probe')
    return p


def _lane_sub(lane_dir, sub):
    """<resolved lane>/<sub>, refusing a symlinked <sub> (its target would otherwise become permitted)."""
    lane = Path(os.path.realpath(str(lane_dir)))
    d = lane / sub
    if d.is_symlink():
        raise BoundaryError(f'{d} is a symlink')
    return str(d)


def check_admin(path, lane_dir):
    """Administrative files (staging, ledger, locks): only under the resolved lane's runs/ or checks/."""
    r = os.path.realpath(str(path))
    ok = any(r.startswith(_lane_sub(lane_dir, sub) + '/') for sub in ('runs', 'checks'))
    if not ok or any(f in r + '/' for f in FORBIDDEN):
        raise BoundaryError(f'refusing administrative path {r}')
    return Path(path)


def check_local(path, lane_dir):
    """Local (non-Slurm) acceptance / control outputs: only the resolved lane's checks/ directory or /tmp/."""
    r = os.path.realpath(str(path))
    ok = r.startswith(_lane_sub(lane_dir, 'checks') + '/') or r.startswith('/tmp/')
    if os.environ.get('SLURM_JOB_ID'):
        ok = (r + '/').startswith(LANE_NS)
    if not ok or any(f in r + '/' for f in FORBIDDEN):
        raise BoundaryError(f'refusing local output {r}')
    return Path(path)


def check_read_source(path):
    """A source file read by a slimming / verification step: resolved inside the allowed root."""
    r = os.path.realpath(str(path))
    if any(f in r + '/' for f in FORBIDDEN) or not (r + '/').startswith(allowed_root()):
        raise BoundaryError(f'refusing to read {r}')
    return Path(path)


def check_relative(src):
    """A configured relative source path: no absolute path, no '..' component."""
    p = Path(src)
    if p.is_absolute() or '..' in p.parts:
        raise BoundaryError(f'source must be a plain relative path: {src}')
    return p
