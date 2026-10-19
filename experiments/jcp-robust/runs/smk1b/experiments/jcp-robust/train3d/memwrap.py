"""Run a script unchanged (runpy, as `python script args`) and then record the process's device peak memory.

    python memwrap.py <record.json> <script.py> [script args ...]

Added for the jcp-robust 3D memory pilot (DESIGN.md 3.3, 6) and used for production too, so every seed records the
same numbers. train2w.py itself is byte-identical and is not edited: runpy.run_path executes it as __main__ with its
own __file__, so its relative sys.path line and every code path are those of a direct `python train2w.py` call.
Recorded: jax.devices()[0].memory_stats() (peak_bytes_in_use, bytes_limit, ...) and this process's ru_maxrss.
Host MaxRSS for the whole job comes from `sacct -j <id> --format=MaxRSS` (the authoritative number, DESIGN 3.3).
"""
import json
import resource
import runpy
import sys
import time
import traceback
from pathlib import Path

rec_path, script = Path(sys.argv[1]), sys.argv[2]
sys.argv = [script] + sys.argv[3:]
t0 = time.time()
ok, err = False, None
try:
    runpy.run_path(script, run_name='__main__')
    ok = True
except BaseException as e:                       # record, then re-raise (exit status preserved)
    err = ''.join(traceback.format_exception_only(type(e), e)).strip()
    raise
finally:
    rec = dict(script=script, argv=sys.argv[1:], ok=ok, error=err, seconds=time.time() - t0,
               ru_maxrss_kB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    try:
        import jax
        d = jax.devices()[0]
        st = d.memory_stats() or {}
        rec.update(device=str(getattr(d, 'device_kind', d)), backend=jax.default_backend(),
                   memory_stats={k: int(v) for k, v in st.items() if isinstance(v, (int, float))},
                   peak_GiB=(st.get('peak_bytes_in_use', 0) / 2 ** 30) if st else None)
    except Exception as e:                        # CPU backends may return None; never mask the run's own error
        rec['memory_stats_error'] = repr(e)
    rec_path.write_text(json.dumps(rec, indent=1) + '\n')
    print(f"MEMWRAP device peak {rec.get('peak_GiB')} GiB, process maxrss {rec['ru_maxrss_kB'] / 2 ** 20:.1f} GiB",
          flush=True)
