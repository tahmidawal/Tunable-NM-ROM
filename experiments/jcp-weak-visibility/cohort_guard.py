"""Portable source/seed/count cohort validation; never alters generated parameters."""
import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def extract(path, name, namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def regenerate(cfg, draw):
    here = Path(__file__).resolve().parent
    assert cfg['seed_first_draw'] == 7090702 and cfg['seed_first_count'] == 4
    physical = draw(cfg['seed_first_draw'], cfg['seed_first_count'])
    record = dict(phase=cfg.get('cohort_phase', 'initial-dev2'), first_draw_seed=cfg['seed_first_draw'],
                  first_draw_count=cfg['seed_first_count'], selected_case_ids=cfg['case_ids'])
    if cfg.get('cohort_phase') == 'remaining-dev6':
        assert cfg['case_ids'] == [2, 3, 4, 5]
        assert cfg['seed_second_draw'] == 911702 and cfg['seed_second_count'] == 2
        physical = np.concatenate((physical, draw(cfg['seed_second_draw'], cfg['seed_second_count'])))
        # Execute only the two original metadata functions, never module imports or other cohorts.
        old_draw = extract(here/'vendor/legacy_engines.py', 'params_draw', {'np':np})
        old_cohort = extract(here/'vendor/legacy_qcore.py', 'cohort', {'np':np, 'e':SimpleNamespace(params_draw=old_draw)})
        canonical = old_cohort('dev6')
        expected = np.asarray(cfg['local_parameter_reference'], dtype=np.float64)
        assert sha(expected) == cfg['local_parameter_sha256']
        ulp = np.abs(physical.view(np.int64)-expected.view(np.int64))
        record.update(second_draw_seed=cfg['seed_second_draw'], second_draw_count=cfg['seed_second_count'],
            legacy_same_host_bitwise=bool(np.array_equal(physical, canonical)),
            canonical_same_host_sha256=sha(canonical), local_parameter_sha256=sha(expected),
            local_portability_ulp=ulp.tolist(), local_portability_max_ulp=int(ulp.max()),
            first_four_columns_exact=bool(np.array_equal(physical[:, :4], expected[:, :4])),
            physical_parameters=physical.tolist(), canonical_parameters=canonical.tolist(),
            guard_passed=bool(np.array_equal(physical, canonical) and np.array_equal(physical[:, :4], expected[:, :4]) and np.max(ulp[:, 4]) <= 4))
    else:
        assert cfg['case_ids'] == [0, 1]
        record['guard_passed'] = True
    record.update(parameter_sha256=sha(physical), selected_parameter_sha256=sha(physical[cfg['case_ids']]))
    return physical, record
