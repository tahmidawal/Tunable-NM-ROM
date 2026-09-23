"""DESIGN section 2's accounting, derived from pinned artefacts instead of typed.

Run locally; writes `reports/accounting.json`, which the report generator reads. Sources, all
hash-pinned here:

  * `checks/pinned/train-index.json` and `validation-index.json` -- byte copies of the pinned
    cache's own indices, whose sha256 are the `5333584b...` / `468b9e70...` literals every
    operator lane's audit asserts. Holding the files makes those literals checkable here.
  * `checks/pinned/refinement-gate-256.json` -- the output-256 slice of the calibration gate.
  * `reports/2026-09-10-historical-poisson-and-burgers-cost-audit.json` (repository root) --
    the NM-ROM checkpoint's own training configuration.

It also computes what no artefact records: how close the 32 validation cases sit to the
training set in parameter space as that set grows 36x, which is the thing "more data" actually
buys an operator on a five-parameter family (design audit M10).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
REPO = LANE.parents[1]
PINNED = LANE / 'checks/pinned'
# `main`'s reports/ is not checked out in this worktree, so the checkpoint's own configuration
# block is pinned here with the hash of the file it came from.
COST_AUDIT = PINNED / 'nmrom-checkpoint-config.json'
RUNGS = (128, 512, 2048, 4608)
# The five generation descriptors' declared ranges (engines.params_draw); nu is log-uniform.
RANGES = ((.15, .85), (.15, .85), (.05, .20), (.5, 2.), (math.log(.01), math.log(.1)))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalised(descriptors):
    """Each descriptor on [0, 1] over its own declared range; nu in log space, as it is drawn."""
    out = np.array(descriptors, dtype=np.float64).copy()
    out[:, 4] = np.log(out[:, 4])
    for column, (low, high) in enumerate(RANGES):
        out[:, column] = (out[:, column] - low) / (high - low)
    return out


def coverage(train, validation, rungs):
    """Nearest training neighbour of each validation case, per nested prefix."""
    rows = []
    for rung in rungs:
        if rung > len(train):
            continue
        distance = np.linalg.norm(validation[:, None] - train[None, :rung], axis=2).min(axis=1)
        spacing = np.linalg.norm(train[:rung, None] - train[None, :rung], axis=2)
        np.fill_diagonal(spacing, np.inf)
        rows.append(dict(cases=rung, nearest_mean=float(distance.mean()),
                         nearest_median=float(np.median(distance)), nearest_max=float(distance.max()),
                         train_nearest_neighbour_median=float(np.median(spacing.min(axis=1)))))
    return rows


def main():
    sys.path.insert(0, str(LANE))
    import data
    import refine
    refine.configure()
    import engines

    train = json.loads((PINNED / 'train-index.json').read_text())
    validation = json.loads((PINNED / 'validation-index.json').read_text())
    gate = json.loads((PINNED / 'refinement-gate-256.json').read_text())
    cost_audit = json.loads(COST_AUDIT.read_text())
    cost = cost_audit['current_burgers_checkpoint_cfg']

    seconds = np.array([r['reference']['wall_seconds_including_first_compile'] for r in train['records']])
    keys = ('cx', 'cy', 'width', 'amplitude', 'nu')
    train_descriptors = [[r['generation_descriptors'][k] for k in keys] for r in train['records']]
    validation_descriptors = [[r['generation_descriptors'][k] for k in keys] for r in validation['records']]

    # The extended training draws are reproducible without solving anything: the same case seeds
    # and the same params_draw the generator uses.
    extended = np.array([engines.params_draw(data.case_seed('train', i), 1)[0] for i in range(max(RUNGS))])
    if not np.allclose(extended[:len(train_descriptors)], np.array(train_descriptors), rtol=0, atol=0):
        raise RuntimeError('regenerated descriptors differ from the pinned index')

    candidates = {str(c['intervals']): c for c in gate['by_output_256']['candidates']}
    per_case = {str(c['intervals']): max(row['candidates'][i]['difference_from_anchor']
                                         for row in gate['by_output_256']['cases'])
                for i, c in enumerate(gate['by_output_256']['candidates'])}

    operator = dict(training_trajectories=train['count'], validation_trajectories=validation['count'],
                    supervised_output_fields=train['count'] * 5,
                    distinct_solution_states=train['count'] * 6,
                    train_index_sha256=sha256(PINNED / 'train-index.json'),
                    validation_index_sha256=sha256(PINNED / 'validation-index.json'),
                    protocol_sha256=train['protocol_sha256'],
                    protocol_in_use_sha256=data.sha(data.PROTOCOL_PATH),
                    reference_setting=train['reference_setting'],
                    generator_sources=train['provenance']['source_sha256'],
                    generating_job=train['provenance']['job_id'], gpu=train['provenance']['gpu'],
                    per_case_seconds=dict(mean=float(seconds.mean()), median=float(np.median(seconds)),
                                          minimum=float(seconds.min()), maximum=float(seconds.max()),
                                          total_hours=float(seconds.sum() / 3600)))
    parity = dict(
        target_trajectories=cost['hfit_n_traj'],
        trajectory_ratio=cost['hfit_n_traj'] / train['count'],
        supervised_output_fields_at_parity=cost['hfit_n_traj'] * 5,
        pinned_protocol_gpu_hours_at_parity=float(seconds.mean() * cost['hfit_n_traj'] / 3600),
        note='an extrapolation from this job\'s own recorded per-case wall time on one GPU type, '
             'not a proof about every possible generation strategy')
    nmrom = dict(bank_trajectories=cost['n_traj'], head_trajectories=cost['hfit_n_traj'],
                 head_extra_trajectories=cost['hfit_extra_traj'], steps_per_trajectory=cost['num_steps'],
                 states_per_trajectory=cost['num_steps'] + 1,
                 available_head_states=cost['hfit_n_traj'] * (cost['num_steps'] + 1),
                 bank_snapshot_cap=cost['max_snaps'],
                 retained_head_codes=131072,
                 retained_head_codes_source='LAB-LOG 2026-09-16 prose; the only record in this '
                                            'repository, and not regenerable from the cited JSON',
                 source=cost_audit['extracted_from'], source_sha256=cost_audit['source_sha256'],
                 extract_sha256=sha256(COST_AUDIT),
                 caution='these state counts are fitted latent codes for a reconstruction objective. '
                         'They are NOT comparable with an operator\'s supervised output fields and '
                         'no ratio between the two is reported.')
    out = dict(
        generated_by='reports/accounting.py', rungs=list(RUNGS), operator=operator, nmrom=nmrom,
        parity=parity,
        reference_candidates={key: dict(worst_empirical_margin=value['worst_empirical_margin'],
                                        worst_difference_from_anchor=per_case[key],
                                        work_proxy=value['work_proxy'], passing=value['passing'],
                                        dt=value['dt'])
                              for key, value in candidates.items()},
        reference_candidate_note='"margin" is the candidate\'s difference from the anchor PLUS the '
                                 'anchor\'s own space and time refinement differences (data.py '
                                 'evaluate_gate); "difference_from_anchor" is the direct discrepancy, '
                                 'worst over the 8 calibration cases, in the same fixed-initial metric '
                                 'the models are graded in',
        coverage=coverage(normalised(extended), normalised(validation_descriptors), RUNGS),
        coverage_note='normalised distance: each descriptor scaled to [0, 1] over its declared '
                      'params_draw range, nu in log space. Reported so the report can say what more '
                      'data buys on a five-parameter family; our own head was trained at 4608.')
    (HERE / 'accounting.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'coverage'}, indent=2)[:4000])
    print(json.dumps(out['coverage'], indent=2))


if __name__ == '__main__':
    main()
