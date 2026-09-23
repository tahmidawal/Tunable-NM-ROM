"""Write `specs/ladder01.json` from the training-bank indices `gen01` actually produced.

Two things this exists to prevent, both raised by the design audit (B2, B4):

* **A missing rung must degrade, not void.** `audit.py` asserts every arm in the spec is
  present, so naming `index-04608.json` in the spec before knowing the generation reached
  4608 cases would let a short generation fail the whole ladder audit rather than cost it one
  rung. The rungs are therefore chosen from the indices `gen01` reports, after it has run.
* **A data ladder must hold the schedule constant in optimisation steps, not epochs.** Both
  of `train.py`'s patiences count epochs, and an epoch is `ceil(N/8)` gradient steps, so a
  36x change in N silently changes the learning-rate schedule and the early-stopping rule by
  the same factor. Each rung therefore carries a `patience` and a `plateau_patience` scaled
  so that `patience * steps_per_epoch` is the published 128-case value in STEPS.

    python make_ladder_spec.py --cache-json runs/gen01/archive/gen01/out/cache.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = 8
BASE_CASES = 128                       # the published training-set size
BASE_PATIENCE = 250                    # published early-stopping patience, in epochs
BASE_PLATEAU = 20                      # published ReduceLROnPlateau patience, in epochs
FAMILIES = (('fno', 'code/configs/fno/large.json'),
            ('unet', 'code/configs/unet/medium.json'),
            ('tsol', 'code/configs/transolver/small.json'))


def steps_per_epoch(cases):
    return -(-cases // BATCH)


def step_matched(cases):
    """Patiences in epochs that equal the published patiences in gradient steps."""
    per = steps_per_epoch(cases)
    base = steps_per_epoch(BASE_CASES)
    return dict(patience=max(1, -(-BASE_PATIENCE * base // per)),
                plateau_patience=max(1, -(-BASE_PLATEAU * base // per)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-json', required=True, type=Path,
                        help="gen01's out/cache.json, which lists the indices it wrote")
    parser.add_argument('--out', type=Path, default=HERE / 'specs/ladder01.json')
    parser.add_argument('--rungs', type=int, nargs='+', default=[128, 512, 2048, 4608])
    parser.add_argument('--per-arm-seconds', type=float, default=3000.)
    args = parser.parse_args()

    cache = json.loads(args.cache_json.read_text())
    available = {}
    for name in cache['bank_index_sha256']:
        available[int(name.removeprefix('index-').removesuffix('.json'))] = name
    rungs = [n for n in sorted(args.rungs) if n in available]
    dropped = [n for n in sorted(args.rungs) if n not in available]
    # If the top rung was not reached, fall back to the largest index that WAS written, so
    # the ladder still has a top rung and the report can name the shortfall exactly.
    if max(available) not in rungs:
        rungs.append(max(available))
    assert len(rungs) >= 2, f'a ladder needs at least two rungs; available {sorted(available)}'

    arms = []
    for family, config in FAMILIES:
        for n in rungs:
            arms.append(dict(name=f'{family}-n{n:05d}', config=config,
                             train_index=f'data/trainbig/{available[n]}',
                             override=step_matched(n)))
    # Control G2: the published configuration, the same budget and the same 128 PHYSICAL
    # cases at the PINNED 4096/1.5625e-4 fidelity. Its only difference from `unet-n00128` is
    # the fidelity of the training targets, so their difference is the fidelity effect at
    # fixed data size (DESIGN 3.2).
    arms.append(dict(name='unet-pinned128', config='code/configs/unet/medium.json',
                     train_index='data/train/index.json', override=step_matched(BASE_CASES)))
    # Does the budget bind at the top rung? Same rung, three times the wall.
    top = max(rungs)
    arms.append(dict(name=f'unet-n{top:05d}-long', config='code/configs/unet/medium.json',
                     train_index=f'data/trainbig/{available[top]}',
                     override=step_matched(top), seconds=3 * args.per_arm_seconds))

    training = sum(a.get('seconds', args.per_arm_seconds) for a in arms)
    spec = dict(
        job_name='opstune_ladder01', time='18:00:00', pde='burgers', prefix='lad',
        family='unet', smoke_families=['fno', 'unet', 'transolver'], arms=arms,
        refine_learning_rate=None, per_arm_seconds=args.per_arm_seconds,
        # Generous slack over the training total: each 4608-case arm re-reads its whole
        # training set several times inside dataset.load_pair (checksum, schema validation and
        # a content-equality pass), which is minutes of single-threaded I/O per arm and is NOT
        # charged to the arm's own wall budget but IS charged to the global one (audit M12).
        global_seconds=training + 14000., reserve_seconds=2500.,
        data_source='/cluster/tufts/paralab/tawal01/opstune_grid_20260922/cache',
        data_dirs=['train', 'validation', 'refinement', 'trainbig'],
        train_index='data/train/index.json',
        disjointness_indices=[f'data/trainbig/{available[top]}'],
        rungs=rungs, rungs_requested=sorted(args.rungs), rungs_unavailable=dropped,
        steps_per_epoch={str(n): steps_per_epoch(n) for n in rungs},
        step_matched_patience={str(n): step_matched(n) for n in rungs},
        design='Error versus training-set size for all three families, using each family '
               'PUBLISHED reference configuration unchanged, so the only thing varying along '
               'a ladder is the number of training cases. Every rung draws from the same '
               'cheap-fidelity bank, so target fidelity is constant along a ladder; '
               'unet-pinned128 is control G2 and varies only the fidelity at fixed size. '
               'Equal 3000 s wall per rung, the published per-arm budget, so the n=128 rung '
               'is comparable to the published arms as well as to its own ladder. Both '
               'patiences are scaled per rung so the learning-rate schedule and the '
               'early-stopping rule are constant in GRADIENT STEPS rather than in epochs, '
               'without which the top rung would finish at its initial learning rate while '
               'the bottom rung had annealed to the 1e-5 floor.')
    args.out.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(dict(out=str(args.out), rungs=rungs, unavailable=dropped, arms=len(arms),
                          training_seconds=training, global_seconds=spec['global_seconds'],
                          step_matched_patience=spec['step_matched_patience']), indent=2))


if __name__ == '__main__':
    main()
