"""Generate `config-g1.json`, `config-g2.json` and `config-s1.json`.

Every inherited constant (mesh, time step, cohort seeds, direction rule, solver contract,
timing protocol, FOM controls, the cohort/direction/reference hashes) is READ from the
parent lane's committed `config-r2.json` (job `qrg201`), which itself read them from the
`btq102` archive; nothing is typed. Only the cells and the fidelity expectations are new.

    python make_configs.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / 'experiments/q-ridge/config-r2.json'
K = 16

INHERIT = ('intervals', 'dt', 'eval_seed', 'eval_cases', 'eval_fresh_seed', 'eval_fresh_cases',
           'reference_mesh', 'reference_dt', 'train_seed', 'train_trajectories',
           'train_state_stride', 'snapshot_ntol', 'snapshot_ltol', 'residual_snapshots',
           'residual_starts', 'residual_budget', 'residual_seed', 'decoder_code_subsample',
           'gauss_jordan_max', 'cold_axis_points', 'strict', 'ic_gtol', 'inner_damping',
           'recon_starts', 'recon_budget', 'reps', 'order_seed', 'burn_seconds',
           'step_budget_provenance', 'expected_physical_sha256', 'expected_directions_sha256',
           'source_directions_sha256', 'source_reference_sha256', 'source_job_ids',
           'fom_settings', 'cohort_note')


def cell(q, M, *labels, linear=None):
    assert M > K + q, (q, M)
    d = dict(q=q, M=M, labels=list(labels))
    if linear:
        d['linear'] = linear
    return d


# The crossed grid. `fixed256` / `fixed1088`: the two fixed-M columns. `4x` / `8x` / `16x`
# / `2x`: the scheduled columns M = c (K+q). `bridge`: the cell (q_k, M_{k+1}) that makes
# the rung-by-rung decomposition of the scheduled 4x ladder exact (DESIGN.md §4.2).
G1 = [cell(0, 64, '4x'), cell(0, 128, '8x'), cell(0, 256, '16x', 'fixed256'),
      cell(0, 1088, 'fixed1088'),
      cell(16, 128, '4x'), cell(16, 192, 'bridge'), cell(16, 256, '8x', 'fixed256'),
      cell(16, 512, '16x'), cell(16, 1088, 'fixed1088'),
      cell(32, 192, '4x'), cell(32, 256, 'fixed256'), cell(32, 320, 'bridge'),
      cell(32, 384, '8x'), cell(32, 768, '16x'), cell(32, 1088, 'fixed1088'),
      # solver control: the same cell with the inner linear solve the q >= 64 rows use, so
      # the gj -> lu switch at K + q > 64 is measured on its own (DESIGN.md §3)
      cell(32, 1088, 'solver-control', linear='lu')]
G2 = [cell(0, 256, 'fixed256', 'anchor'), cell(0, 1088, 'fixed1088', 'anchor'),
      cell(64, 256, 'fixed256'), cell(64, 320, '4x'), cell(64, 576, 'bridge'),
      cell(64, 640, '8x'), cell(64, 1088, 'fixed1088'), cell(64, 1280, '16x'),
      cell(128, 256, 'fixed256'), cell(128, 576, '4x'), cell(128, 1088, 'fixed1088', 'bridge'),
      cell(128, 1152, '8x'),
      cell(256, 544, '2x'), cell(256, 1088, '4x', 'fixed1088'), cell(256, 2176, '8x')]
S1 = [cell(0, M, 'sat') for M in (64, 128, 256, 512, 1088, 2048, 4096)] + \
     [cell(64, M, 'sat') for M in (128, 256, 320, 512, 1088, 2048, 4096)]

# --- round 2 (DESIGN.md §A3) ------------------------------------------------------
# E1: the pure-rank ladder extended to q = R = 512, the whole bank, at fixed M = 1088 --
# every rung in ONE job, which is what makes the within-job cost ladder exist. The two
# extra q = 512 cells at 4(K+q) and 6(K+q) separate "the rank has run out" from "this cell
# is under-tested": M = 1088 is only 2.06 tests per unknown at q = 512, and q = 64 at 1.6
# tests per unknown was visibly starved (1.8032 % against 1.1255 % at 3.2x).
E1 = [cell(0, 64, 'anchor')] + [cell(q, 1088, 'fixed1088') for q in (0, 16, 32, 64, 128, 256, 512)] + \
     [cell(512, 2112, '4x'), cell(512, 3168, '6x')]
# E2: where does the best cell in the campaign stop improving? q = 256 against M from the
# 4(K+q) rung out to 24 tests per unknown, plus the two anchors.
E2 = [cell(0, 64, 'anchor'), cell(0, 1088, 'anchor')] + \
     [cell(256, M, 'sat256') for M in (1088, 2176, 3264, 4352, 6528)]


def exp(source, arm, tol=1e-9, cond=None, note=None):
    d = dict(source=source, arm=arm, tolerance=tol)
    if cond is not None:
        d['tolerance_if_directions_bitwise'] = cond
    if note:
        d['note'] = note
    return d


MINE = ('bqx101', 'bqx201', 'bqx301')
UNCOND = 'q = 0 carries no correction directions, so this reproduction is unconditional'
COND = ('directions are GPU-model dependent across jobs: 1e-9 if the directions hash is '
        'bitwise, 1e-3 otherwise (b-ladder-top convention); the achieved difference is reported')

EXPECT = {
    'q0_M64_dense': [exp('btq101', 'q0_m4_dense_base', note=UNCOND),
                     exp('cclad01', 'q0_m4_dense_block', note=UNCOND),
                     exp('qrg201', 'q0_m4_dense_l0', note=UNCOND),
                     exp('qtd02', 'q0_M64_dense', note=UNCOND)],
    'q0_M128_dense': [exp('qrg201', 'q0_m8_dense_l0', note=UNCOND)],
    'q0_M256_dense': [exp('cclad01', 'q0_m256_dense_block', note=UNCOND),
                      exp('qrg201', 'q0_m16_dense_l0', note=UNCOND),
                      exp('qtd02', 'q0_M256_dense', note=UNCOND)],
    'q16_M128_dense': [exp('cclad01', 'q16_m4_dense_block', 1e-3, 1e-9, COND),
                       exp('qrg201', 'q16_m4_dense_l0', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q16_M128_dense', 1e-3, 1e-9, COND)],
    'q16_M256_dense': [exp('cclad01', 'q16_m256_dense_block', 1e-3, 1e-9, COND),
                       exp('qrg201', 'q16_m8_dense_l0', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q16_M256_dense', 1e-3, 1e-9, COND)],
    'q16_M512_dense': [exp('qrg201', 'q16_m16_dense_l0', 1e-3, 1e-9, COND)],
    'q32_M192_dense': [exp('cclad01', 'q32_m4_dense_block', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q32_M192_dense', 1e-3, 1e-9, COND)],
    'q32_M256_dense': [exp('cclad01', 'q32_m256_dense_block', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q32_M256_dense', 1e-3, 1e-9, COND)],
    'q64_M256_dense': [exp('cclad01', 'q64_m256_dense_block', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q64_M256_dense', 1e-3, 1e-9, COND)],
    'q64_M320_dense': [exp('cclad01', 'q64_m4_dense_block', 1e-3, 1e-9, COND),
                       exp('qrg201', 'q64_m4_dense_l0', 1e-3, 1e-9, COND),
                       exp('qtd02', 'old_q64_M320_dense', 1e-3, 1e-9, COND)],
    'q64_M640_dense': [exp('qrg201', 'q64_m8_dense_l0', 1e-3, 1e-9, COND)],
    'q64_M1280_dense': [exp('qrg201', 'q64_m16_dense_l0', 1e-3, 1e-9, COND)],
    'q128_M256_dense': [exp('cclad01', 'q128_m256_dense_block', 1e-3, 1e-9, COND),
                        exp('qtd02', 'old_q128_M256_dense', 1e-3, 1e-9, COND)],
    'q128_M576_dense': [exp('cclad01', 'q128_m4_dense_block', 1e-3, 1e-9, COND),
                        exp('qtd02', 'old_q128_M576_dense', 1e-3, 1e-9, COND)],
    'q256_M544_dense': [exp('btq102', 'q256_m2_dense_base_b600', 1e-3, 1e-9, COND)],
    'q256_M1088_dense': [exp('qtd02', 'old_q256_M1088_dense', 1e-3, 1e-9, COND),
                         exp('cclad01', 'q256_m4_dense_block', 1e-3, 1e-9,
                             COND + '; cclad01 ran this rung at budget 180')],
}
# Round 2 gates against this lane's own round-1 jobs. The direction matrix is nondeterministic
# run to run even on one GPU model (bqx101/201/301 hash to three different values), so every
# q > 0 pair is judged at the loose tier and the ACHIEVED difference is the evidence; round 1
# came in at <= 9.2e-9 against four earlier jobs.
SELF = 'against this lane\'s own round-1 job; the direction matrix is run-to-run nondeterministic, so the loose tier applies and the achieved difference is what is reported'
# Applied ONLY to the round-2 jobs: config-g1/g2/s1.json are already staged, submitted and
# audited, and must regenerate byte-identically.
EXPECT_R2 = {k: list(v) for k, v in EXPECT.items()}
for _arm, _q in (('q0_M64_dense', 0), ('q0_M1088_dense', 0), ('q16_M1088_dense', 16),
                 ('q32_M1088_dense', 32), ('q64_M1088_dense', 64), ('q128_M1088_dense', 128),
                 ('q256_M1088_dense', 256), ('q256_M2176_dense', 256)):
    for _src in MINE:
        EXPECT_R2.setdefault(_arm, [])
        if _q == 0:
            EXPECT_R2[_arm].append(exp(_src, _arm, note=UNCOND + '; ' + SELF))
        else:
            EXPECT_R2[_arm].append(exp(_src, _arm, 1e-3, 1e-9, SELF))


def main():
    base = json.loads(PARENT.read_text())
    shared = {k: base[k] for k in INHERIT}
    assert shared['strict']['step_budget'] == 600 and shared['reps'] == 3
    shared['parent_config'] = str(PARENT.relative_to(ROOT))
    shared['parent_attempt'] = base['attempt']
    shared['quadrature'] = 'dense'
    # The parent's expected_directions_sha256 (qlad01's) matches no comparator this lane can
    # reproduce (the matrix is GPU-model dependent); no expectation is declared, the sources'
    # hashes are carried as probes.
    shared['expected_directions_sha256'] = None
    shared['K'] = K
    jobs = {
        'g1': dict(attempt='bqx101', question='G1',
                   purpose=('the crossed (q, M) grid, rows q in {0, 16, 32}: fixed M in {256, 1088}, '
                            'scheduled M in {4, 8, 16}(K+q), and the bridge cells (q_k, M_{k+1}); '
                            'dense quadrature, budget 600'),
                   cells=G1, q_ladder=[0, 16, 32]),
        'g2': dict(attempt='bqx201', question='G2',
                   purpose=('the crossed (q, M) grid, rows q in {64, 128, 256}: fixed M in {256, 1088}, '
                            'scheduled M in {4, 8}(K+q) plus 16(K+q) at q = 64 and 2(K+q) at q = 256, '
                            'the bridge cells, and two q = 0 anchor arms that are in-job fidelity '
                            'gates; dense quadrature, budget 600'),
                   cells=G2, q_ladder=[0, 64, 128, 256]),
        'e1': dict(attempt='bqx401', question='E1',
                   purpose=('round 2: the pure-rank ladder extended to q = R = 512 at fixed M = 1088, every '
                            'rung in one job so the within-job cost ladder covers all of it, plus q = 512 at '
                            '4(K+q) and 6(K+q) to tell a rank limit from an under-tested cell'),
                   cells=E1, q_ladder=[0, 16, 32, 64, 128, 256, 512]),
        'e2': dict(attempt='bqx501', question='E2',
                   purpose=('round 2: where the campaign\'s best cell stops improving — q = 256 against '
                            'M in {1088, 2176, 3264, 4352, 6528}, i.e. 4 to 24 tests per unknown'),
                   cells=E2, q_ladder=[0, 256]),
        's1': dict(attempt='bqx301', question='S1',
                   purpose=('the saturation sweep: M in {64, ..., 4096} at q = 0 and {128, ..., 4096} at q = 64 '
                            '(with the row\'s own 4(K+q) = 320 cell for the within-job cost ratio), '
                            'dense quadrature, budget 600; where does the M effect stop and what '
                            'does it cost'),
                   cells=S1, q_ladder=[0, 64]),
    }
    for tag, job in jobs.items():
        names = [f"q{c['q']}_M{c['M']}_dense" + (f"_{c['linear']}" if c.get('linear') else '')
                 for c in job['cells']]
        assert len(names) == len(set(names)), names
        cfg = dict(shared)
        cfg.update(job)
        table = EXPECT_R2 if tag in ('e1', 'e2') else EXPECT
        cfg['expectations'] = {n: table[n] for n in names if n in table}
        cfg['gate_note'] = ('every q = 0 expectation is unconditional at 1e-9; the audit gates '
                            'each (arm, source) pair separately and reports the achieved '
                            'relative difference on both metrics')
        path = HERE / f'config-{tag}.json'
        path.write_text(json.dumps(cfg, indent=2) + '\n')
        print(path.name, job['attempt'], 'cells', len(job['cells']),
              'unique M', sorted({c['M'] for c in job['cells']}),
              'gated arms', len(cfg['expectations']))


if __name__ == '__main__':
    main()
