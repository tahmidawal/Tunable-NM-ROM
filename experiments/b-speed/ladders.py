"""The pre-registered arm list. Declared before any run; see DESIGN.md section 4.

`class`:
  bitwise        the identical floating-point expressions are evaluated, so the output
                 field sha256 must equal the incumbent's
  reassociation  algebraically exact, different rounding; fields must agree to 1e-12
                 and the integer iteration and exit-reason vectors must be identical
  labelled       deliberately not a parity arm (f32 outputs); never eligible for the
                 headline factor
"""
from __future__ import annotations

import fast as F


def _a(cls, note, **kw):
    return dict(opts=F.opts(**kw), cls=cls, note=note)


ISOLATED = {
    'o_none': _a('bitwise', 'the optimisation harness with every switch off; must be '
                            'bitwise identical to arms.make_query'),
    'o_fuse': _a('bitwise', 'one-pass residual+Jacobian via jax.linearize, no lax.cond, '
                            'J^T r reused for the stationarity ratio', fuse=1),
    'o_hoist': _a('bitwise', 'nu*Lambda and 1+dt*nu*Lambda computed once per query', hoist=1),
    'o_share': _a('bitwise', 'the step projection A h(z) and the probe residual at z share '
                             'one evaluation', share=1),
    'o_unroll5': _a('bitwise', 'the 50-step scan unrolled 5x (loop control only)', unroll=5),
    'o_probe': _a('reassociation', 'the two extrapolation probes as one batched evaluation',
                  probe=1),
    'o_lean': _a('reassociation', 'head output layer folded into the stencil bank and the '
                                  'test projection; residual rescaled', lean=1),
    'o_block2': _a('reassociation', 'block Gauss-Jordan, 2 unknowns per stage', block=2),
    'o_block4': _a('reassociation', 'block Gauss-Jordan, 4 unknowns per stage', block=4),
    'o_block8': _a('reassociation', 'block Gauss-Jordan, 8 unknowns per stage', block=8),
    'o_nodot': _a('reassociation', 'broadcast-reduce matvecs instead of cuBLAS calls on '
                                   'tiny shapes', nodot=1),
    'o_decfused': _a('reassociation', 'decode as one G [h(z_1)..h(z_6)] matmul instead of a '
                                      'vmap over six', decode='fused'),
    'o_decleaan': _a('reassociation', 'decode folded: (G Wm^T) f(z) + G b3, 144/512 of the '
                                      'bytes and FLOPs', decode='lean'),
}

CUMULATIVE = {
    'L1': _a('bitwise', 'fuse', fuse=1),
    'L2': _a('bitwise', 'fuse + hoist + share', fuse=1, hoist=1, share=1),
    'L3': _a('bitwise', 'L2 + scan unroll 5', fuse=1, hoist=1, share=1, unroll=5),
    'L4': _a('reassociation', 'L3 + lean fold', fuse=1, hoist=1, share=1, unroll=5, lean=1),
    'L5': _a('reassociation', 'L4 + block-4 solve', fuse=1, hoist=1, share=1, unroll=5,
             lean=1, block=4),
    'L6': _a('reassociation', 'L5 + batched probe', fuse=1, hoist=1, share=1, unroll=5,
             lean=1, block=4, probe=1),
    'L7': _a('reassociation', 'L6 + nodot + folded decode: the full port', fuse=1, hoist=1,
             share=1, unroll=5, lean=1, block=4, probe=1, nodot=1, decode='lean'),
}

LABELLED = {
    'f32out': _a('labelled', 'L7 with float32 output fields; NOT a parity arm', fuse=1,
                 hoist=1, share=1, unroll=5, lean=1, block=4, probe=1, nodot=1,
                 decode='lean_f32'),
}

ALL = {**ISOLATED, **CUMULATIVE, **LABELLED}
ARMS = {k: v['opts'] for k, v in ALL.items()}

# The arm whose optimisations the throughput and fine-mesh jobs inherit, chosen before
# any run: the full port. If it fails a parity gate the report says so and falls back to
# the deepest cumulative arm that passes, which is recorded, not chosen silently.
THROUGHPUT_PREFERENCE = ['L7', 'L6', 'L5', 'L4', 'L3', 'L2', 'L1']
