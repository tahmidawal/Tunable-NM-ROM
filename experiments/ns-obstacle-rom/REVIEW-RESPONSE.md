# Independent infrastructure review response

Status: repair prepared for independent diff confirmation. The coordinator forwarded the independent Codex review of commit `b8e50edbf`; the principal algebra was consistent, with one blocking timing-evidence defect. This record does not certify FE integration, training or physical experiments.

## Blocking finding: mutable invocation evidence

The recorder hashed output arrays but retained references to the caller's arrays and accounting dictionaries. A solver reusing buffers could overwrite earlier scored fields and work counts while leaving their recorded hashes unchanged.

**Repair:** deep-copy the complete returned record immediately after each timed call, before hashing/storage and before any subsequent invocation. Convert the retained output values to arrays and mark them read-only. Work counts, nested accounting and component-time dictionaries come from the same independent snapshot. Record evidence-copy/hash time separately; caller-required output materialization stays inside the timed callable.

**Regression:** the synthetic smoke deliberately returns one shared NumPy array, one mutable work dictionary with a nested list, and one mutable component-time dictionary from both methods on every repetition. After collection it mutates every original object again. Each retained output must still match its original hash and expected call index, each accounting history must remain unchanged, and retained arrays must be read-only. The generated result lives in `CPU-SMOKE.json` under `timing_reused_buffer_regression`.

## Interpretation clarification: sampled rule versus online cost

The current compact rule changes convection tensor assembly, then uses the same online tensor dimensions as the full-rule control. Point-count reduction is not an online-solve speedup result. Assembly/setup must be charged, and a direct sampled evaluator would require separate implementation, review and measurement. This limitation is now stated in `API.md`.

No equations, thresholds, training data or physical admission status changed. Independent confirmation of this repair remains pending when this response is written.
