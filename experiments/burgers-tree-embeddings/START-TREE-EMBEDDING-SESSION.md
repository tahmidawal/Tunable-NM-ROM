# Start the Burgers tree embedding experiment

Paste the following into the new session, with working directory `/home/tahmid/Dev/QuadtreeNMROM`:

```text
Execute the new Burgers tree-embedding study in:
/home/tahmid/Dev/QuadtreeNMROM/TREE-EMBEDDING-PLAN.md

Also read:
/home/tahmid/Dev/QuadtreeNMROM/tree-embedding-protocol.json
/home/tahmid/Dev/QuadtreeNMROM/EXECUTION-OWNER.md
/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md
and all applicable AGENTS.md and CLAUDE.md instructions.

I am starting this NEW study in this session. Begin implementation, training and evaluation under the staged plan; do not stop at another proposal. The goal is lower total PDE-solve time than strong NM-ROM baselines at the same 1% physical error and actual 4096-by-4096 output.

Create a separate owned worktree, branch and cluster namespace, and claim this study as specified in the plan. The earlier burgers-quadtree campaign remains paused. Do not resume its worker, write into its implementation lane, overwrite its artifacts, or reuse its namespace. Reading and vendoring verified utilities with provenance is allowed.

Start with the bounded 8-GPU-hour batch for references, global models and shared direct/nonlinear child decoders. Expand in finite documented batches within this 2D study when the plan's evidence and gates justify it. Preserve unsuccessful recipes, distinguish reconstruction from actual PDE solving, and keep final tests unopened until method freeze.

Children are independent correction variables. Retain parents and fixed windows, so zero children preserve the existing field. The central test is improvement in error per unit of cost versus more global variables; nesting alone does not prove faster solving.

Use the common verified weak PDE operator for the first decoder comparison, account for adaptation/initialization/reconstruction costs, and include the planned NM-ROM competitors in confirmation. Do not switch to coarse physics, pointwise residual fitting, or truth-assisted online refinement without a clearly separated diagnostic.

Work autonomously within this scope. Keep concise progress updates and the canonical lab log current. No automatic merges or unrelated 3D campaign. Report the first representation plots and decision, then continue through the justified stages. If a gate cannot be passed after the bounded repairs, deliver the evidence and a scoped conclusion rather than hiding the failure or running an unbounded search.
```

The current planning session has not launched any training or submitted jobs for this new study. The launch instruction above is intended to be issued by the user in the new session.
