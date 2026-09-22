# Codex design audit — could not run (2026-09-22)

`codex exec -m gpt-6-astra -s read-only -o reports/codex-design-audit.md -` was invoked
headless with the brief in `design-audit-prompt.txt`, before the first job. Codex started,
but its bubblewrap sandbox failed on every file read:

```
bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted
```

It retried without shell initialisation, failed identically, read no file, and returned a
note saying so — explicitly "not a passing review". This is the same failure
`experiments/no-second/reports/codex-design-audit.md` records for 2026-09-17, so it is an
environment problem on this machine, not a one-off.

Per that lane's precedent the independent audit was performed instead by a Claude subagent
with the same adversarial brief: `design-audit-2026-09-22.md`. Its findings and their
disposition — every one accepted, rejected or deferred, with the action taken — are in
`DESIGN.md` §A2. Codex should be retried for the report audit if its sandbox is repaired.
