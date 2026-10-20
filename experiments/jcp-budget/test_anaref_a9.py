"""Regressions for the terminal-case and T = 1.0 eligibility rules (A9.1: audit codex-job-a9-1; A11.1: audit
codex-job-t1-1).
- an A9 case (tight_after_refine = false, refined, whole-case 'unresolved-floor') is never eligible, even with an
  accepted window and every other gate passing; the same window without the A9 flag keeps the per-window rule;
- at T = 1.0 any case whose whole-run judge did not accept is terminal (no refinement at T = 1.0);
- ana7.elig_for at T = 1.0 takes W1/W2 from the T = 0.5 references AND the T = 1.0 prefix / non-terminal status, and W3
  from the T = 1.0 references."""
import json
from pathlib import Path

from anaref import a9_terminal, terminal, window_eligible
from ana7 import elig_for

base = dict(outcome='unresolved-floor', refined=True, tight_after_refine=False, G_R2=dict(passed=True), horizon=.5)
args = ('accepted', True, True, 'validated', False, None)
t1 = dict(outcome='unresolved-order', refined=False, tight_after_refine=False, G_R2=dict(passed=True), horizon=1.)


def case(T, w, elig, prefix=True, term=False):
    return dict(dim=2, cohort='val32', case=3, horizon=T, sha256={}, final=[],
                eligibility={x: dict(eligible=e, prefix=prefix, terminal=term, tier='validated') for x, e in zip(w, elig)})


AR = dict(cases=[case(.5, ('W1', 'W2'), (True, True)), case(1., ('W1', 'W2', 'W3'), (True, True, True))])
E = elig_for(AR, 1.)
AR_bad = dict(cases=[case(.5, ('W1', 'W2'), (True, False)), case(1., ('W1', 'W2', 'W3'), (True, True, True), prefix=False)])
Eb = elig_for(AR_bad, 1.)
r = dict(
    a9_case_ineligible=a9_terminal(base) and not window_eligible(base, *args),
    pre_a9_window_rule_kept=window_eligible(dict(base, tight_after_refine=True), *args),
    unrefined_not_terminal_at_T05=not a9_terminal(dict(base, refined=False)) and not terminal(dict(base, refined=False)),
    accepted_case_eligible=window_eligible(dict(base, outcome='accepted'), *args),
    legacy_record_default=not a9_terminal({k: v for k, v in base.items() if k != 'tight_after_refine'}),
    t1_failed_case_terminal=terminal(t1) and not window_eligible(t1, *args),
    t1_accepted_case_eligible=window_eligible(dict(t1, outcome='accepted'), *args),
    t1_all_eligible=all(E[('val32|3', w)]['eligible'] for w in ('W1', 'W2', 'W3')),
    t1_W2_from_T05=not Eb[('val32|3', 'W2')]['eligible'],
    t1_prefix_required=not Eb[('val32|3', 'W1')]['eligible'] and not Eb[('val32|3', 'W3')]['eligible'],
)
r['passed'] = all(r.values())
(Path(__file__).resolve().parent / 'checks' / 'test_anaref_a9.json').write_text(json.dumps(r, indent=1) + '\n')
print(r)
assert r['passed']
