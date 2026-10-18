"""Regression (A9, audit codex-job-a9-1): a whole-case 'unresolved-floor' under tight_after_refine = false is never
eligible, even with an accepted window and every other gate passing; the same window without the A9 flag stays eligible."""
import json
from pathlib import Path

from anaref import a9_terminal, window_eligible

base = dict(outcome='unresolved-floor', refined=True, tight_after_refine=False, G_R2=dict(passed=True))
args = ('accepted', True, True, 'validated', False, None)
r = dict(
    a9_case_ineligible=a9_terminal(base) and not window_eligible(base, *args),
    pre_a9_window_rule_kept=window_eligible(dict(base, tight_after_refine=True), *args),
    unrefined_not_terminal=not a9_terminal(dict(base, refined=False)),
    accepted_case_eligible=window_eligible(dict(base, outcome='accepted'), *args),
    legacy_record_default=not a9_terminal({k: v for k, v in base.items() if k != 'tight_after_refine'}),
)
r['passed'] = all(r.values())
(Path(__file__).resolve().parent / 'checks' / 'test_anaref_a9.json').write_text(json.dumps(r, indent=1) + '\n')
print(r)
assert r['passed']
