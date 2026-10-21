"""imp-fronts §B5 adaptive rule / step policy as a PURE state machine (DESIGN B5, Amendments 1-3; Amendment 3 governs).

No JAX, no NumPy: the in-job driver (fpolicy.py) and the unit tests (tests/test_policy.py) share this code. The driver
supplies run records; this module decides which runs a state needs and what follows from their evidence.

State (rule, f): `rule` on the escalation chain, factor f in {1, 0.5, 0.25} with r2acc's dt = DT0 * f.
Round R(rule, f) needs (Amendment 5 B9/B5 Q1 scope)
    prod  = (rule, f)                         recorded run with the in-job per-step fresh-Jacobian certificate
    tight = (rule, f) at tight tolerance      the Q1 tight-solve distance of prod (run when the state is visited, so a
                                              promoted half gets its tight solve then)
    succ  = (q2_successor[rule], f)           replayed from prod's own initial state at the same factor; per-step
                                              certificate, no tight
    half  = (rule, f/2)          if f > 0.25  per-step certificate, no tight; reused as prod of (rule, f/2)
Tighter (one per anchor case, at its terminal state) is run by the driver after the loop, not by this module.
Evidence a record carries (None / missing = unavailable):
    prod: valid (bool)
    succ: valid (bool), dist_succ (prod vs succ rollout distance), rho_succ_max (rho at prod's own reached states)
    half: valid (bool), dist_half (prod vs half rollout distance)
Check precedence at each state (Amendment 3 B5):
    (1) prod missing / invalid / non-finite                    -> INVALID (e = inf)
    (2) Q2 evidence missing or non-finite, or Q2 fails         -> next_candidate[rule] at f, or Q2-FAIL if none
    (3) f = 0.25                                                -> STEP-UNCONVERGED
    (4) half missing / invalid / non-finite distance            -> STEP-UNAVAILABLE
    (5) step distance <= 2.5e-4                                 -> CONVERGED
    (6) else                                                    -> (rule, f/2)
Rounds are capped at 10; round 10 is evaluated fully before the cap applies (the cap only labels states left
non-terminal: ROUND-CAP). The in-job GPU-time ceiling labels unfinished cases BUDGET-CAP. Both are scored at the
latest valid prod. No certified start rule -> start at the chain's first rule, flagged 'uncertified_start'.
A case is numerically qualified iff terminal CONVERGED and its terminal prod passes Q1.
"""
from __future__ import annotations

import math

BAR_Q2, BAR_RHO, BAR_STEP = 2.5e-4, .116, 2.5e-4
FACTORS = (1., .5, .25)
F_MIN = .25
MAX_ROUNDS = 10

# Amendment 3 B5, verbatim
NEXT_CANDIDATE = {'fib4181': 'fib6765', 'fib6765': 'gauss96', 'gauss96': 'fib10946', 'fib10946': 'gauss128',
                  'gauss128': 'gauss160', 'gauss160': 'gauss192', 'gauss192': 'gauss256'}
CHAIN = ('fib4181', 'fib6765', 'gauss96', 'fib10946', 'gauss128', 'gauss160', 'gauss192', 'gauss256')
Q2_SUCCESSOR = {'fib4181': 'fib6765', 'fib6765': 'fib10946', 'gauss96': 'gauss128', 'fib10946': 'fib17711',
                'gauss128': 'gauss160', 'gauss160': 'gauss192', 'gauss192': 'gauss256', 'gauss256': 'gauss384'}

TERMINALS = ('INVALID', 'Q2-FAIL', 'STEP-UNCONVERGED', 'STEP-UNAVAILABLE', 'CONVERGED', 'ROUND-CAP', 'BUDGET-CAP')


def finite(x):
    return x is not None and isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def start_state(selected):
    """(rule, f, uncertified_start). selected = the bank's certified selected R384 rule or None."""
    if selected is None:
        return CHAIN[0], 1., True
    if selected not in Q2_SUCCESSOR:
        raise ValueError(f'selected rule {selected} is not on the escalation chain')
    return selected, 1., False


def half_factor(f):
    return None if f <= F_MIN + 1e-15 else f / 2.


def needed(rule, f):
    """Runs a round at (rule, f) needs: [(kind, rule, factor)]; kind in prod / tight / succ / half (in this order)."""
    out = [('prod', rule, f), ('tight', rule, f), ('succ', Q2_SUCCESSOR[rule], f)]
    h = half_factor(f)
    if h is not None:
        out.append(('half', rule, h))
    return out


def run_ok(rec):
    return bool(rec) and rec.get('valid') is True


def q2_pass(succ):
    """Q2 at the state: succ valid, rollout distance and rho at prod's own reached states both finite and within bar."""
    if not run_ok(succ):
        return False
    d, r = succ.get('dist_succ'), succ.get('rho_succ_max')
    return finite(d) and finite(r) and d <= BAR_Q2 and r <= BAR_RHO


def decide(rule, f, prod, succ, half, next_candidate=None):
    """Transition at (rule, f) from the round's three records. Returns ('terminal', label) or ('next', (rule', f'))."""
    nc = NEXT_CANDIDATE if next_candidate is None else next_candidate
    if not run_ok(prod):                                                # (1)
        return 'terminal', 'INVALID'
    if not q2_pass(succ):                                               # (2)
        nxt = nc.get(rule)
        return ('next', (nxt, f)) if nxt is not None else ('terminal', 'Q2-FAIL')
    if half_factor(f) is None:                                          # (3)
        return 'terminal', 'STEP-UNCONVERGED'
    if not run_ok(half) or not finite(half.get('dist_half')):           # (4)
        return 'terminal', 'STEP-UNAVAILABLE'
    if half['dist_half'] <= BAR_STEP:                                   # (5)
        return 'terminal', 'CONVERGED'
    return 'next', (rule, half_factor(f))                               # (6)


class Case:
    """Per-case policy bookkeeping. history = [(round, rule, f, decision)], scored = run key of the scored prod."""

    def __init__(self, cid, selected):
        self.cid = cid
        self.rule, self.f, self.uncertified_start = start_state(selected)
        self.history, self.terminal, self.scored, self.latest_valid_prod = [], None, None, None
        self.rounds = 0

    def state(self):
        return (self.rule, self.f)

    def as_dict(self):
        return dict(case=self.cid, terminal=self.terminal, scored=self.scored, rounds=self.rounds,
                    uncertified_start=self.uncertified_start, history=self.history,
                    final_state=[self.rule, self.f], latest_valid_prod=self.latest_valid_prod)


def run_key(cid, kind, rule, f):
    """Runs are keyed by their identity; a 'half' run IS the later prod of (rule, f/2) (same certified run)."""
    cls = {'prod': 'cert', 'half': 'cert', 'tight': 'tight', 'succ': 'succ'}[kind]
    return f'{cid}|{cls}|{rule}|{f:g}'


def advance(case, records, max_rounds=MAX_ROUNDS, next_candidate=None):
    """Evaluate one round of `case` from `records` (dict run_key -> record or missing). Mutates and returns the case."""
    assert case.terminal is None
    case.rounds += 1
    rule, f = case.state()
    rk = {k: run_key(case.cid, k, r_, f_) for k, r_, f_ in needed(rule, f)}
    prod, succ = records.get(rk['prod']), records.get(rk['succ'])
    half = records.get(rk['half']) if 'half' in rk else None
    if run_ok(prod):
        case.latest_valid_prod = rk['prod']
    kind, val = decide(rule, f, prod, succ, half, next_candidate)
    case.history.append(dict(round=case.rounds, rule=rule, f=f, decision=kind, value=val,
                             prod=rk['prod'], succ=rk['succ'], half=rk.get('half')))
    if kind == 'terminal':
        case.terminal = val
        case.scored = None if val == 'INVALID' else rk['prod']
    else:
        case.rule, case.f = val
        if case.rounds >= max_rounds:          # round max_rounds evaluated fully; the cap labels what is left open
            case.terminal, case.scored = 'ROUND-CAP', case.latest_valid_prod
    return case


def budget_cap(case, store=None):
    """Ceiling reached with the case non-terminal: BUDGET-CAP (unqualified), scored at its latest valid prod (the current
    state's prod counts when it already exists, e.g. a promoted half)."""
    assert case.terminal is None
    if store is not None:
        k = run_key(case.cid, 'prod', case.rule, case.f)
        if run_ok(store.get(k)):
            case.latest_valid_prod = k
    case.terminal, case.scored = 'BUDGET-CAP', case.latest_valid_prod
    case.history.append(dict(round=case.rounds + 1, rule=case.rule, f=case.f, decision='terminal', value='BUDGET-CAP'))
    return case


def qualified(case, q1_of):
    """Numerically qualified iff terminal CONVERGED and the terminal prod passes Q1 (q1_of: run_key -> bool | None)."""
    return bool(case.terminal == 'CONVERGED' and q1_of(case.scored) is True)


def drive(cases, execute, max_rounds=MAX_ROUNDS, over_budget=lambda: False, next_candidate=None, on_round=None):
    """Round loop shared by the driver and the tests. execute(requests) runs the requests not yet in the record store,
    each a dict(key, cid, kind, rule, f, state_rule, state_f, prod_key) in state order (prod, tight, succ, half of the same
    state), and returns {key: record}; succ / half records carry their distance to the state's prod (prod_key). It may
    stop early (budget) by returning fewer records and making over_budget() true. Records persist across rounds (a half
    run is the later prod of (rule, f/2), keyed identically; a succ rule is the successor of exactly one rule, so a succ
    key fixes its prod). on_round(cases, store) is called after every round (persistence for the collector)."""
    store = {}
    while True:
        open_ = [c for c in cases if c.terminal is None]
        if not open_:
            break
        if over_budget():
            for c in open_:
                budget_cap(c, store)
            break
        req, seen = [], set()
        for c in open_:
            rule, f = c.state()
            for kind, r_, f_ in needed(rule, f):
                k = run_key(c.cid, kind, r_, f_)
                if k not in store and k not in seen:
                    seen.add(k)
                    req.append(dict(key=k, cid=c.cid, kind=kind, rule=r_, f=f_, state_rule=rule, state_f=f,
                                    prod_key=run_key(c.cid, 'prod', rule, f)))
        store.update(execute(req))
        if over_budget() and any(q['key'] not in store for q in req):
            for c in open_:            # the round could not be completed: unfinished cases stop here
                ks = [run_key(c.cid, k_, r_, f_) for k_, r_, f_ in needed(*c.state())]
                if any(k not in store for k in ks):
                    budget_cap(c, store)
        for c in open_:
            if c.terminal is None:
                advance(c, store, max_rounds, next_candidate)
        if on_round is not None:
            on_round(cases, store)
    return store


def paths(selected):
    """Every reachable state sequence from the start (each state either escalates, halves, or ends): used by the cost
    certificate (resources.py) to bound per-case solve inventories."""
    r0, f0, _ = start_state(selected)
    out, stack = [], [[(r0, f0)]]
    while stack:
        p = stack.pop()
        out.append(p)
        rule, f = p[-1]
        if NEXT_CANDIDATE.get(rule):
            stack.append(p + [(NEXT_CANDIDATE[rule], f)])
        if half_factor(f) is not None:
            stack.append(p + [(rule, half_factor(f))])
    return out


def path_solves(path):
    """Solve inventory of a state path (Amendment 5): per visited state prod + tight + succ (+ half if f > 0.25), with a
    half run promoted to the next state's prod not re-solved. Returns [(kind, rule, f)]."""
    done, out = set(), []
    for rule, f in path:
        for kind, r_, f_ in needed(rule, f):
            k = run_key('x', kind, r_, f_)
            if k not in done:
                done.add(k)
                out.append((kind, r_, f_))
    return out
