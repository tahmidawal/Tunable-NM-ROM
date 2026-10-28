"""Static retained-source invariants. Imports no numerical package."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def fun(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    for p in HERE.rglob('*.py'):
        ast.parse(p.read_text(), filename=str(p))
    original = ast.parse((ROOT/'experiments/jcp-heads/hcore.py').read_text())
    modified = ast.parse((HERE/'stepcore.py').read_text())
    for name in ('head_fn', 'initialize', 'decode', 'make_head_certifier', 'make_head_jac', 'trust_of', 'zsub_of'):
        assert dump(fun(original, name)) == dump(fun(modified, name)), name
    for name in ('inner', 'dscale', 'res', 'evalJ'):
        assert dump(fun(original, name)) == dump(fun(modified, name)), name
    for tree in (original, modified):
        body = fun(tree, 'body')
        lm_calls = [n for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'lm']
        assert len(lm_calls) == 1
        assert dump(lm_calls[0]) == dump(ast.parse("lm(zi, args, tol, sch['gtol'], lam0)", mode='eval').body)
    assert dump(fun(original, 'body').body[-1]) == dump(fun(modified, 'body').body[-1]), 'fine history/damping carry'
    print('PASS: syntax; retained residual/Jacobian, initial fit, decoding, fine LM call and history/damping carry')


if __name__ == '__main__': main()
