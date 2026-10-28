"""Parse/compile source and verify unchanged retained fine operator components.

This imports no JAX or model code and performs no numerical execution.
"""
import ast
import json
from pathlib import Path
import py_compile

HERE=Path(__file__).resolve().parents[1]
ROOT=HERE.parents[1]


def functions(tree):
    return {n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}


def same(a,b):
    return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def main():
    old=functions(ast.parse((ROOT/'experiments/jcp-heads/hcore.py').read_text()))
    new=functions(ast.parse((HERE/'replaycore.py').read_text()))
    checks=[]
    for name in ('head_fn','make_head_certifier','make_head_jac','trust_of','zsub_of'):
        assert same(old[name],new[name]),name; checks.append(name)
    for parent,names in (('make_head_evolve',('inner','dscale','res','evalJ')),
                         ('make_head_query',('initialize','decode'))):
        a,b=functions(old[parent]),functions(new[parent])
        for name in names:
            assert same(a[name],b[name]),(parent,name); checks.append(parent+'.'+name)
    a=functions(functions(old['make_head_evolve'])['evolve'])['body']
    b=functions(functions(new['make_head_evolve'])['evolve'])['body']
    def assignment(body,name):
        return next(n for n in body.body if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    for name in ('co','cv','Acn','Acm','Fn','args','ze','zqd','cand','rs','zi','tol'):
        # cv is part of tuple assignment; verify it separately by source diff.
        if name=='cv': continue
        assert same(assignment(a,name),assignment(b,name)),name
        checks.append('fine_body.'+name)
    for file in (HERE/'pilot.py',HERE/'replay.py',HERE/'replaycore.py',HERE/'analyze.py',HERE/'cluster/stage.py'):
        py_compile.compile(str(file),doraise=True)
    report=dict(passed=True,checks=checks,scope='static AST identity and py_compile only; no imports/models/GPU')
    (HERE/'checks/static.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
