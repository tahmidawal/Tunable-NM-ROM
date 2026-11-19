"""Pure orchestration checks: failed gates and full reservations stop advancement.

This tests the submission flow through a source-extracted function with external
effects replaced. It never reads solutions, submits jobs or changes real files.
"""
from pathlib import Path
import ast,copy,json,shlex

path=Path(__file__).parent/'cluster/finish_b5.py'
tree=ast.parse(path.read_text());original=next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='main')
# Exercise the registered dependency flow starting at the first retained gate.
# Startup ownership writes have separate filesystem guards and are not executed.
node=copy.deepcopy(original)
start=next(i for i,v in enumerate(node.body) if isinstance(v,ast.If) and isinstance(v.test,ast.UnaryOp)
    and isinstance(v.test.operand,ast.Call) and isinstance(v.test.operand.func,ast.Name)
    and v.test.operand.func.id=='collect_and_audit')
node.body=node.body[start:]
module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]))
checks=[]
for failed in ['b5cache','b5normal','b5compact','reservation','b5block1',None]:
    submitted=[];audited=[];closed=[]
    def collect(names):
        audited.append(names)
        return failed not in names
    def submit(name,args,minutes):submitted.append((name,minutes,args))
    def close(reason):closed.append(reason);return reason
    def replace(tokens,flag,value):
        tokens=list(tokens)
        if flag in tokens:tokens[tokens.index(flag)+1]=value
        else:tokens.extend([flag,value])
        return tokens
    env=dict(collect_and_audit=collect,submit=submit,close=close,replace=replace,
        reconcile=lambda:[dict(name='fixture',state='COMPLETED',elapsed_seconds=40000 if failed=='reservation' else 36000)],
        TERMINAL={'COMPLETED'},
        reg=dict(cumulative_ceiling_gpu_seconds=42296),ns='/owned',base=['--compare-previous','/owned/baseline'])
    exec(compile(module,str(path),'exec'),env);env['main']()
    names=[v[0] for v in submitted]
    expected={'b5cache':[],'b5normal':['b5normal'],'b5compact':['b5normal','b5compact'],
        'reservation':['b5normal','b5compact']}.get(failed,['b5normal','b5compact','b5block0','b5block1'])
    assert names==expected,(failed,names,expected)
    assert len(closed)==1
    assert all(v[1]<=45 for v in submitted)
    if 'b5block0' in names:
        assert [v[1] for v in submitted[-2:]]==[45,45]
        assert submitted[-2][2][-2:]==['--block-index','0'] and submitted[-1][2][-2:]==['--block-index','1']
    checks.append(dict(failed_condition=failed,submitted=names,closed=True))
print(json.dumps(dict(status='passed',checks=checks,scope='No external effects; dependency and reservation gates only'),indent=2))
