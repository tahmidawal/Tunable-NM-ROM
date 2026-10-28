"""Generate complete AST spans for normal loads/bindings and metadata in G1."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def scan(path):
    source = path.read_text()
    tree = ast.parse(source)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def statement(node):
        while not isinstance(node, ast.stmt):
            node = parents[node]
        return node
    def scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return '<module>'
    def span(node):
        return {'start': node.lineno, 'end': node.end_lineno, 'scope': scope(node),
                'source': ast.get_source_segment(source, node)}
    loads = [node for node in ast.walk(tree) if isinstance(node, ast.Name)
             and isinstance(node.ctx, ast.Load) and node.id in ('normal', 'raw_normal')]
    grouped = {}
    for node in loads:
        stmt = statement(node)
        grouped.setdefault(stmt, []).append(node)
    records = []
    for stmt, nodes in sorted(grouped.items(), key=lambda item: (item[0].lineno,item[0].col_offset)):
        names = {node.id for node in nodes}
        category = ('guard' if isinstance(stmt, ast.If) else
                    'ordinary_BND_consumer_or_forwarding' if 'normal' in names else 'raw_normal_provenance')
        row = span(stmt)
        row.update(category=category, occurrences=[{'name': node.id, 'line': node.lineno,
                   'column': node.col_offset} for node in sorted(nodes, key=lambda n:(n.lineno,n.col_offset))])
        records.append(row)
    bindings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in ('normal','raw_normal'):
            bindings.append(dict(span(statement(node)), name=node.id, kind='assignment'))
        elif isinstance(node, ast.arg) and node.arg == 'normal':
            owner = parents[parents[node]]
            bindings.append({'start': node.lineno, 'end': node.end_lineno, 'scope': owner.name,
                             'name': node.arg, 'kind':'function_parameter'})
    metadata_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ('normal_certificate','orientation') and isinstance(node.ctx,ast.Load):
            stmt = statement(node)
            if stmt not in grouped:
                metadata_nodes.add(stmt)
    metadata = [dict(span(stmt),category='certificate_metadata_not_normal_consumer')
                for stmt in sorted(metadata_nodes,key=lambda n:(n.lineno,n.col_offset))]
    coverage = sum(len(row['occurrences']) for row in records)
    assert coverage == len(loads)
    return {'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'normal_name_load_count':len(loads), 'covered_load_count':coverage,
            'bindings':bindings, 'uses':records, 'metadata':metadata}


files = {name: scan(HERE/name) for name in ('fom.py','preflight.py')}
changed=[]
for name in files:
    old=subprocess.check_output(['git','show','9d431a0cc:experiments/ns-obstacle-2d/'+name],cwd=ROOT)
    changed.append({'path':name,'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':files[name]['sha256']})
record={'schema':'ns-obstacle-normal-callsite-ast-v1','scope':'isolated G1 candidate; G2 source excluded',
        'method':'Every AST Name load of normal/raw_normal is grouped by its nearest full statement span, including multiline dictionaries/comprehensions/lambdas. Bindings and certificate-only metadata are separate.',
        'changed_files':changed,'files':files,
        'helper_sha256':hashlib.sha256((ROOT/'oriented_normals.py').read_bytes()).hexdigest(),
        'volume_facet_scope':'helper uses terms(oriented) for ordinary integration and terms(raw) for volume-facet LinearForm; factor_cf/oriented never enter facet integrands',
        'evidence_only_delta':'4b8fe1905 sibling certificate path retained',
        'native_repair_execution':False}
(HERE/'CALLSITE-AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({name:{'normal_name_load_count':data['normal_name_load_count'],
                       'covered_load_count':data['covered_load_count'],'statement_spans':len(data['uses'])}
                  for name,data in files.items()},indent=2))
