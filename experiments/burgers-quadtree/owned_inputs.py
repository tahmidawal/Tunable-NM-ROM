from campaign import *
NS=Path('/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117')
def verified(job,relative):
 root=NS/job
 manifest={line.split('  ',1)[1]:line.split('  ',1)[0] for line in (root/'OUTPUTS.sha256').read_text().splitlines()}
 p=root/'output'/relative;assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest['output/'+relative]
 return p
