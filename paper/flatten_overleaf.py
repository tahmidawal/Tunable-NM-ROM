# Flatten \input{} recursively, reproducing TeX end-of-file spacing exactly.
# Usage (from paper/): python flatten_overleaf.py main.tex ../paper_latex/main.tex
import re,sys
pat=re.compile(r'\\input\{([^}]+)\}')
def expand(path,top=False):
    p=path if path.endswith('.tex') else path+'.tex'
    s=pat.sub(lambda m: expand(m.group(1)), open(p).read())
    if top: return s
    if s.endswith('\n'): s=s[:-1]
    lines=s.split('\n'); lines[-1]=lines[-1].rstrip(' '); s='\n'.join(lines); last=lines[-1]
    if re.search(r'(?<!\\)%',last) or last=='': tail='\n'   # comment/blank: endline eaten
    elif re.search(r'\\[A-Za-z]+$',last): tail='{}'        # endline after control word skipped
    else: tail=' {}'                                        # endline space; keep next endline a space
    return '%\n'+s+tail
out=expand(sys.argv[1],True)
open(sys.argv[2],"w").write(out)
