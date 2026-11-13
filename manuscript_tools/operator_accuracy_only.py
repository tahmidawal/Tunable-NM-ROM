"""Apply the author's accuracy-only neural-operator presentation to the saved draft.

All error cells are retained from the recorded manuscript; no measurements are created.
"""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT/'paper_revision'
s = subprocess.check_output(['git','-c','gc.auto=0','show',
    'a489a8c13:paper_revision/main.tex'],cwd=ROOT,text=True)

def replace(old,new):
    global s
    assert s.count(old)==1, old[:100]
    s=s.replace(old,new)

def table(label,fn):
    global s
    at=s.index('\\label{'+label+'}')
    start=s.rfind('\\begin{table}',0,at)
    end=s.index('\\end{table}',at)+len('\\end{table}')
    s=s[:start]+fn(s[start:end])+s[end:]

def transform_main(block):
    lines=[]
    for line in block.splitlines():
        if re.match(r'^(FNO|U-Net)',line) and '&' in line:
            cells=line.removesuffix(r' \\').split('&')
            assert len(cells)==5
            cells[-1]=' --- '
            line='&'.join(cells).rstrip()+r' \\'
        lines.append(line)
    block='\n'.join(lines)
    old="performance, not convergence. The 128-trajectory rows rerun the submission's data budget."
    assert old in block
    return block.replace(old,old+' Neural operators are compared by prediction error only; inference times are not reported for those rows.')

table('tab:baselines',transform_main)
replace('neural operators), each against accuracy-matched comparators.',
        'neural operators). Classical-solver costs are accuracy-matched; neural-operator comparisons report prediction errors under the stated training budgets.')
replace('''(0.43--0.51\\,\\% against 0.875\\,\\%) and are 6--7 times faster (11 times for
the smaller FNO).''','''(0.43--0.51\\,\\% against 0.875\\,\\%). We compare these direct predictors
by error, separately from the timing comparison with numerical solvers and ROMs.''')
replace('''gives 2.41--2.52\\,\\% in 18--23\\,ms against 0.90--1.39\\,\\% in 13--29\\,ms''',
        '''gives 2.41--2.52\\,\\% worst-case error against 0.90--1.39\\,\\%''')
replace('''$9.9{\\times}10^{-3}$ at Reynolds 105 in 0.45\\,s on a GPU, where our width''',
        '''$9.9{\\times}10^{-3}$ at Reynolds 105, where our width''')
replace('''six at $5.5{\\times}10^{-3}$ in 0.83\\,s on a CPU node; over the long window''',
        '''six at $5.5{\\times}10^{-3}$; over the long window''')

def transform_old(block):
    cap=block.index('\\caption{');label=block.index('\\label{')
    block=block[:cap]+r'''\caption{\rev{Historical accuracy-only operator comparison. Burgers operators
used 128 trajectories and are superseded by the matched-data
Table~\ref{tab:baselines}; heat runs completed only 6--7 epochs and support
no claim.} Worst-case relative errors (\%) at each measured cell; both NM-ROM
settings come from one trained model. Operator sizes were selected on validation
data under the reported training budget. No operator timing comparison is made.}
'''+block[label:]
    block=block.replace(r'\begin{tabular}{@{}llrrrrrrrrrrrr@{}}',r'\begin{tabular}{@{}llrrrrrr@{}}')
    lines=[]
    for line in block.splitlines():
        if '\\multicolumn' in line:
            lines.append(r'PDE & $N$ & NM-ROM accurate & NM-ROM second setting & FNO & U-Net & Transolver & DeepONet \\')
        elif line.startswith('\\cmidrule') or line.startswith('PDE & $N$ & Err.'):
            continue
        elif '&' in line and line.rstrip().endswith(r'\\'):
            cells=line.removesuffix(r' \\').split('&')
            assert len(cells)==14,len(cells)
            cells=[cells[i] for i in [0,1,2,4,6,8,10,12]]
            line='&'.join(cells).rstrip()+r' \\'
            line=re.sub(r'\\textbf\{([^{}]+)\}',r'\1',line)
            lines.append(line)
        else:lines.append(line)
    return '\n'.join(lines)

table('tab:operators',transform_old)
start=s.index('In these jobs the\nBurgers accurate setting')
end=s.index('\n\n\\begin{table}',start)
s=s[:start]+r'''The accurate NM-ROM setting is the span with $R'=384$; the second
setting uses the head with $k=16$ on Burgers and the span with $R'=16$ on
Navier--Stokes. Blank cells indicate unavailable measurements.'''+s[end:]
replace('Speedup as in Table~\\ref{tab:operators}: one full-order setting',
        'Speedup uses one full-order setting')
replace('''This table was timed in a separate run from
Table~\\ref{tab:operators} (H200 at $2048^2$, H100 there), so the same setting can show a different speedup.''',
        '''The reduced-model timing comparison uses a separate allocation.''')

def transform_matched(block):
    block=block.replace(r'\begin{tabular}{@{}llrrr@{\hspace{10pt}}rr@{}}',r'\begin{tabular}{@{}llrr@{\hspace{10pt}}r@{}}')
    block=block.replace(r'\multicolumn{3}{c}{$256^2$} & \multicolumn{2}{c}{$1024^2$}',r'\multicolumn{2}{c}{$256^2$} & $1024^2$')
    block=block.replace(r'\cmidrule(lr){3-5}\cmidrule(l){6-7}',r'\cmidrule(lr){3-4}\cmidrule(l){5-5}')
    lines=[]
    for line in block.splitlines():
        if '&' in line and '\\multicolumn' not in line and line.rstrip().endswith(r'\\'):
            cells=line.removesuffix(r' \\').split('&')
            assert len(cells)==7,len(cells)
            line='&'.join(cells[i] for i in [0,1,2,3,5]).rstrip()+r' \\'
        lines.append(line)
    return '\n'.join(lines)

table('tab:ops-matched',transform_matched)
replace('''Table~\\ref{tab:coordnet} in 18.4/21.0/23.5\\,ms; it is faster than our
coordinate bank with the head only at $128^3$.''',
        '''Table~\\ref{tab:coordnet}; the coordinate-bank head gives
0.90/1.23/1.39\\,\\% on the same cohort. This is an accuracy-only comparison.''')
replace('''training each), timed on one GPU.''','''training each). We report prediction errors and acceptance targets only.''')
replace('''band. Inference per window on an L40S GPU: Transolver-style 2.5\\,s, FNO
0.45\\,s, DeepONet 2.8\\,s.''','''band. The operator comparison is accuracy-only.''')
replace('''pseudo-spectral solvers; green: neural operators (one point per trained
model).''','''pseudo-spectral solvers. Neural operators are compared by error in
Tables~\\ref{tab:baselines} and~\\ref{tab:ops-matched} and are omitted from this cost plot.''')
replace('All online coefficients come\nfrom the PDE residual, without access to the reference solution.',
        'Our intended use is repeated queries within a parameterised family after\n'
        'offline learning and assembly. The learned network provides a compact\n'
        'representation; each query still determines its coefficients from the PDE\n'
        'residual, without access to the reference solution.')
(PAPER/'main.tex').write_text(s)
print('Converted neural-operator presentation to accuracy only; retained error cells.')
