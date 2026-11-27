#!/usr/bin/env python3
"""Small nodes in a deep tree, and Hybrid moving them to the CPU.

(a) HIGGS, CART, depth <= 30: each level's samples by the size of their node
    (measured, data/higgs-cart-d30-levels.json, made by tree_levels.py);
(b) the share of each level's nodes and samples that Hybrid grows on the CPU
    (nodes below m = 512, the swept --handoff-rows for HIGGS at depth 30 on hopper; measured, same file);
(c) a schematic Hybrid timeline: GPU levels, hand-offs, CPU subtree tasks.

  python3 paper/tools/diagrams/small_nodes.py   # writes paper/figs/diagram-small-nodes.pdf
Needs pdflatex with TikZ, pgfplots and lmodern.
"""
import json, subprocess, os, heapq, shutil, tempfile
HERE=os.path.dirname(os.path.abspath(__file__))
L=json.load(open(os.path.join(HERE,'data','higgs-cart-d30-levels.json')))
def co(f): return ' '.join(f'({r["d"]},{f(r):.3f})' for r in L)
share=[co(lambda r:100*r['rows_ge32k']/r['rows']),
       co(lambda r:100*(r['rows_ge4k']-r['rows_ge32k'])/r['rows']),
       co(lambda r:100*(r['rows_ge512']-r['rows_ge4k'])/r['rows']),
       co(lambda r:100*(r['rows']-r['rows_ge512'])/r['rows'])]
cpu_rows=co(lambda r:100*(r['rows']-r['rows_ge512'])/r['rows'])
cpu_nodes=co(lambda r:100*(r['nodes']-r['ge512'])/r['nodes'])
AW=30.0; AH=18.9; TOP=-2.5; LEGY=-29.0
AX=rf'''width={AW}mm,height={AH}mm,scale only axis,xmin=0,xmax=30,xtick={{0,10,20,30}},axis line style={{muted}},tick style={{muted}},
 tick label style={{font=\sixpt}},label style={{font=\scriptsize}},title style={{font=\scriptsize,align=center,yshift=-1mm}},
 ylabel style={{yshift=-0.7mm}},xlabel style={{yshift=1.5mm}},ymin=0,ymax=100,ytick={{0,50,100}}'''
o=[]
# (a) neutral ramp (node size is not a device)
fills=['gA','gB','gC','gD']
plots=''.join(rf'\addplot[fill={f},draw=white,line width=0.3pt] coordinates {{{s}}} \closedcycle;'+'\n' for f,s in zip(fills,share))
o.append(rf'''\begin{{axis}}[{AX},at={{(0,{TOP}mm)}},anchor=north west,stack plots=y,area style,ylabel={{samples in level (\%)}},xlabel={{tree level}}]
{plots}\end{{axis}}''')
leg=[('gA',r'${\ge}\,32{,}768$'),('gB',r'$4{,}096$--$32{,}767$'),('gC',r'$512$--$4{,}095$'),('gD',r'${<}\,512$')]
for i,(c,l) in enumerate(leg):
    x=-6.5+(i%2)*17; y=LEGY-(i//2)*2.6
    o.append(rf'\fill[{c}] ({x}mm,{y}mm) rectangle ++(1.8mm,1.8mm); \node[font=\legendfont,anchor=west,inner sep=0] at ({x+2.2}mm,{y+0.9}mm) {{{l}}};')
# (b) CPU share (CPU = blue)
BX=AW+10.3
o.append(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at (-6.5mm,1.0mm) {{\textbf{{(a)}} samples by node size}};')
o.append(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at ({BX-6.5}mm,1.0mm) {{\textbf{{(b)}} Hybrid: CPU share}};')
o.append(rf'''\begin{{axis}}[{AX},at={{({BX}mm,{TOP}mm)}},anchor=north west,ylabel={{on the CPU (\%)}},xlabel={{tree level}}]
\addplot[cpu,thick] coordinates {{{cpu_nodes}}};
\addplot[cpu,thick,dashed] coordinates {{{cpu_rows}}};
\end{{axis}}''')
for i,(st,l) in enumerate([('','of the nodes'),('dashed','of the samples')]):
    y=LEGY-i*2.6
    o.append(rf'\draw[cpu,thick,{st}] ({BX-6.5}mm,{y+0.9}mm) -- ++(4mm,0); \node[font=\legendfont,anchor=west,inner sep=0] at ({BX-1.9}mm,{y+0.9}mm) {{{l}}};')
# (c) timeline: GPU green, CPU threads blue
CX=BX+AW+4.0; LW=111.5-CX-6.5; X=CX+6.5
o.append(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at ({CX}mm,1.0mm) {{\textbf{{(c)}} Hybrid over time}};')
lanes=['GPU','CPU 1','CPU 2','CPU 3','CPU 4']; LH=2.5; P=(AH-LH)/4
for i,l in enumerate(lanes):
    y=-2.5-i*P
    o.append(rf'\fill[lane] ({X}mm,{y}mm) rectangle ++({LW}mm,{-LH}mm);')
    o.append(rf'\node[font=\sixpt,anchor=east,inner sep=0] at ({X-0.8}mm,{y-LH/2}mm) {{{l}}};')
K=LW/34.0
t=0; levels=[6,5.6,5.2,4.6,3.6,2.6,1.8,1.2]; ho=[]
for d,w in enumerate(levels):
    o.append(rf'\fill[gpu] ({X+t*K:.2f}mm,-2.5mm) rectangle ++({(w-0.4)*K:.2f}mm,{-LH}mm);'); t+=w
    if d>=3: ho.append(t)
free=[(0,i) for i in range(4)]
cols=['cpu']*4; tasks=[[3,2.5,2],[4,3,2.5,2],[3.5,2.5,2,1.5],[2.5,2,1.5],[1.5,1.2]]
for h,ts in zip(ho,tasks):
    for w in ts:
        f,i=heapq.heappop(free); s=max(f,h+0.4)
        o.append(rf'\fill[{cols[i]}] ({X+s*K:.2f}mm,{-2.5-(i+1)*P:.2f}mm) rectangle ++({(w-0.3)*K:.2f}mm,{-LH}mm);'); heapq.heappush(free,(s+w,i))
    o.append(rf'\draw[xfer,line width=1.1pt] ({X+(h-0.2)*K:.2f}mm,{-2.5-LH-0.1:.2f}mm) -- ++(0,-0.9mm);')
o.append(rf'\draw[-{{Latex[length=1mm]}},muted] ({X}mm,{TOP-AH:.2f}mm) -- ({X+LW}mm,{TOP-AH:.2f}mm) node[pos=1,anchor=north east,font=\scriptsize,inner sep=0.6mm] {{time}};')
o.append(rf'\fill[gpu] ({CX}mm,{LEGY}mm) rectangle ++(1.8mm,1.8mm); \node[font=\legendfont,anchor=west,inner sep=0] at ({CX+2.2}mm,{LEGY+0.9}mm) {{GPU levels}};')
o.append(rf'\fill[cpu] ({CX+17}mm,{LEGY}mm) rectangle ++(1.8mm,1.8mm); \node[font=\legendfont,anchor=west,inner sep=0] at ({CX+19.2}mm,{LEGY+0.9}mm) {{CPU subtrees}};')
o.append(rf'\draw[xfer,line width=1.1pt] ({CX+0.9}mm,{LEGY-2.6+1.5}mm) -- ++(0,-1.2mm); \node[font=\legendfont,anchor=west,inner sep=0] at ({CX+2.2}mm,{LEGY-1.7}mm) {{hand-off of nodes ${{<}}\,m$}};')
tex=r'''\documentclass[border=1pt]{standalone}
\input{common}
\usepackage{pgfplots}\pgfplotsset{compat=1.17}
\begin{document}\footnotesize
\begin{tikzpicture}
'''+'\n'.join(o)+r'''
\end{tikzpicture}
\end{document}'''
with tempfile.TemporaryDirectory() as tmp:
    open(os.path.join(tmp,'d.tex'),'w').write(tex)
    shutil.copy(os.path.join(HERE,'common.tex'),tmp)
    r=subprocess.run(['pdflatex','-interaction=nonstopmode','d.tex'],cwd=tmp,capture_output=True,text=True)
    if r.returncode: raise SystemExit(r.stdout[-2000:])
    out=os.path.normpath(os.path.join(HERE,'..','..','figs','diagram-small-nodes.pdf'))
    shutil.copy(os.path.join(tmp,'d.pdf'),out)
print('wrote',out)
