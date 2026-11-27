#!/usr/bin/env python3
"""Schematic diagram: task parallelism in Multicore CPU over time.

(a) a small tree drawn node-link style, coloured by the thread that grows each node,
dark = split by attribute on all threads; (b) the same nodes on four threads over
time, from a small event simulation: after a split of a node above the task
threshold the thread passes the left child to the pool and continues with the right
one. Thresholds are scaled to the toy tree (32,768 and 4,096 in the code).
Node size in (a) grows with the number of samples; a grey arrow leads from (a) to (b).

  python3 paper/tools/diagrams/task_parallel.py   # writes paper/figs/diagram-task-parallel.pdf
Needs pdflatex with TikZ and lmodern.
"""
import os, sys, math, heapq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip import *; from mini import *
DATA=32; TASK=7    # schematic thresholds (32,768 and 4,096 in the code)
nl=list(nodes()); kids={p:[q for q,_,_,_ in nl if len(q)==len(p)+1 and q.startswith(p)] for p,_,_,_ in nl}
size={p:n for p,n,_,_ in nl}
letters={p:chr(97+i) for i,(p,_,_,_) in enumerate(nl)}
cols=['thA','thB','thC','thD']; light={'thC','thD'}   # light fills take dark text
own={}; blocks=[]; spawns=[]
# event simulation: free threads heap, task queue (FIFO by ready time)
free=[(0.0,i) for i in range(4)]; heapq.heapify(free)
q=[(0.0,0,'')]; cnt=1
while q:
    rt,_,task=heapq.heappop(q)
    t,th=heapq.heappop(free); t=max(t,rt)
    stack=[task]; cur=t
    while stack:
        p=stack.pop(); n=size[p]
        if n>=DATA:   # all threads together
            others=[heapq.heappop(free) for _ in range(len(free))]
            st=max([cur]+[f for f,_ in others]); d=n/4
            blocks.append((p,'all',st,st+d)); own[p]='all'
            for _,i in others: heapq.heappush(free,(st+d,i))
            cur=st+d
        else:
            blocks.append((p,th,cur,cur+n)); own[p]=th; cur+=n
        k=kids[p]
        if k:
            l,r=k
            if n>=TASK: heapq.heappush(q,(cur,cnt,l)); cnt+=1; spawns.append((p,l,cur)); stack.append(r)
            else: stack+=[r,l]
    heapq.heappush(free,(cur,th))
end=max(b[3] for b in blocks)
def col(p): return 'root' if own[p]=='all' else cols[own[p]]
def tcol(p): return 'ink' if col(p) in light else 'white'
spawned={(p,l) for p,l,_ in spawns}
# hand-offs drawn in (b): between two different single threads
moves=[(p,l,t) for p,l,t in spawns if own[p]!='all' and own[l]!='all' and own[p]!=own[l]]
DASH=r'ink,line width=0.6pt,dashed,dash pattern=on 0.8pt off 0.6pt'
RIGHT=119.9; LV=4.4; TOPY=-1.9   # right edge, tree level spacing, centre of the root (mm)

def figure():
    S=Strip()
    # ---------- (a) node-link tree ----------
    SL=5.2; xs={}; slot=[0]   # leaves on equal slots, parents centred over children
    def lay(p):
        k=kids[p]
        if not k: xs[p]=slot[0]*SL; slot[0]+=1
        else:
            for c in k: lay(c)
            xs[p]=(xs[k[0]]+xs[k[-1]])/2
    lay('')
    def dia(p): return 2.3+2.1*math.sqrt(size[p]/32)
    x0=max(dia(p)/2-xs[p] for p in xs)
    for p,n,d,_ in nl:
        S.add(rf'\node[circle,fill={col(p)},minimum size={dia(p):.2f}mm,inner sep=0,font=\sixpt,text={tcol(p)}] (n{p}) at ({x0+xs[p]:.2f}mm,{TOPY-d*LV:.2f}mm) {{{letters[p]}}};')
    AW=x0+max(xs[p]+dia(p)/2 for p in xs)
    title='the tree'; gap=9.5
    for p,n,d,_ in nl:
        for c in kids[p]:
            if (p,c) in spawned and own[p]!='all':   # child queued as a task
                S.add(rf'\draw[-{{Latex[length=1.1mm,width=0.9mm]}},{DASH},shorten >=0.3pt] (n{p}) -- (n{c});')
            else:
                S.add(rf'\draw[-{{Latex[length=1.1mm,width=0.9mm]}},muted,line width=0.45pt,shorten >=0.3pt] (n{p}) -- (n{c});')
    S.add(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at (0mm,1.2mm) {{\textbf{{(a)}} {title}}};')
    # ---------- (b) the same nodes on four thread lanes ----------
    LAB=12.6; X0=AW+gap+LAB; TW=RIGHT-X0; K=TW/end
    NL=4; LH=2.9; LG=(3*LV+3.3-NL*LH)/(NL-1)   # lanes span the height of the tree
    ytop=TOPY+1.65
    def ly(i): return ytop-i*(LH+LG)
    S.add(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at ({X0-LAB}mm,1.2mm) {{\textbf{{(b)}} the same nodes on four threads over time}};')
    for i in range(NL):
        y=ly(i)
        S.add(rf'\fill[lane,rounded corners=0.6mm] ({X0}mm,{y}mm) rectangle ++({TW}mm,{-LH}mm);')
        S.add(rf'\fill[{cols[i]}] ({X0-LAB+0.9}mm,{y-LH/2}mm) circle (0.9mm);')
        S.add(txt(X0-LAB+2.4,y-LH/2,f'thread {i+1}',anchor='west'))
    for p,th,a,b in blocks:
        for i in (range(NL) if th=='all' else [th]):
            y=ly(i)
            S.add(rf'\fill[{col(p)},rounded corners=0.8mm] ({X0+a*K+0.15:.2f}mm,{y:.2f}mm) rectangle ++({(b-a)*K-0.3:.2f}mm,{-LH}mm);')
            S.add(txt(X0+(a+b)/2*K,y-LH/2,letters[p],extra=f',text={tcol(p)}'))
    # hand-off arrows; two leaving at the same moment are pulled apart
    seen={}
    for p,l,t in moves:
        k=round(t,3); same=sum(round(tt,3)==k for _,_,tt in moves)
        off=0.7*(2*seen.get(k,0)-(same-1)); seen[k]=seen.get(k,0)+1
        st=[a for q,_,a,_ in blocks if q==l][0]
        y1=ly(own[p])-LH/2; y2=ly(own[l])-LH/2
        S.add(rf'\draw[-{{Latex[length=1.4mm,width=1.1mm]}},{DASH}] ({X0+t*K+off:.2f}mm,{y1:.2f}mm) to[bend left=10] ({X0+st*K+0.2+off:.2f}mm,{y2:.2f}mm);')
    # time axis and key on one row, a little below the last lane
    yb=ly(NL-1)-LH-1.5; KW=40.5
    S.add(rf'\draw[-{{Latex[length=1.1mm]}},muted] ({X0}mm,{yb}mm) -- ({X0+TW-KW}mm,{yb}mm) node[pos=0,anchor=east,font=\scriptsize,inner sep=0.8mm,text=ink] {{time}};')
    S.add(rf'\draw[-{{Latex[length=1.1mm,width=0.9mm]}},{DASH}] ({X0+TW-KW+3}mm,{yb}mm) -- ++(4mm,0) node[right,font=\legendfont,inner sep=0.6mm,text=ink] {{child passed to an idle thread}};')
    # grey "maps to" arrow between the panels
    mx=AW+4.8; my=TOPY-1.5*LV
    S.add(rf'\fill[black!12] ({mx-3.2}mm,{my+1.6}mm) -- ({mx+0.6}mm,{my+1.6}mm) -- ({mx+0.6}mm,{my+3.0}mm) -- ({mx+3.4}mm,{my}mm) -- ({mx+0.6}mm,{my-3.0}mm) -- ({mx+0.6}mm,{my-1.6}mm) -- ({mx-3.2}mm,{my-1.6}mm) -- cycle;')
    return S

figure().write('diagram-task-parallel')
