#!/usr/bin/env python3
"""Schematic diagram: task parallelism in Multicore CPU over time.

(a) a small tree (icicle: box width = samples), coloured by the thread that grows
each node, dark = split by attribute on all threads; (b) the four threads over time,
from a small event simulation: after a split of a node above the task threshold the
thread passes the left child to the pool and continues with the right one.
Thresholds are scaled to the toy tree (32,768 and 4,096 in the paper).

  python3 paper/tools/diagrams/task_parallel.py   # writes paper/figs/diagram-task-parallel.pdf
Needs pdflatex with TikZ and lmodern.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strip import *; from mini import *
import heapq
S=Strip()
DATA=32; TASK=7    # schematic thresholds (32,768 and 4,096 in the paper)
nl=list(nodes()); kids={p:[q for q,_,_,_ in nl if len(q)==len(p)+1 and q.startswith(p)] for p,_,_,_ in nl}
size={p:n for p,n,_,_ in nl}
letters={p:chr(97+i) for i,(p,_,_,_) in enumerate(nl)}
cols=['tA','tB','tC','tD']
own={}; blocks=[]; spawns=[]
# event simulation: free threads heap, task queue (FIFO by ready time)
free=[(0.0,i) for i in range(4)]; heapq.heapify(free)
q=[(0.0,0,'')]; cnt=1
while q:
    rt,_,task=heapq.heappop(q)
    t,th=heapq.heappop(free); t=max(t,rt)
    stack=[task]; cur=t
    while stack:
        p=stack.pop()
        n=size[p]
        if n>=DATA:   # all threads together
            others=[heapq.heappop(free) for _ in range(len(free))]
            st=max([cur]+[f for f,_ in others]); d=n/4
            blocks.append((p,'all',st,st+d)); own[p]='all'
            for _,i in others: heapq.heappush(free,(st+d,i))
            cur=st+d
        else:
            blocks.append((p,th,cur,cur+n)); own[p]=th; cur+=n*1.0
        k=kids[p]
        if k:
            l,r=k
            if n>=TASK:
                heapq.heappush(q,(cur,cnt,l)); cnt+=1; spawns.append((p,l,cur)); stack.append(r)
            else: stack+=[r,l]
    heapq.heappush(free,(cur,th))
end=max(b[3] for b in blocks)
# (a) icicle with letters, colour = thread
S.PW=34
H=2.6
S.title(0,r'\textbf{(a)} the tree')
def colour(p,n,d): return 'ink' if own[p]=='all' else cols[own[p]]
def lab(p,n,d,a,b,y,H): return [txt((a+b)/2,y-H/2,letters[p],extra=',text=white')] if b-a>1.6 else []
S.add(icicle(0,0,S.PW,H,colour,lab))
# (b) timeline
X0=S.PW+12; TW=122-X0-1; K=TW/end; LH=2.2
S.add(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at ({X0-8}mm,3.2mm) {{\textbf{{(b)}} four threads over time}};')
for i in range(4):
    y=-i*(LH+0.5)
    S.add(rect(X0,y,TW,LH,'black!4'))
    S.add(txt(X0-1,y-LH/2,f'thread {i+1}',anchor='east'))
for p,th,a,b in blocks:
    ths=range(4) if th=='all' else [th]
    for i in ths:
        y=-i*(LH+0.5); c='ink' if th=='all' else cols[th]
        S.add(rect(X0+a*K+0.15,y,(b-a)*K-0.3,LH,c))
        if (b-a)*K>1.5: S.add(txt(X0+(a+b)/2*K,y-LH/2,letters[p],extra=',text=white'))
for p,l,t in spawns:
    th=own[p]; tl=own[l]
    if th=='all' or tl=='all': continue
    st=[a for q,_,a,_ in blocks if q==l][0]
    y1=-th*(LH+0.5)-LH/2; y2=-tl*(LH+0.5)-LH/2
    if th!=tl: S.add(rf'\draw[-{{Latex[length=1.4mm,width=1.1mm]}},ink,line width=0.6pt] ({X0+t*K:.2f}mm,{y1:.2f}mm) to[bend left=10] ({X0+st*K+0.2:.2f}mm,{y2:.2f}mm);')
S.add(rf'\draw[-{{Latex[length=1.1mm]}},muted] ({X0}mm,{-4*(LH+0.5)-0.4}mm) -- ({X0+TW}mm,{-4*(LH+0.5)-0.4}mm) node[pos=0,anchor=east,font=\tiny,inner sep=0.8mm] {{time}};')
S.write('diagram-task-parallel')
