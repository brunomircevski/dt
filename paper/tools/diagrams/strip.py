"""Helpers for the schematic diagrams: a row of panels drawn with TikZ (standalone PDF,
LNCS text width), titles, notes and CPU/GPU role lines."""
import subprocess
class Strip:
    def __init__(s,pw=27.5,gap=3.6,note_y=-15.2):
        s.o=[]; s.PW=pw; s.GAP=gap; s.NY=note_y
    def px(s,i): return i*(s.PW+s.GAP)
    def title(s,i,t,y=3.2): s.o.append(rf'\node[font=\scriptsize,anchor=south west,inner sep=0] at ({s.px(i)}mm,{y}mm) {{{t}}};')
    def note(s,i,t,y=None):
        y=s.NY if y is None else y
        s.o.append(rf'\node[font=\tiny,anchor=north west,inner sep=0,text width={s.PW}mm,align=flush left] at ({s.px(i)}mm,{y}mm) {{{t}}};')
    def lanes(s,i,cpu,gpu,y=-11.3):
        for k,(lab,txt,col) in enumerate([('CPU',cpu,'tA'),('GPU',gpu,'gpu')]):
            if txt is None: continue
            yy=y-k*1.8
            s.o.append(rf'\node[font=\tiny,anchor=west,inner sep=0,text={col}] at ({s.px(i)}mm,{yy}mm) {{\textbf{{{lab}}}}};')
            s.o.append(rf'\node[font=\tiny,anchor=west,inner sep=0] at ({s.px(i)+6.3}mm,{yy}mm) {{{txt}}};')
    def add(s,x):
        if isinstance(x,str): s.o.append(x)
        else: s.o+=x
    def write(s,name):
        """Build name.pdf with pdflatex in a temporary directory, then copy it to paper/figs/."""
        import os, shutil, tempfile
        here=os.path.dirname(os.path.abspath(__file__))
        out=os.path.join(here,'..','..','figs')
        tex=r"""\documentclass[border=1pt]{standalone}
\input{common}
\begin{document}\footnotesize
\begin{tikzpicture}
"""+'\n'.join(s.o)+r"""
\end{tikzpicture}
\end{document}"""
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp,name+'.tex'),'w').write(tex)
            shutil.copy(os.path.join(here,'common.tex'),tmp)
            r=subprocess.run(['pdflatex','-interaction=nonstopmode',name+'.tex'],cwd=tmp,capture_output=True,text=True)
            if r.returncode: raise SystemExit(r.stdout[-2000:])
            shutil.copy(os.path.join(tmp,name+'.pdf'),os.path.join(out,name+'.pdf'))
        print('wrote', os.path.normpath(os.path.join(out,name+'.pdf')))
def rect(x,y,w,h,col,extra=''):
    return rf'\fill[{col}{extra}] ({x:.2f}mm,{y:.2f}mm) rectangle ++({w:.2f}mm,{-h:.2f}mm);'
def txt(x,y,t,font=r'\tiny',anchor='center',extra=''):
    return rf'\node[font={font},anchor={anchor},inner sep=0{extra}] at ({x:.2f}mm,{y:.2f}mm) {{{t}}};'
