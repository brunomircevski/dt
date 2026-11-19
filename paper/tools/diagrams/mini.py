"""A small fixed tree drawn as an icicle (row = level, box width = samples)."""
# a small fixed tree drawn as an icicle; colour(node) -> tikz fill
T=(32,[(20,[(13,[(9,[]),(4,[])]),(7,[(5,[]),(2,[])])]),(12,[(4,[]),(8,[(5,[]),(3,[])])])])
def nodes(t=T,d=0,x=0,path=''):
    n,k=t; yield (path,n,d,x)
    for i,c in enumerate(k):
        yield from nodes(c,d+1,x,path+str(i)); x+=c[0]
def icicle(x0,y0,W,H,colour,extra=None):
    o=[]
    for p,n,d,x in nodes():
        a=x0+x/32*W; b=x0+(x+n)/32*W; y=y0-d*H
        col=colour(p,n,d)
        o.append(rf'\fill[{col}] ({a:.2f}mm,{y:.2f}mm) rectangle ({b:.2f}mm,{y-H:.2f}mm);')
        if col=='STRIPE':
            o[-1]=''
            w=(b-a)/4
            for i,c in enumerate(['tA','tB','tC','tD']):
                o.append(rf'\fill[{c}] ({a+i*w:.2f}mm,{y:.2f}mm) rectangle ({a+(i+1)*w:.2f}mm,{y-H:.2f}mm);')
        o.append(rf'\draw[white,line width=0.6pt] ({a:.2f}mm,{y:.2f}mm) rectangle ({b:.2f}mm,{y-H:.2f}mm);')
        if extra: o+=extra(p,n,d,a,b,y,H)
    return o
