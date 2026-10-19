import re,sys,statistics,collections
rows=collections.defaultdict(list); trees=collections.defaultdict(set)
for f in sys.argv[1:]:
    node='hopper' if 'hopper' in f else 'ampere'
    for line in open(f):
        m=re.match(r'^(base|new5)\s+(\S+) (.*?)\s+train\s+([\d.]+).*?levels\s+([\d.]+).*tree (\w+)',line)
        if m:
            v,d,p,t,l,tr=m.groups()
            rows[(node,d,p.strip(),v)].append(float(t)); trees[(node,d,p.strip())].add(tr)
print('| node | case | base (s) | staged (s) | change | same tree |')
print('|---|---|--:|--:|--:|---|')
for key in sorted({k[:3] for k in rows}):
    b=rows[key+('base',)]; n=rows[key+('new5',)]
    if not b or not n: continue
    mb,mn=statistics.median(b),statistics.median(n)
    print('| %s | %s %s | %.2f (%s) | %.2f (%s) | %+.1f%% | %s |'%(key[0],key[1],key[2],mb,' '.join('%.2f'%x for x in b),mn,' '.join('%.2f'%x for x in n),(mn/mb-1)*100,'yes' if len(trees[key])==1 else 'NO'))
