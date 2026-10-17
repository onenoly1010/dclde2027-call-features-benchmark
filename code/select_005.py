"""Deterministic stratified selection for cetacean-005 (no randomness). Keeps the 55 cetacean-003 files and tops each
deployment up to 150 files. Round-robin over classes (HW, NRKW, TKW, OKW); each pick prefers an unused day, then files
with both HW and KW, then more target classes, then more events of that class, then name."""
import csv,collections,json
A='/workspace/cetacean-003/data/dclde2027_kw/Annotations.csv'
rows=[x for x in csv.DictReader(open(A)) if x['Provider']=='DFO_CRP']
def lab(x):
    if x['ClassSpecies']=='KW': return x['Ecotype'] if x['Ecotype'] in('NRKW','TKW','OKW') else 'KWo'
    return x['ClassSpecies']
per=collections.defaultdict(collections.Counter);ds={}
for x in rows: per[x['Soundfile']][lab(x)]+=1;ds[x['Soundfile']]=x['Dataset']
old=set(json.load(open('/workspace/cetacean-003/data/selected_files.json')))
T=['HW','NRKW','TKW','OKW'];day=lambda f:f.split('.')[-2][:8];both=lambda f:per[f]['HW']>0 and sum(per[f][k] for k in('NRKW','TKW','OKW','KWo'))>0
sel=set(old)
for d in('NorthBc','WVanIsl'):
    mine=lambda:[f for f in sel if ds.get(f)==d]
    used=collections.Counter(day(f) for f in mine())
    while len(mine())<150:
        progress=False
        for t in T:
            if len(mine())>=150: break
            c=[f for f in per if ds[f]==d and f not in sel and per[f][t]>=3]
            if not c: continue
            c.sort(key=lambda f:(used[day(f)],not both(f),-sum(per[f][k]>0 for k in T),-per[f][t],f))
            sel.add(c[0]);used[day(c[0])]+=1;progress=True
        if not progress: break
json.dump(sorted(sel),open('/workspace/cetacean-005/data/selected_files.json','w'),indent=0)
for d in('NorthBc','WVanIsl'):
    F=[f for f in sel if ds[f]==d];print(d,len(F),'days',len({day(f) for f in F}),'both',sum(both(f) for f in F),{t:(sum(per[f][t] for f in F),len({day(f) for f in F if per[f][t]})) for t in T})
