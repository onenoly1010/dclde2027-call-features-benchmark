"""Deterministic subset selection from DCLDE 2027 DFO_CRP annotations (no randomness)."""
import csv,collections,json,sys
A='/workspace/cetacean-003/data/dclde2027_kw/Annotations.csv'
rows=[x for x in csv.DictReader(open(A)) if x['Provider']=='DFO_CRP']
def lab(x):
    if x['ClassSpecies']=='KW': return 'KW_'+x['Ecotype'] if x['Ecotype'] in ('NRKW','TKW','OKW','SRKW') else 'KW_unk'
    return x['ClassSpecies']
per=collections.defaultdict(collections.Counter)
ds={}
for x in rows: per[x['Soundfile']][lab(x)]+=1; ds[x['Soundfile']]=x['Dataset']
targets=['KW_NRKW','KW_TKW','KW_OKW','HW']
DAYS=8  # max distinct days per (deployment,class)
sel=set()
for d in ('NorthBc','WVanIsl'):
    for t in targets:
        cands=[f for f in per if ds[f]==d and per[f][t]>=5]
        # prefer files with more distinct classes, then more target events, then name
        cands.sort(key=lambda f:(-len([k for k in per[f] if k in targets]),-per[f][t],f))
        days=set()
        for f in cands:
            day=f.split('.')[-2][:8]
            if day in days: continue
            days.add(day); sel.add(f)
            if len(days)>=DAYS: break
json.dump(sorted(sel),open('/workspace/cetacean-003/data/selected_files.json','w'),indent=0)
print(len(sel))
