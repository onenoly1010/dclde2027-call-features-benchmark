import soundfile as sf,hashlib,json,csv,glob,os,collections
D='/workspace/cetacean-003/data'
sel=set(json.load(open(D+'/selected_files.json')))
rows=[x for x in csv.DictReader(open(D+'/dclde2027_kw/Annotations.csv')) if x['Soundfile'] in sel]
def lab(x):
    if x['ClassSpecies']=='KW': return 'KW_'+(x['Ecotype'] if x['Ecotype']!='NA' else 'unk')
    return x['ClassSpecies']
man=[];bad=[]
info={}
for p in sorted(glob.glob(D+'/dclde2027_kw/audio/*/*.flac')):
    i=sf.info(p); h=hashlib.sha256(open(p,'rb').read()).hexdigest()
    f=os.path.basename(p); dep=f.split('.')[1]; rec=f.split('.')[2]; day=f.split('.')[3][:8]
    info[f]=(i.samplerate,i.duration)
    man.append(dict(file=os.path.relpath(p,D),sha256=h,bytes=os.path.getsize(p),samplerate=i.samplerate,channels=i.channels,frames=i.frames,duration_s=round(i.duration,3),subtype=i.subtype,deployment=dep,recorder=rec,session_day=day))
with open(D+'/MANIFEST.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(man[0]));w.writeheader();w.writerows(man)
# annotation subset + checks
oob=0;nyq=0
with open(D+'/annotations_subset.csv','w',newline='') as fh:
    w=csv.writer(fh);w.writerow(['Soundfile','Dataset','deployment','session_day','label','ClassSpecies','Ecotype','KW_certain','FileBeginSec','FileEndSec','LowFreqHz','HighFreqHz','UTC'])
    for x in rows:
        sr,dur=info[x['Soundfile']]
        b,e=float(x['FileBeginSec']),float(x['FileEndSec'])
        if not(0<=b<e<=dur+0.01): oob+=1
        if float(x['HighFreqHz'])>sr/2+1: nyq+=1
        w.writerow([x['Soundfile'],x['Dataset'],x['Soundfile'].split('.')[1],x['Soundfile'].split('.')[3][:8],lab(x),x['ClassSpecies'],x['Ecotype'],x['KW_certain'],b,e,x['LowFreqHz'],x['HighFreqHz'],x['UTC']])
S=collections.defaultdict(collections.Counter);days=collections.defaultdict(lambda:collections.defaultdict(set));fd=collections.defaultdict(set)
for x in rows:
    d=x['Dataset'];S[d][lab(x)]+=1;days[d][lab(x)].add(x['Soundfile'].split('.')[3][:8]);fd[d].add(x['Soundfile'].split('.')[3][:8])
summary={'files':len(man),'bytes':sum(m['bytes'] for m in man),'audio_hours':round(sum(m['duration_s'] for m in man)/3600,2),
 'samplerates':dict(collections.Counter((m['deployment'],m['recorder'],m['samplerate'],m['channels'],m['subtype']) and f"{m['deployment']}|{m['recorder']}|{m['samplerate']}Hz|{m['channels']}ch|{m['subtype']}" for m in man)),
 'durations_s':dict(collections.Counter(f"{m['deployment']}|{round(m['duration_s'])}" for m in man)),
 'events':len(rows),'events_out_of_file_bounds':oob,'events_above_nyquist':nyq,
 'per_deployment':{d:{'session_days':len(fd[d]),'labels':{k:{'events':n,'days':len(days[d][k])} for k,n in S[d].most_common()}} for d in S},
 'multi_label_files':sum(1 for f in sel if len({lab(x) for x in rows if x['Soundfile']==f})>=2)}
json.dump(summary,open(D+'/SUBSET_SUMMARY.json','w'),indent=1);print(json.dumps(summary,indent=1))
