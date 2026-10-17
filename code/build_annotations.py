import csv,collections,json
D='/workspace/cetacean-005/data';man={m['file'].split('/')[-1]:m for m in csv.DictReader(open(D+'/MANIFEST.csv'))}
rows=[x for x in csv.DictReader(open('/workspace/cetacean-003/data/dclde2027_kw/Annotations.csv')) if x['Soundfile'] in man]
oob=nyq=0;w=csv.writer(open(D+'/annotations_subset.csv','w',newline=''))
w.writerow(['Soundfile','Dataset','session_day','ClassSpecies','Ecotype','FileBeginSec','FileEndSec','LowFreqHz','HighFreqHz'])
for x in rows:
    m=man[x['Soundfile']];b,e=float(x['FileBeginSec']),float(x['FileEndSec'])
    if not(0<=b<e<=float(m['duration_s'])+0.01): oob+=1;continue
    if float(x['HighFreqHz'])>int(m['samplerate'])/2+1: nyq+=1
    w.writerow([x['Soundfile'],x['Dataset'],m['session_day'],x['ClassSpecies'],x['Ecotype'],b,e,x['LowFreqHz'],x['HighFreqHz']])
short=[f for f,m in man.items() if float(m['duration_s'])<200]
print(len(rows),'oob_dropped',oob,'nyq',nyq,'short_files',len(short),collections.Counter((x['Dataset'],x['ClassSpecies'],x['Ecotype'] if x['ClassSpecies']=='KW' else '') for x in rows))
