import json,os,hashlib,base64,urllib.request,csv,soundfile as sf,shutil
D='/workspace/cetacean-005/data';OLD='/workspace/cetacean-003/data/dclde2027_kw/audio'
L={os.path.basename(o['name']):o for o in json.load(open(D+'/gcs_listing.json'))}
sel=json.load(open(D+'/selected_files.json'));man=[]
for f in sel:
    o=L[f];sub=o['name'].split('/')[-2];p=f"{D}/audio/{sub}/{f}";os.makedirs(os.path.dirname(p),exist_ok=True)
    url="https://storage.googleapis.com/noaa-passive-bioacoustic/"+o['name']
    if not os.path.exists(p):
        if os.path.exists(f"{OLD}/{sub}/{f}"): shutil.copy(f"{OLD}/{sub}/{f}",p)
        else: urllib.request.urlretrieve(url,p)
    b=open(p,'rb').read();md5=base64.b64encode(hashlib.md5(b).digest()).decode();i=sf.info(p)
    man.append(dict(file=f"audio/{sub}/{f}",url=url,gcs_generation=o['generation'],gcs_md5=o['md5Hash'],md5_ok=md5==o['md5Hash'],size_ok=len(b)==int(o['size']),
      sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),samplerate=i.samplerate,channels=i.channels,duration_s=round(i.duration,3),
      deployment='NorthBc' if sub=='northbc' else 'WVanIsl',session_day=f.split('.')[3][:8],in_cetacean003=os.path.exists(f"{OLD}/{sub}/{f}")))
w=csv.DictWriter(open(D+'/MANIFEST.csv','w',newline=''),fieldnames=list(man[0]));w.writeheader();w.writerows(man)
print(len(man),'md5_ok',sum(m['md5_ok'] for m in man),'size_ok',sum(m['size_ok'] for m in man),'bytes',sum(m['bytes'] for m in man),
  {(m['deployment'],m['samplerate'],round(m['duration_s'])) for m in man})
