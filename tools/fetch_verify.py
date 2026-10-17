#!/usr/bin/env python3
"""Fetch (or locate locally) every audio file in FILE_MANIFEST.csv, verify SHA-256 + GCS md5, and lay out the
data folders the run scripts expect. Standard library only. Audio is NOT shipped in this repo."""
import argparse,csv,hashlib,base64,os,sys,shutil,urllib.request
ANN_URL="https://storage.googleapis.com/noaa-passive-bioacoustic/dclde/2027/dclde_2027_killer_whales/Annotations.csv"
ANN_SHA="407f1c0f7d95a39c7bace99bbef40646e2fe62403c6237c20ddf374e88e21b15"
def hashes(p):
    s,m=hashlib.sha256(),hashlib.md5()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): s.update(b);m.update(b)
    return s.hexdigest(),base64.b64encode(m.digest()).decode()
def place(src,dst):
    os.makedirs(os.path.dirname(dst),exist_ok=True)
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(os.path.abspath(src),dst)
ap=argparse.ArgumentParser();ap.add_argument("--manifest",default="FILE_MANIFEST.csv");ap.add_argument("--work",default="work")
ap.add_argument("--local-audio",help="directory tree that already holds the FLAC files (searched by file name); no download");
ap.add_argument("--local-annotations",help="local copy of DCLDE 2027 Annotations.csv");ap.add_argument("--no-annotations",action="store_true")
a=ap.parse_args();rows=list(csv.DictReader(open(a.manifest)));store=os.path.join(a.work,"audio_store")
local={}
if a.local_audio:
    for dp,_,fs in os.walk(a.local_audio,followlinks=True):
        for f in fs:
            if f.endswith(".flac"): local.setdefault(f,os.path.join(dp,f))
ok=bad=missing=0;fails=[]
for r in rows:
    name=os.path.basename(r["file"]);sub=r["file"].split("/")[-2];dst=os.path.join(store,sub,name)
    if a.local_audio:
        if name not in local: missing+=1;fails.append(("missing",name));continue
        place(local[name],dst)
    elif not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst),exist_ok=True);tmp=dst+".part";urllib.request.urlretrieve(r["url"],tmp);os.replace(tmp,dst)
    sha,md5=hashes(dst)
    if sha==r["sha256"] and md5==r["gcs_md5"]: ok+=1
    else: bad+=1;fails.append(("hash_mismatch",name))
print(f"audio: {len(rows)} in manifest | verified {ok} | mismatched {bad} | missing {missing}")
for x in fails[:20]: print("  FAIL",*x)
if bad or missing: sys.exit(1)
# layouts expected by the as-run scripts (paths are rewritten by reproduce.sh)
s003=set(l.strip().strip('",') for l in open("data/run003_subset/selected_files.json") if l.strip().strip('",[]'))
for d,key in (("cetacean-003/data","run003_subset"),("cetacean-005/data","run005_subset")):
    root=os.path.join(a.work,d);os.makedirs(root,exist_ok=True)
    for f in ("MANIFEST.csv","annotations_subset.csv","selected_files.json"): shutil.copy(f"data/{key}/{f}",os.path.join(root,f))
for r in rows:
    name=os.path.basename(r["file"]);sub=r["file"].split("/")[-2];src=os.path.join(store,sub,name)
    place(src,os.path.join(a.work,"cetacean-005/data/audio",sub,name))
    if name in s003: place(src,os.path.join(a.work,"cetacean-003/data/dclde2027_kw/audio",sub,name))
n003=sum(len(fs) for _,_,fs in os.walk(os.path.join(a.work,"cetacean-003/data/dclde2027_kw/audio")))
print(f"layout: cetacean-003 files {n003} (expect 55) | cetacean-005 files {len(rows)}")
if n003!=55: sys.exit(1)
if not a.no_annotations:
    dst=os.path.join(a.work,"cetacean-003/data/dclde2027_kw/Annotations.csv")
    if a.local_annotations: place(a.local_annotations,dst)
    elif not os.path.exists(dst): urllib.request.urlretrieve(ANN_URL,dst)
    sha=hashes(dst)[0];print("Annotations.csv sha256",("OK" if sha==ANN_SHA else "MISMATCH "+sha))
    if sha!=ANN_SHA: sys.exit(1)
print("verification OK")
