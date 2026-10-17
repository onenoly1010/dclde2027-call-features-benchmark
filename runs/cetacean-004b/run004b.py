"""cetacean-004b (feature extraction copied verbatim from run004.py). cetacean-004. Same 55-file DCLDE 2027 DFO subset as 003. (1) supervised LR/RF on 003 baseline features;
(2) call-focused features: band-limited, background-subtracted (adjacent non-event audio), peak-track contour.
Band source: AUTO (estimated from audio) or LABELLED (annotator box -> LEAKS annotator info, flagged).
Local only. Seed 0. Clusters/classifiers find acoustic patterns, not meaning."""
import sys,csv,json,os,time,numpy as np,librosa,warnings
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans,HDBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (silhouette_score as SIL,adjusted_rand_score as ARI,normalized_mutual_info_score as NMI,
  balanced_accuracy_score as BAC,precision_recall_fscore_support as PRF)
warnings.filterwarnings("ignore");SEED=0;SR=16000;NF=2048;HOP=256;np.random.seed(SEED)
D="/workspace/cetacean-003/data";out=sys.argv[1];t0=time.time()
man={os.path.basename(m["file"]):m for m in csv.DictReader(open(f"{D}/MANIFEST.csv"))}
ann=list(csv.DictReader(open(f"{D}/annotations_subset.csv")));byf={}
for a in ann: byf.setdefault(a["Soundfile"],[]).append(a)
freqs=librosa.fft_frequencies(sr=SR,n_fft=NF);fps=SR/HOP
def contour(S,bandmask,bgcol):
    """S: magnitude (F,T). background-subtract, restrict to band, peak track over frames."""
    P=np.maximum(S**2-bgcol[:,None]**2,0)[bandmask];fb=freqs[bandmask]
    if P.shape[0]<2: return None
    e=P.sum(0);keep=e>0.1*e.max() if e.max()>0 else np.ones(len(e),bool)
    pk=fb[P.argmax(0)][keep]
    if len(pk)<2: pk=np.r_[pk,pk] if len(pk) else np.array([fb.mean()]*2)
    lp=np.log2(np.maximum(pk,1));t=np.arange(len(lp))/fps
    ms=P.mean(1);ms=ms/(ms.sum()+1e-20);c=float((fb*ms).sum())
    shape=np.interp(np.linspace(0,1,8),np.linspace(0,1,len(lp)),lp)-np.median(lp)
    snr=10*np.log10((S[bandmask]**2).mean()/((bgcol[bandmask]**2).mean()+1e-20)+1e-20)
    f=dict(c_start_hz=pk[0],c_end_hz=pk[-1],c_min_hz=pk.min(),c_max_hz=pk.max(),c_med_hz=np.median(pk),
      c_range_oct=lp.max()-lp.min(),c_slope_oct_s=np.polyfit(t,lp,1)[0] if len(lp)>2 else 0.0,c_curv=np.polyfit(np.linspace(0,1,len(lp)),lp,2)[0] if len(lp)>3 else 0.0,
      c_std_oct=lp.std(),c_absrate_oct_s=np.median(np.abs(np.diff(lp)))*fps,c_jump_frac=(np.abs(np.diff(lp))>0.25).mean(),
      c_up_frac=(np.diff(lp)>0).mean(),c_voiced_frac=keep.mean(),b_centroid_hz=c,b_bw_hz=np.sqrt(((fb-c)**2*ms).sum()),
      b_flatness=float(np.exp(np.log(ms+1e-20).mean())/(ms.mean()+1e-20)),b_snr_db=snr)
    for i,v in enumerate(shape): f[f"c_shape{i}"]=v
    return {k:float(v) for k,v in f.items()}
ev=[]
for fn,rows in sorted(byf.items()):
    y,_=librosa.load(f"{D}/{man[fn]['file']}",sr=SR,mono=True)
    Sf=np.abs(librosa.stft(y,n_fft=NF,hop_length=HOP));T=Sf.shape[1]
    occ=np.zeros(T,bool)
    for a in rows: occ[int(float(a["FileBeginSec"])*fps):int(float(a["FileEndSec"])*fps)+1]=True
    filebg=np.median(Sf[:,~occ] if (~occ).sum()>20 else Sf,axis=1)
    for a in rows:
        s=float(a["FileBeginSec"]);e=min(float(a["FileEndSec"]),s+10.0);seg=y[int(s*SR):int(e*SR)]
        if len(seg)<NF: seg=np.pad(seg,(0,NF-len(seg)))
        S=np.abs(librosa.stft(seg,n_fft=NF,hop_length=HOP))
        mf=librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S**2,sr=SR)),n_mfcc=13)
        pk=freqs[S.argmax(0)];ons=librosa.onset.onset_detect(y=seg,sr=SR,units="time")
        f=dict(file=fn,dep=a["Dataset"],day=a["Dataset"]+"_"+a["session_day"],sp=a["ClassSpecies"],eco=a["Ecotype"] if a["ClassSpecies"]=="KW" else "",
          duration_s=e-s,centroid_hz=float(librosa.feature.spectral_centroid(S=S,sr=SR).mean()),bandwidth_hz=float(librosa.feature.spectral_bandwidth(S=S,sr=SR).mean()),
          rolloff_hz=float(librosa.feature.spectral_rolloff(S=S,sr=SR).mean()),flatness=float(librosa.feature.spectral_flatness(S=S).mean()),
          zcr=float(librosa.feature.zero_crossing_rate(seg).mean()),peak_freq_median_hz=float(np.median(pk)),
          peak_freq_slope_hz_s=float(np.polyfit(np.arange(len(pk))*HOP/SR,pk,1)[0]) if len(pk)>1 else 0.0,onset_rate_per_s=len(ons)/max(e-s,1e-3))
        for k in range(13): f[f"mfcc{k}"]=float(mf[k].mean())
        # local background: up to 3 s each side, non-event frames only; fallback file-level non-event median
        a0,a1=int(s*fps),int(e*fps);idx=np.r_[max(0,a0-int(3*fps)):a0,a1+1:min(T,a1+1+int(3*fps))];idx=idx[~occ[idx]] if len(idx) else idx
        bg=np.median(Sf[:,idx],axis=1) if len(idx)>=10 else filebg;f["bg_local_frames"]=int(len(idx))
        # AUTO band: bins where bg-subtracted mean power within 10 dB of its max, above 200 Hz
        Pm=np.maximum((S**2).mean(1)-bg**2,0);Pm[freqs<200]=0
        if Pm.max()>0:
            ok=np.where(10*np.log10(Pm/Pm.max()+1e-20)>-10)[0];alo,ahi=freqs[ok.min()],freqs[ok.max()]
            if ahi-alo<100: alo,ahi=max(200,alo-50),ahi+50
        else: alo,ahi=200,SR/2
        for tag,(lo,hi) in (("A",(alo,ahi)),("L",(float(a["LowFreqHz"]),float(a["HighFreqHz"])))):
            m=(freqs>=lo)&(freqs<=hi)
            if m.sum()<2: m=(freqs>=lo-31.25)&(freqs<=hi+31.25)
            c=contour(S,m,bg) or {}
            for k,v in c.items(): f[f"{tag}_{k}"]=v
        f["A_band_lo_hz"],f["A_band_hi_hz"]=float(alo),float(ahi)
        ev.append(f)

import inspect
META=("file","dep","day","sp","eco","bg_local_frames")
BASE=[k for k in ev[0] if k not in META and not k[:2] in("A_","L_")];CA=[k for k in ev[0] if k.startswith("A_")]
for e in ev:
    for k in CA: e.setdefault(k,0.0)
def M(keys,I):
    B=np.nan_to_num(np.array([[ev[i][k] for k in keys] for i in I],float));return np.sign(B)*np.log1p(np.abs(B))
FS={"trivial_duration_peakfreq":["duration_s","peak_freq_median_hz"],"baseline003":BASE,"call_AUTOband":CA}
def clf(name,seed,ntree=300):
    if name=="LR": return make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=3000))
    return RandomForestClassifier(ntree,class_weight="balanced_subsample",min_samples_leaf=3,n_jobs=8,random_state=seed)
def splits(y,days,deps,seed):
    return {"day5fold":list(GroupKFold(5,shuffle=True,random_state=seed).split(y,y,days)),
     "NorthBc->WVanIsl":[(np.where(deps=="NorthBc")[0],np.where(deps=="WVanIsl")[0])],
     "WVanIsl->NorthBc":[(np.where(deps=="WVanIsl")[0],np.where(deps=="NorthBc")[0])]}
def oof(X,y,sp,mn,seed,ntree=300):
    P=np.empty(len(y),object);m=np.zeros(len(y),bool)
    for tr,te in sp: P[te]=clf(mn,seed,ntree).fit(X[tr],y[tr]).predict(X[te]);m[te]=True
    return P,m
def boot(y,P,days,rng,B=2000):
    ud=np.unique(days);idx={d:np.where(days==d)[0] for d in ud};v=[]
    for _ in range(B):
        ii=np.concatenate([idx[d] for d in rng.choice(ud,len(ud))])
        if len(set(y[ii]))<len(set(y)): continue
        v.append(BAC(y[ii],P[ii].astype(str)))
    return [float(np.percentile(v,2.5)),float(np.percentile(v,97.5))]
NPERM=int(os.environ.get("NPERM",200));res={}
plan={"species":[("trivial_duration_peakfreq",["LR","RF"]),("baseline003",["LR","RF"]),("call_AUTOband",["LR","RF"])],
      "ecotype":[("trivial_duration_peakfreq",["RF"]),("baseline003",["LR"]),("call_AUTOband",["RF"])]}
PERM={("species","call_AUTOband","LR"),("species","call_AUTOband","RF"),("ecotype","call_AUTOband","RF")}
for level,items in plan.items():
    I=[i for i,e in enumerate(ev) if (e["sp"] in("HW","KW") if level=="species" else e["eco"] in("NRKW","TKW","OKW"))]
    y=np.array([ev[i]["sp" if level=="species" else "eco"] for i in I]);days=np.array([ev[i]["day"] for i in I]);deps=np.array([ev[i]["dep"] for i in I])
    res[level]={}
    for fs,models in items:
        X=M(FS[fs],I)
        for mn in models:
            r={};key=f"{fs}|{mn}"
            for sn in ("day5fold","NorthBc->WVanIsl","WVanIsl->NorthBc"):
                seeds=[];
                for sd in range(5):
                    P,m=oof(X,y,splits(y,days,deps,sd)[sn],mn,sd);seeds.append(float(BAC(y[m],P[m].astype(str))))
                    if sd==0: ci=boot(y[m],P[m],days[m],np.random.default_rng(0));P0,m0=P,m
                r[sn]=dict(bal_acc_seeds=seeds,mean=float(np.mean(seeds)),sd=float(np.std(seeds)),seed0=seeds[0],ci95_day_bootstrap_seed0=ci)
                if (level,fs,mn) in PERM:
                    rng=np.random.default_rng(1);null=[]
                    for _ in range(NPERM):
                        yp=y.copy()
                        for d in np.unique(deps): ii=np.where(deps==d)[0];yp[ii]=rng.permutation(y[ii])
                        P,m=oof(X,yp,splits(y,days,deps,0)[sn],mn,0,ntree=100);null.append(float(BAC(yp[m],P[m].astype(str))))
                    r[sn]["perm"]=dict(n=NPERM,null_mean=float(np.mean(null)),null_p95=float(np.percentile(null,95)),
                        p_value=float((1+sum(v>=seeds[0] for v in null))/(1+NPERM)),note="labels shuffled at event level within deployment; RF perms use 100 trees")
                print(level,key,sn,{k:v for k,v in r[sn].items() if k!="bal_acc_seeds"},flush=True)
            res[level][key]=r
json.dump(dict(results=res,runtime_s=time.time()-t0,nperm=NPERM),open(f"{out}/results.json","w"),indent=1,default=float)
print("done",round(time.time()-t0))
