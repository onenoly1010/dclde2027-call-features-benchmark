"""cetacean-004. Same 55-file DCLDE 2027 DFO subset as 003. (1) supervised LR/RF on 003 baseline features;
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
print("features",len(ev),round(time.time()-t0),"s",flush=True)
META=("file","dep","day","sp","eco","bg_local_frames")
BASE=[k for k in ev[0] if k not in META and not k[:2] in("A_","L_")]
CA=[k for k in ev[0] if k.startswith("A_")];CL=[k for k in ev[0] if k.startswith("L_")]
for e in ev:
    for k in CA+CL: e.setdefault(k,0.0)
def M(keys,I):
    B=np.nan_to_num(np.array([[ev[i][k] for k in keys] for i in I],float));return np.sign(B)*np.log1p(np.abs(B))
FS={"trivial_duration_peakfreq":["duration_s","peak_freq_median_hz"],"baseline003":BASE,"call_AUTOband":CA,"baseline+call_AUTOband":BASE+CA,
    "call_LABELLEDband__LEAK":CL}
def clf(name):
    if name=="LR": return make_pipeline(StandardScaler(),LogisticRegression(class_weight="balanced",max_iter=3000,C=1.0))
    return RandomForestClassifier(300,class_weight="balanced_subsample",min_samples_leaf=3,n_jobs=8,random_state=SEED)
def scores(y,p,classes):
    pr,rc,_,n=PRF(y,p,labels=classes,zero_division=0)
    return dict(balanced_acc=float(BAC(y,p)),macro_recall=float(rc.mean()),accuracy=float((y==p).mean()),
      per_class={c:dict(precision=float(pr[i]),recall=float(rc[i]),n=int(n[i])) for i,c in enumerate(classes)})
def splits(y,days,deps):
    S=[("day5fold",list(GroupKFold(5).split(y,y,days)))]
    for a,b in (("NorthBc","WVanIsl"),("WVanIsl","NorthBc")): S.append((f"{a}->{b}",[(np.where(deps==a)[0],np.where(deps==b)[0])]))
    return S
def supervised(X,y,days,deps,classes):
    r={}
    for sn,sp in splits(y,days,deps):
        r[sn]={}
        for mn in ("majority","LR","RF"):
            P=np.empty(len(y),object);mask=np.zeros(len(y),bool)
            for tr,te in sp:
                if mn=="majority": P[te]=Counter(y[tr]).most_common(1)[0][0]
                else: P[te]=clf(mn).fit(X[tr],y[tr]).predict(X[te])
                mask[te]=True
            r[sn][mn]=scores(y[mask],P[mask].astype(str),classes)
    return r
def bestk(X):
    best=None
    for k in range(2,13):
        l=KMeans(k,random_state=SEED,n_init=10).fit_predict(X);s=float(SIL(X,l,sample_size=min(2000,len(X)),random_state=SEED))
        if best is None or s>best[2]: best=(k,l,s)
    return best
def unsup(X,y,files,days,deps,classes):
    Z=StandardScaler().fit_transform(X);k,l,s=bestk(Z);hd=HDBSCAN(min_cluster_size=10).fit_predict(Z);nn=hd>=0
    r=dict(full=dict(k=k,silhouette=s,ARI_label=ARI(y,l),NMI_label=NMI(y,l),ARI_file=ARI(files,l),ARI_day=ARI(days,l),ARI_deployment=ARI(deps,l),
      hdb_clusters=int(len(set(hd))-(-1 in hd)),hdb_coverage=float(nn.mean()),hdb_ARI_label=ARI(y[nn],hd[nn]) if nn.sum()>1 else None,
      hdb_ARI_file=ARI(files[nn],hd[nn]) if nn.sum()>1 else None,hdb_ARI_deployment=ARI(deps[nn],hd[nn]) if nn.sum()>1 else None))
    for sn,sp in splits(y,days,deps):
        P=np.empty(len(y),object);mask=np.zeros(len(y),bool);aris=[];cov=[];PH=np.empty(len(y),object)
        for tr,te in sp:
            sc=StandardScaler().fit(X[tr]);Xtr,Xte=sc.transform(X[tr]),sc.transform(X[te]);kk,_,_=bestk(Xtr)
            km=KMeans(kk,random_state=SEED,n_init=10).fit(Xtr);mp={c:Counter(y[tr][km.labels_==c]).most_common(1)[0][0] for c in set(km.labels_)}
            pt=km.predict(Xte);P[te]=[mp[c] for c in pt];aris.append(ARI(y[te],pt));mask[te]=True
            h=HDBSCAN(min_cluster_size=10).fit_predict(Xtr);mh={c:Counter(y[tr][h==c]).most_common(1)[0][0] for c in set(h) if c!=-1}
            ph=KNeighborsClassifier(5).fit(Xtr,h).predict(Xte);cov.append(float((ph!=-1).mean()));PH[te]=[mh.get(c,"NOISE") for c in ph]
        cv=mask&(PH!="NOISE")
        r[sn]=dict(km_ARI_mean=float(np.mean(aris)),km_mapped=scores(y[mask],P[mask].astype(str),classes),hdb_coverage=float(np.mean(cov)),
          hdb_mapped_covered=scores(y[cv],PH[cv].astype(str),classes) if cv.sum() else None)
    return r
res={}
for level in ("species","ecotype"):
    I=[i for i,e in enumerate(ev) if (e["sp"] in("HW","KW") if level=="species" else e["eco"] in("NRKW","TKW","OKW"))]
    y=np.array([ev[i]["sp" if level=="species" else "eco"] for i in I]);classes=sorted(set(y))
    files=np.array([ev[i]["file"] for i in I]);days=np.array([ev[i]["day"] for i in I]);deps=np.array([ev[i]["dep"] for i in I])
    res[level]=dict(n=len(I),counts=dict(Counter(y)),featuresets={})
    for fs,keys in FS.items():
        X=M(keys,I);r=dict(n_features=len(keys),leaks_annotator_info=fs.endswith("LEAK"),supervised=supervised(X,y,days,deps,classes))
        # deployment-prediction check: GroupKFold by day (deployment constant per day -> test days unseen)
        dp=np.empty(len(y),object)
        for tr,te in GroupKFold(5).split(X,deps,days): dp[te]=clf("RF").fit(X[tr],deps[tr]).predict(X[te])
        r["deployment_check_RF_day5fold"]=scores(deps,dp.astype(str),sorted(set(deps)))
        if fs!="trivial_duration_peakfreq" and fs!="baseline003": r["unsupervised"]=unsup(X,y,files,days,deps,classes)
        res[level]["featuresets"][fs]=r
        s=r["supervised"];print(level,fs,{sn:{m:round(s[sn][m]["balanced_acc"],3) for m in s[sn]} for sn in s},"dep",round(r["deployment_check_RF_day5fold"]["balanced_acc"],3),flush=True)
json.dump(dict(n_events=len(ev),band_auto_median_hz=[float(np.median([e["A_band_lo_hz"] for e in ev])),float(np.median([e["A_band_hi_hz"] for e in ev]))],
  local_bg_fallback_frac=float(np.mean([e["bg_local_frames"]<10 for e in ev])),results=res,runtime_s=time.time()-t0,seed=SEED,sr=SR),
  open(f"{out}/results.json","w"),indent=1,default=float)
print("done",round(time.time()-t0),"s")
