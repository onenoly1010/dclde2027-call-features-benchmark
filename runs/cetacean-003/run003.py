"""cetacean-003. DCLDE 2027 DFO subset. Features = cetacean-002 BASE / X_ (deltas) extractor, computed on the
human-annotated event boxes (time only; annotated freq bounds NOT used as features). Both deployments resampled
to 16 kHz. Clusters are acoustic pattern groups, not meaning. Seed 0. Local only."""
import sys,csv,json,os,time,numpy as np,librosa,warnings
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans,HDBSCAN
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score as SIL,adjusted_rand_score as ARI,normalized_mutual_info_score as NMI
warnings.filterwarnings("ignore");SEED=0;SR=16000;np.random.seed(SEED)
D="/workspace/cetacean-003/data";out=sys.argv[1];t0=time.time()
man={os.path.basename(m["file"]):m for m in csv.DictReader(open(f"{D}/MANIFEST.csv"))}
ann=list(csv.DictReader(open(f"{D}/annotations_subset.csv")))
ev=[];byf={}
for a in ann: byf.setdefault(a["Soundfile"],[]).append(a)
for fn,rows in sorted(byf.items()):
    y,_=librosa.load(f"{D}/{man[fn]['file']}",sr=SR,mono=True)   # documented resample 16384->16000 for WVanIsl
    for a in rows:
        s=float(a["FileBeginSec"]);e=min(float(a["FileEndSec"]),s+10.0);seg=y[int(s*SR):int(e*SR)]
        if len(seg)<2048: seg=np.pad(seg,(0,2048-len(seg)))
        S=np.abs(librosa.stft(seg,n_fft=2048,hop_length=256))
        mf=librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S**2,sr=SR)),n_mfcc=13)
        freqs=librosa.fft_frequencies(sr=SR,n_fft=2048);pk=freqs[S.argmax(0)];ons=librosa.onset.onset_detect(y=seg,sr=SR,units="time")
        f=dict(file=fn,dep=a["Dataset"],day=a["Dataset"]+"_"+a["session_day"],label=a["label"],sp=a["ClassSpecies"],eco=a["Ecotype"] if a["ClassSpecies"]=="KW" else "",
          duration_s=e-s,centroid_hz=float(librosa.feature.spectral_centroid(S=S,sr=SR).mean()),
          bandwidth_hz=float(librosa.feature.spectral_bandwidth(S=S,sr=SR).mean()),rolloff_hz=float(librosa.feature.spectral_rolloff(S=S,sr=SR).mean()),
          flatness=float(librosa.feature.spectral_flatness(S=S).mean()),zcr=float(librosa.feature.zero_crossing_rate(seg).mean()),
          peak_freq_median_hz=float(np.median(pk)),peak_freq_slope_hz_s=float(np.polyfit(np.arange(len(pk))*256/SR,pk,1)[0]) if len(pk)>1 else 0.0,
          onset_rate_per_s=len(ons)/max(e-s,1e-3))
        for k in range(13): f[f"mfcc{k}"]=float(mf[k].mean())
        n=mf.shape[1];w=min(9,n if n%2 else n-1)
        if w>=3: d1=librosa.feature.delta(mf,width=w,mode="nearest");d2=librosa.feature.delta(mf,order=2,width=w,mode="nearest")
        else: d1=d2=np.zeros_like(mf)
        for k in range(13):
            f[f"X_mfcc{k}_std"]=float(mf[k].std());f[f"X_d1_{k}_mean"]=float(d1[k].mean());f[f"X_d1_{k}_std"]=float(d1[k].std())
            f[f"X_d2_{k}_mean"]=float(d2[k].mean());f[f"X_d2_{k}_std"]=float(d2[k].std())
        ev.append(f)
print("features",len(ev),round(time.time()-t0),"s",flush=True)
META=("file","dep","day","label","sp","eco")
BASE=[k for k in ev[0] if k not in META and not k.startswith("X_")];EXT=[k for k in ev[0] if k.startswith("X_")]
A=lambda keys,I:np.sign(B:=np.array([[ev[i][k] for k in keys] for i in I],float))*np.log1p(np.abs(B))
def filenorm(X,files):
    X=X.copy()
    for r in np.unique(files):
        i=files==r;mu=np.median(X[i],0);sd=X[i].std(0);sd[sd<1e-9]=1.0;X[i]=(X[i]-mu)/sd if i.sum()>1 else X[i]-mu
    return X
def bestk(X,ks=range(2,13)):
    best=None;sil={}
    for k in ks:
        l=KMeans(k,random_state=SEED,n_init=10).fit_predict(X);sil[k]=float(SIL(X,l,sample_size=min(2000,len(X)),random_state=SEED))
        if best is None or sil[k]>sil[best[0]]: best=(k,l)
    return best[0],best[1],sil
def prf(y,p,classes):
    r={}
    for c in classes:
        tp=int(((p==c)&(y==c)).sum());r[c]=dict(precision=tp/max((p==c).sum(),1),recall=tp/max((y==c).sum(),1),n_test=int((y==c).sum()))
    r["macro_recall"]=float(np.mean([r[c]["recall"] for c in classes]));r["accuracy"]=float((p==y).mean());return r
def mapping(cl,y):
    return {c:Counter(y[cl==c]).most_common(1)[0][0] for c in set(cl) if c!=-1}
def evalsplit(R,y,files,tr,te,classes,norm):
    Xa=filenorm(R,files) if norm else R   # per-file norm uses only that file's own events (unsupervised, no labels)
    sc=StandardScaler().fit(Xa[tr]);Xtr,Xte=sc.transform(Xa[tr]),sc.transform(Xa[te])
    k,ltr,_=bestk(Xtr);km=KMeans(k,random_state=SEED,n_init=10).fit(Xtr);mp=mapping(km.labels_,y[tr])
    pte=km.predict(Xte);pk=np.array([mp[c] for c in pte])
    h=HDBSCAN(min_cluster_size=10).fit_predict(Xtr);mh=mapping(h,y[tr])
    ph=KNeighborsClassifier(5).fit(Xtr,h).predict(Xte);cov=float((ph!=-1).mean());phl=np.array([mh.get(c,"NOISE") for c in ph])
    cv=ph!=-1
    maj=Counter(y[tr]).most_common(1)[0][0]
    return dict(n_train=len(tr),n_test=len(te),k=int(k),km_ARI_test=float(ARI(y[te],pte)),km_NMI_test=float(NMI(y[te],pte)),km_mapped=prf(y[te],pk,classes),
      hdb_train_clusters=int(len(set(h))-(-1 in h)),hdb_train_noise_frac=float((h==-1).mean()),hdb_test_coverage=cov,
      hdb_ARI_test_covered=float(ARI(y[te][cv],ph[cv])) if cv.sum()>1 else None,hdb_mapped_covered=prf(y[te][cv],phl[cv],classes) if cv.any() else None,
      majority_class=maj,majority_baseline=prf(y[te],np.array([maj]*len(te)),classes))
res={}
for level in ("species","ecotype"):
    if level=="species": I=[i for i,e in enumerate(ev) if e["sp"] in("HW","KW")];y=np.array([ev[i]["sp"] for i in I])
    else: I=[i for i,e in enumerate(ev) if e["eco"] in("NRKW","TKW","OKW")];y=np.array([ev[i]["eco"] for i in I])
    classes=sorted(set(y));files=np.array([ev[i]["file"] for i in I]);days=np.array([ev[i]["day"] for i in I]);deps=np.array([ev[i]["dep"] for i in I])
    V={"0_trivial_duration_peakfreq":(A(["duration_s","peak_freq_median_hz"],I),False),"1_baseline":(A(BASE,I),False),
       "2_plus_deltas":(A(BASE+EXT,I),False),"3_baseline_filenorm":(A(BASE,I),True),"4_deltas_filenorm":(A(BASE+EXT,I),True)}
    res[level]=dict(n=len(I),counts=dict(Counter(y)),per_deployment={d:dict(Counter(y[deps==d])) for d in set(deps)},variants={})
    for name,(R,norm) in V.items():
        X=StandardScaler().fit_transform(filenorm(R,files) if norm else R);k,l,sil=bestk(X);hd=HDBSCAN(min_cluster_size=10).fit_predict(X);nn=hd>=0
        r=dict(n_features=R.shape[1],full=dict(k=int(k),silhouette=sil[k],ARI_label=ARI(y,l),NMI_label=NMI(y,l),ARI_file=ARI(files,l),ARI_day=ARI(days,l),ARI_deployment=ARI(deps,l),
            NMI_file=NMI(files,l),NMI_deployment=NMI(deps,l),
            hdb_clusters=int(len(set(hd))-(-1 in hd)),hdb_coverage=float(nn.mean()),hdb_ARI_label=ARI(y[nn],hd[nn]) if nn.sum()>1 else None,
            hdb_ARI_file=ARI(files[nn],hd[nn]) if nn.sum()>1 else None,hdb_ARI_deployment=ARI(deps[nn],hd[nn]) if nn.sum()>1 else None))
        gk=GroupKFold(n_splits=5);folds=[evalsplit(R,y,files,tr,te,classes,norm) for tr,te in gk.split(R,y,days)]
        r["holdout_day_5fold"]=dict(folds=folds,mean_km_ARI=float(np.mean([f["km_ARI_test"] for f in folds])),
            mean_km_macro_recall=float(np.mean([f["km_mapped"]["macro_recall"] for f in folds])),mean_km_acc=float(np.mean([f["km_mapped"]["accuracy"] for f in folds])),
            mean_majority_acc=float(np.mean([f["majority_baseline"]["accuracy"] for f in folds])),mean_hdb_coverage=float(np.mean([f["hdb_test_coverage"] for f in folds])),
            mean_hdb_macro_recall_covered=float(np.mean([f["hdb_mapped_covered"]["macro_recall"] for f in folds if f["hdb_mapped_covered"]])))
        for a,b in (("NorthBc","WVanIsl"),("WVanIsl","NorthBc")):
            r[f"train_{a}_test_{b}"]=evalsplit(R,y,files,np.where(deps==a)[0],np.where(deps==b)[0],classes,norm)
        res[level]["variants"][name]=r
        print(level,name,"fullARI",round(r["full"]["ARI_label"],3),"file",round(r["full"]["ARI_file"],3),"dep",round(r["full"]["ARI_deployment"],3),
              "dayCV ARI",round(r["holdout_day_5fold"]["mean_km_ARI"],3),"macroR",round(r["holdout_day_5fold"]["mean_km_macro_recall"],3),flush=True)
json.dump(dict(n_events=len(ev),label_counts=dict(Counter(e["label"] for e in ev)),results=res,runtime_s=time.time()-t0,seed=SEED,sr=SR,
  note="Clusters are acoustic pattern groups, not meaning. Event boxes from human annotation (time bounds only)."),open(f"{out}/results.json","w"),indent=1,default=float)
